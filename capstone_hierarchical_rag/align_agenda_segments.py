#!/usr/bin/env python3
import os
import re
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from pymilvus import connections, Collection, utility
from neo4j import GraphDatabase
from openai import AsyncOpenAI

# Load env
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

EMBED_DIM = 1536
SPARSE_DIM = 1000

@dataclass
class TranscriptChunk:
    content: str
    start_time: float
    duration: float
    metadata: Dict[str, Any]

@dataclass
class AgendaItem:
    item_id: str
    title: str
    description: str
    url: Optional[str]
    order: int

class AgendaTranscriptAligner:
    def __init__(self) -> None:
        self.milvus_uri = os.getenv("MILVUS_URI", "")
        self.milvus_token = os.getenv("MILVUS_TOKEN", "")
        self.collection_name = os.getenv("MILVUS_COLLECTION", "TOWN_OF_DAVIE_RAG")
        self.neo4j_uri = os.getenv("NEO4J_URI", "")
        self.neo4j_username = os.getenv("NEO4J_USERNAME", "")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required")
        # Connect
        connections.connect(alias="default", uri=self.milvus_uri, token=self.milvus_token)
        if not utility.has_collection(self.collection_name):
            raise RuntimeError(f"Milvus collection not found: {self.collection_name}")
        self.col = Collection(self.collection_name)
        self.col.load()
        self.driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_username, self.neo4j_password))
        self.openai = AsyncOpenAI(api_key=self.openai_api_key)

    async def get_embedding(self, text: str) -> List[float]:
        resp = await self.openai.embeddings.create(model="text-embedding-3-small", input=text)
        return resp.data[0].embedding

    def _candidate_indices(self, item: AgendaItem, transcripts: List[TranscriptChunk]) -> List[int]:
        """Return candidate transcript indices for this agenda item ordered by start_time.
        Uses a moderate threshold and falls back to top-N scores if too few pass.
        """
        title = item.title or ""
        desc = item.description or ""
        key_terms = self._terms(f"{title} {desc}")
        amounts = self._amounts(f"{title} {desc}")
        vendor = self._vendor(title, desc)
        scores: List[float] = []
        for ch in transcripts:
            scores.append(self._score((ch.content or "").lower(), key_terms, vendor, amounts))
        threshold = 1.5
        cands = [i for i, sc in enumerate(scores) if sc >= threshold]
        if not cands:
            # fallback to top 3 by score
            ranked = sorted(range(len(transcripts)), key=lambda i: scores[i], reverse=True)
            cands = ranked[:3]
        # order by start time ascending
        cands = sorted(cands, key=lambda i: transcripts[i].start_time)
        return cands

    def _stopwords(self) -> set:
        return {"the","and","for","with","from","that","this","what","when","where","who","which","did","were","was","are","is","to","of","in","on","at","a","an","be"}

    def _amounts(self, text: str) -> List[str]:
        pats = [
            r"\$\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?",
            r"\$\d+(?:\.\d+)?\s*(?:m|million|b|billion)\b",
            r"\b\d+(?:\.\d+)?%\b",
        ]
        out: List[str] = []
        seen = set()
        for p in pats:
            for m in re.findall(p, text, flags=re.IGNORECASE):
                k = m.lower()
                if k not in seen:
                    seen.add(k)
                    out.append(m)
        return out

    def _vendor(self, title: str, desc: str) -> Optional[str]:
        text = f"{title} {desc}".strip()
        pats = [
            r"(?:agreement|contract|services|purchase|psa|engagement)\s+with\s+([A-Z][^,;\n]{2,60})",
            r"award\s+to\s+([A-Z][^,;\n]{2,60})",
            r"with\s+([A-Z][A-Za-z&\.'\- ]{2,60})\s+(?:for|to)",
        ]
        for p in pats:
            m = re.search(p, text, flags=re.IGNORECASE)
            if m:
                name = m.group(1).strip()
                if len(name) > 2:
                    return name
        return None

    def _terms(self, text: str) -> List[str]:
        toks = re.findall(r"[a-zA-Z0-9]+", (text or "").lower())
        return [t for t in toks if len(t) > 2 and t not in self._stopwords()]

    def _score(self, text_l: str, key_terms: List[str], vendor: Optional[str], amounts: List[str]) -> float:
        s = 0.0
        for t in key_terms:
            if t in text_l:
                s += 1.0
        if vendor and vendor.lower() in text_l:
            s += 2.0
        for a in amounts:
            if a.lower() in text_l:
                s += 1.2
        for kw in ("motion","approve","approved","adopt","adopted","award","awarded","contract"):
            if kw in text_l:
                s += 0.5
                break
        return s

    def _fetch_transcripts(self, meeting_id: str) -> List[TranscriptChunk]:
        expr = f"meeting_id == '{meeting_id}' and chunk_type == 'youtube_transcript'"
        fields = ["content","start_time","duration","metadata"]
        rows = self.col.query(expr=expr, output_fields=fields, timeout=60)
        chunks: List[TranscriptChunk] = []
        for r in rows:
            try:
                meta = r.get("metadata") or {}
                if isinstance(meta, str):
                    meta = json.loads(meta)
                chunks.append(TranscriptChunk(
                    content=r.get("content") or "",
                    start_time=float(r.get("start_time") or 0.0),
                    duration=float(r.get("duration") or 0.0),
                    metadata=meta,
                ))
            except Exception:
                continue
        chunks.sort(key=lambda c: c.start_time)
        return chunks

    def _fetch_agenda_items(self, meeting_id: str) -> List[AgendaItem]:
        # Be robust to graphs where Meeting uses id instead of meeting_id
        q = (
            "MATCH (m:Meeting)-[:HAS_AGENDA_ITEM]->(ai:AgendaItem) "
            "WHERE coalesce(m.meeting_id, m.id) = $mid "
            "RETURN ai.item_id as item_id, ai.title as title, ai.description as description, ai.url as url, coalesce(ai.order,0) as ord "
            "ORDER BY ord ASC"
        )
        out: List[AgendaItem] = []
        with self.driver.session() as s:
            for rec in s.run(q, {"mid": meeting_id}):
                out.append(AgendaItem(
                    item_id=str(rec.get("item_id") or ""),
                    title=(rec.get("title") or ""),
                    description=(rec.get("description") or ""),
                    url=(rec.get("url") or None),
                    order=int(rec.get("ord") or 0),
                ))
        return out

    def _align_item(self, item: AgendaItem, transcripts: List[TranscriptChunk]) -> Tuple[float, float, float, List[int], str]:
        title = item.title or ""
        desc = item.description or ""
        key_terms = self._terms(f"{title} {desc}")
        amounts = self._amounts(f"{title} {desc}")
        vendor = self._vendor(title, desc)
        if not transcripts:
            return 0.0, 0.0, 0.0, [], "no_transcripts"
        # Score each chunk
        scores: List[Tuple[int, float]] = []
        for idx, ch in enumerate(transcripts):
            sc = self._score((ch.content or "").lower(), key_terms, vendor, amounts)
            scores.append((idx, sc))
        # Pick earliest with score threshold
        threshold = 2.0
        candidate_indices = [i for i, sc in scores if sc >= threshold]
        if not candidate_indices:
            # pick best overall
            best_idx, best_sc = max(scores, key=lambda x: x[1])
            if best_sc <= 0.5:
                return 0.0, 0.0, 0.2, [], "low_confidence"
            start_idx = best_idx
            end_idx = best_idx
            method = "best_only"
        else:
            start_idx = min(candidate_indices)
            # extend forward while score stays reasonable and within max window
            max_span_s = 8 * 60.0
            end_idx = start_idx
            start_s = transcripts[start_idx].start_time
            for j in range(start_idx + 1, len(transcripts)):
                if transcripts[j].start_time - start_s > max_span_s:
                    break
                if scores[j][1] >= 1.0:
                    end_idx = j
                else:
                    break
            method = "threshold_window"
        # Compute times
        start_time = transcripts[start_idx].start_time
        last = transcripts[end_idx]
        end_time = last.start_time + max(0.0, last.duration or 0.0)
        # Confidence heuristic
        conf = 0.8 if method == "threshold_window" else 0.6
        return start_time, end_time, conf, list(range(start_idx, end_idx + 1)), method

    async def _embed_and_insert(self, meeting_id: str, meeting_date: str, meeting_type: str, item: AgendaItem,
                                transcripts: List[TranscriptChunk], start_time: float, end_time: float,
                                confidence: float, method: str) -> None:
        # Build content
        parts: List[str] = []
        for ch in transcripts:
            if ch.start_time >= start_time and ch.start_time < end_time + 0.01:
                parts.append(ch.content.strip())
        text = " ".join(parts)
        emb = await self.get_embedding(text)
        # Prepare row
        chunk_id = f"{meeting_id}_agenda_segment_{item.item_id}"
        contents = [text]
        chunk_ids = [chunk_id]
        meeting_ids = [meeting_id]
        meeting_types = [meeting_type]
        meeting_dates = [meeting_date]
        chunk_types = ["youtube_agenda_segment"]
        start_times = [float(start_time)]
        durations = [float(max(0.0, end_time - start_time))]
        metadata = [{
            "source": "transcript_segment",
            "item_id": item.item_id,
            "url": item.url or "",
            "segment_method": method,
            "confidence": confidence,
            "segment_reason": "Agenda-aligned segment based on title/description/vendor/amounts overlap"
        }]
        dense_vecs = [emb]
        sparse_vecs = [[0.0] * SPARSE_DIM]
        rows = [
            contents, chunk_ids, meeting_ids, meeting_types, meeting_dates,
            chunk_types, start_times, durations, metadata, dense_vecs, sparse_vecs
        ]
        self.col.insert(rows)
        try:
            self.col.flush()
        except Exception:
            pass

    async def align_meeting(self, meeting_id: str, meeting_date: str, meeting_type: str) -> None:
        transcripts = self._fetch_transcripts(meeting_id)
        items = self._fetch_agenda_items(meeting_id)
        if not transcripts or not items:
            return
        # Enforce monotonic, non-overlapping assignment with minimal gaps
        last_end = -30.0
        min_gap = 60.0   # require at least 60s separation between starts
        max_span_s = 10 * 60.0
        motion_rx = re.compile(r"\b(motion|second|approve|approved|vote|passes|passed|roll\s*call)\b", re.IGNORECASE)
        for it in items:
            cands = self._candidate_indices(it, transcripts)
            # choose first candidate that starts after last_end + min_gap; else use best
            chosen_idx = None
            for i in cands:
                if transcripts[i].start_time >= last_end + min_gap:
                    chosen_idx = i
                    break
            if chosen_idx is None:
                # fallback: pick first transcript chunk after the constraint
                target = last_end + min_gap
                fallback_idx = None
                for k, ch in enumerate(transcripts):
                    if ch.start_time >= target:
                        fallback_idx = k
                        break
                if fallback_idx is not None:
                    chosen_idx = fallback_idx
                elif cands:
                    # as a last resort pick the latest candidate to keep moving forward
                    chosen_idx = cands[-1]
                else:
                    continue
            start_time = transcripts[chosen_idx].start_time
            # extend end time: until motion/vote cue or score drop, capped by max_span
            title = it.title or ""
            desc = it.description or ""
            key_terms = self._terms(f"{title} {desc}")
            amounts = self._amounts(f"{title} {desc}")
            vendor = self._vendor(title, desc)
            def sc_for(j: int) -> float:
                return self._score((transcripts[j].content or "").lower(), key_terms, vendor, amounts)
            end_idx = chosen_idx
            base_score = sc_for(chosen_idx)
            for j in range(chosen_idx + 1, len(transcripts)):
                if transcripts[j].start_time - start_time > max_span_s:
                    break
                text_l = (transcripts[j].content or "").lower()
                if motion_rx.search(text_l):
                    end_idx = j
                    break
                if sc_for(j) >= max(0.8, base_score * 0.5):
                    end_idx = j
                else:
                    # allow one weak chunk, then stop
                    k = j + 1
                    if k < len(transcripts) and sc_for(k) >= 0.8:
                        end_idx = k
                        continue
                    break
            end_time = transcripts[end_idx].start_time + max(0.0, transcripts[end_idx].duration or 0.0)
            if end_time <= start_time:
                end_time = start_time + min(240.0, transcripts[chosen_idx].duration or 120.0)
            conf = 0.7 if base_score >= 2.0 else 0.6
            await self._embed_and_insert(meeting_id, meeting_date, meeting_type, it, transcripts, start_time, end_time, conf, "monotonic_window")
            last_end = end_time

async def _main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Align agenda items to transcript segments and upsert segments to Milvus")
    ap.add_argument("--meeting-id", required=True, help="Meeting ID to align")
    ap.add_argument("--meeting-date", default="", help="Meeting date string (YYYY-MM-DD)")
    ap.add_argument("--meeting-type", default="Regular Council Meeting", help="Meeting type")
    args = ap.parse_args()
    aln = AgendaTranscriptAligner()
    await aln.align_meeting(args.meeting_id, args.meeting_date, args.meeting_type)
    print("Aligned and inserted segments for", args.meeting_id)

if __name__ == "__main__":
    asyncio.run(_main()) 