#!/usr/bin/env python3
"""
Experimental LLM-based alignment (TEST ONLY)
- Reads transcript chunks from Milvus for one meeting
- Reads agenda items from Neo4j for that meeting
- Builds timestamped transcript windows
- Asks GPT to align each agenda item to a start/end window (JSON)
- Validates, clamps, and prints 5 sample links for manual verification

No writes to Milvus/Neo4j. This is for evaluation only.
"""
import os
import json
import math
import argparse
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

from dotenv import load_dotenv
from pymilvus import connections, Collection
from neo4j import GraphDatabase

try:
    # Prefer synchronous client to keep script simple
    from openai import OpenAI
except Exception:  # fallback name in older sdists
    from openai import OpenAI  # type: ignore

# Simple token utilities for scoring
STOP = {"the","and","for","with","from","that","this","what","when","where","who","which","did","were","was","are","is","to","of","in","on","at","a","an","be"}
def _terms(text: str) -> List[str]:
    import re
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 2 and t not in STOP]

# Score a transcript window against an agenda item by simple token overlap and cue words
def score_window(item_title: str, item_desc: str, win_text: str) -> float:
    item_tokens = set(_terms(item_title + " " + (item_desc or "")))
    if not item_tokens:
        return 0.0
    wtokens = set(_terms(win_text))
    overlap = len(item_tokens & wtokens)
    cues = 0
    cues += 1 if any(k in wtokens for k in ("motion","approve","approved","vote","voted","award","contract")) else 0
    return float(overlap + cues)

# Try to use mapper if present for better YouTube base URLs
def _resolve_base_url(meeting_id: str, neo4j_drv) -> str:
    try:
        import sys
        sys.path.append(str(Path(__file__).resolve().parent))
        from youtube_url_mapper import get_youtube_url  # type: ignore
        u = get_youtube_url(meeting_id)
        if u:
            return u
    except Exception:
        pass
    # Neo4j fallback
    try:
        with neo4j_drv.session() as s:
            rec = s.run("MATCH (m:Meeting {meeting_id:$mid}) RETURN m.url as url", {"mid": meeting_id}).single()
            if rec and rec.get("url"):
                return rec.get("url")
    except Exception:
        pass
    return ""

@dataclass
class TranscriptChunk:
    start: float
    end: float
    text: str

@dataclass
class AgendaItem:
    item_id: str
    title: str
    description: str
    order: int
    url: Optional[str]


def fetch_transcript_chunks(col: Collection, meeting_id: str) -> List[TranscriptChunk]:
    # Milvus caps (offset+limit) around ~16384; fetch in batches
    batch = 4000
    offset = 0
    chunks: List[TranscriptChunk] = []
    while True:
        rows = col.query(
            expr=f"meeting_id == '{meeting_id}' and chunk_type == 'youtube_transcript'",
            output_fields=["content", "start_time", "duration"],
            limit=batch,
            offset=offset,
            timeout=60,
        )
        if not rows:
            break
        for r in rows:
            try:
                st = float(r.get("start_time") or 0.0)
                du = float(r.get("duration") or 0.0)
                text = (r.get("content") or "").strip()
                if du < 0:
                    du = 0.0
                chunks.append(TranscriptChunk(start=st, end=st + du, text=text))
            except Exception:
                continue
        if len(rows) < batch:
            break
        offset += batch
    chunks.sort(key=lambda c: c.start)
    return chunks


def fetch_agenda_items(drv, meeting_id: str, limit_items: Optional[int] = None) -> List[AgendaItem]:
    out: List[AgendaItem] = []
    with drv.session() as s:
        # 1) Direct by item property meeting_id
        cy1 = (
            "MATCH (ai:AgendaItem) WHERE ai.meeting_id = $mid "
            "RETURN ai.item_id as item_id, ai.title as title, ai.description as description, ai.url as url "
            "ORDER BY coalesce(ai.order,0) ASC"
        )
        rows = s.run(cy1, {"mid": meeting_id}).data()
        if not rows:
            # 2) Relationship via Meeting.id
            cy2 = (
                "MATCH (m:Meeting {id:$mid})-[:HAS_AGENDA_ITEM]->(ai:AgendaItem) "
                "RETURN ai.item_id as item_id, ai.title as title, ai.description as description, ai.url as url, coalesce(ai.order,0) as ord "
                "ORDER BY ord ASC"
            )
            rows = s.run(cy2, {"mid": meeting_id}).data()
        if not rows:
            # 3) Relationship via Meeting.meeting_id
            cy3 = (
                "MATCH (m:Meeting {meeting_id:$mid})-[:HAS_AGENDA_ITEM]->(ai:AgendaItem) "
                "RETURN ai.item_id as item_id, ai.title as title, ai.description as description, ai.url as url, coalesce(ai.order,0) as ord "
                "ORDER BY ord ASC"
            )
            rows = s.run(cy3, {"mid": meeting_id}).data()
        for rec in rows:
            out.append(
                AgendaItem(
                    item_id=str(rec.get("item_id") or ""),
                    title=(rec.get("title") or ""),
                    description=(rec.get("description") or ""),
                    url=(rec.get("url") or None),
                    order=int(rec.get("ord") or 0),
                )
            )
    if limit_items is not None:
        out = out[: max(0, int(limit_items))]
    return out


def build_windows(chunks: List[TranscriptChunk], window_sec: int, max_minutes: int) -> List[Dict[str, Any]]:
    if not chunks:
        return []
    total_end = chunks[-1].end
    cutoff = min(total_end, float(max_minutes) * 60.0) if max_minutes > 0 else total_end
    windows: List[Dict[str, Any]] = []
    ws = 0.0
    we = float(window_sec)
    idx = 0
    while ws < cutoff and idx < len(chunks):
        # collect texts overlapping [ws, we)
        texts: List[str] = []
        while idx < len(chunks) and chunks[idx].start < we:
            if chunks[idx].end > ws:
                texts.append(chunks[idx].text)
            idx += 1
        blob = " \n".join(t for t in texts if t)
        # truncate each window to keep tokens bounded
        if len(blob) > 2000:
            blob = blob[:2000] + "..."
        windows.append({"start_s": int(ws), "end_s": int(min(we, cutoff)), "text": blob})
        ws = we
        we = ws + float(window_sec)
    return windows


def call_llm(client: OpenAI, model: str, agenda: List[AgendaItem], windows: List[Dict[str, Any]], dump_base: Optional[str] = None) -> Dict[str, Any]:
    # Compact agenda for prompt
    agenda_payload = [
        {
            "item_id": a.item_id,
            "order": a.order,
            "title": a.title[:280],
            "description": (a.description or "")[:800],
        }
        for a in agenda
    ]
    payload = {
        "agenda_items": agenda_payload,
        "transcript_windows": windows,
        "instructions": (
            "Align each agenda item to a single contiguous time span [start_s,end_s] from the transcript. "
            "Respect agenda order (monotonic, non-overlapping as much as possible). Use cues like item titles/numbers, vendor names, amounts, 'motion', 'vote', 'approved', and the spoken text. "
            "Return high-confidence spans; if uncertain, set start_s and end_s to null and confidence low."
        ),
    }
    if dump_base:
        try:
            p = Path(dump_base)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(str(p) + ".input.json", "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    schema = {
        "name": "agenda_alignment",
        "schema": {
            "type": "object",
            "properties": {
                "segments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "item_id": {"type": "string"},
                            "title": {"type": "string"},
                            "start_s": {"type": ["integer", "null"], "minimum": 0},
                            "end_s": {"type": ["integer", "null"], "minimum": 0},
                            "confidence": {"type": "number"},
                            "evidence": {
                                "type": "object",
                                "properties": {
                                    "phrases": {"type": "array", "items": {"type": "string"}},
                                    "vote_detected": {"type": "boolean"},
                                    "motion_phrase": {"type": ["string", "null"]},
                                },
                                "additionalProperties": False,
                            },
                        },
                        "required": ["item_id", "title", "start_s", "end_s", "confidence"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["segments"],
            "additionalProperties": False,
        },
    }
    system = (
        "You are aligning a city council meeting transcript to agenda items. Return only valid JSON matching the schema."
    )
    if dump_base:
        try:
            with open(str(Path(dump_base)) + ".system.txt", "w", encoding="utf-8") as f:
                f.write(system)
        except Exception:
            pass
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload)},
        ],
        temperature=0.1,
        max_tokens=1200,
        response_format={"type": "json_schema", "json_schema": schema},
    )
    content = resp.choices[0].message.content or "{}"
    if dump_base:
        try:
            with open(str(Path(dump_base)) + ".raw.txt", "w", encoding="utf-8") as f:
                f.write(content)
        except Exception:
            pass
    try:
        return json.loads(content)
    except Exception:
        # Try to extract JSON object substring
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            return json.loads(content[start : end + 1])
        return {"segments": []}


def clamp_and_validate(segments: List[Dict[str, Any]], windows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not windows:
        return segments
    min_s = 0
    max_s = int(max((w.get("end_s") or 0) for w in windows))
    # Sort by agenda order preserved in input; ensure monotonic non-overlap
    # We will enforce start >= last_end and end >= start
    last_end = 0
    out: List[Dict[str, Any]] = []
    for seg in segments:
        st = seg.get("start_s")
        en = seg.get("end_s")
        conf = float(seg.get("confidence") or 0.0)
        # If nulls or low confidence, keep nulls (for analysis), but still clamp if present
        if isinstance(st, int):
            st = max(min_s, min(int(st), max_s))
        if isinstance(en, int):
            en = max(min_s, min(int(en), max_s))
        if isinstance(st, int) and isinstance(en, int):
            if en < st:
                en = st + 30  # minimal 30s if inverted
            # Enforce monotonic start progression
            st2 = max(st, last_end)
            # Preserve at least 20s segment
            en2 = max(en, st2 + 20)
            st, en = st2, en2
            last_end = en
        seg["start_s"], seg["end_s"], seg["confidence"] = st, en, conf
        out.append(seg)
    return out


def print_samples(meeting_id: str, base_url: str, segments: List[Dict[str, Any]], limit: int = 5) -> None:
    print("\nSamples (up to 5):")
    shown = 0
    for seg in segments:
        if shown >= limit:
            break
        st = seg.get("start_s")
        en = seg.get("end_s")
        t = None
        if base_url and isinstance(st, int):
            if "youtu" in base_url.lower():
                sep = "&" if "?" in base_url else "?"
                t = f"{base_url}{sep}t={int(st)}s"
            else:
                t = base_url
        print(
            f"- item_id {seg.get('item_id')} | conf {seg.get('confidence')} | start {st} | end {en} | url {t or ''}"
        )
        shown += 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Experimental LLM agenda alignment (test only)")
    ap.add_argument("--meeting-id", required=True)
    ap.add_argument("--max-minutes", type=int, default=30, help="Cap transcript minutes sent to LLM")
    ap.add_argument("--window-sec", type=int, default=120, help="Window size for transcript aggregation")
    ap.add_argument("--limit-items", type=int, default=8, help="Max agenda items to align in one call")
    ap.add_argument("--model", default=os.getenv("LLM_ALIGNMENT_MODEL", "gpt-4o-mini"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", type=str, default=None, help="Optional path to write alignment JSON")
    ap.add_argument("--dump-base", type=str, default=None, help="Base path (without extension) to dump exact prompt/input/output")
    ap.add_argument("--per-item", action="store_true", help="Align one agenda item per GPT call using top-K windows for that item")
    ap.add_argument("--per-item-topk", type=int, default=8, help="How many top windows to send per agenda item")
    ap.add_argument("--truncate", type=int, default=1200, help="Max characters per window text sent to GPT")
    ap.add_argument("--segmented", action="store_true", help="Loop over time segments with overlap and align unresolved items per segment")
    ap.add_argument("--segment-minutes", type=int, default=30, help="Segment length in minutes")
    ap.add_argument("--segment-overlap", type=int, default=15, help="Segment overlap in minutes")
    args = ap.parse_args()

    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

    # Default dump/output directory to capstone/tmp if not provided
    tmp_dir = Path(__file__).resolve().parent / "tmp"
    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    mode_tag = "seg" if getattr(args, "segmented", False) else ("per_item" if getattr(args, "per_item", False) else "full")
    dump_base = args.dump_base or str(tmp_dir / f"{args.meeting_id}_{mode_tag}")
    out_path = args.out or str(tmp_dir / f"{args.meeting_id}_alignment.json")

    # Milvus
    connections.connect(
        alias="default",
        uri=os.getenv("MILVUS_URI", ""),
        token=os.getenv("MILVUS_TOKEN", ""),
    )
    col = Collection(os.getenv("MILVUS_COLLECTION", "TOWN_OF_DAVIE_RAG"))
    col.load()

    # Neo4j
    drv = GraphDatabase.driver(
        os.getenv("NEO4J_URI", ""),
        auth=(os.getenv("NEO4J_USERNAME", ""), os.getenv("NEO4J_PASSWORD", "")),
    )

    # Fetch data
    chunks = fetch_transcript_chunks(col, args.meeting_id)
    agenda = fetch_agenda_items(drv, args.meeting_id, limit_items=args.limit_items)
    if not chunks:
        print("No transcript chunks found for meeting.")
        return
    if not agenda:
        print("No agenda items found for meeting.")
        return

    # Build windows and optionally truncate text to keep prompts compact
    windows = build_windows(chunks, args.window_sec, args.max_minutes)
    trunc = max(200, int(args.truncate))
    for w in windows:
        t = w.get("text") or ""
        if len(t) > trunc:
            w["text"] = t[:trunc] + "..."
    base_url = _resolve_base_url(args.meeting_id, drv)
    print(f"Prepared windows: {len(windows)}; agenda items: {len(agenda)}; base_url: {base_url[:80] if base_url else ''}")

    if args.dry_run:
        # Show payload sizes only
        total_chars = sum(len(w.get("text") or "") for w in windows)
        print(json.dumps({
            "meeting_id": args.meeting_id,
            "windows": len(windows),
            "agenda_items": len(agenda),
            "transcript_chars": total_chars,
        }, indent=2))
        return

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
    segments: List[Dict[str, Any]] = []
    if args.segmented:
        # Segmented pass over time
        # Determine overall max time from windows
        if not windows:
            print("No windows to segment.")
            return
        max_s = int(max((w.get("end_s") or 0) for w in windows))
        seg_len = max(5, int(args.segment_minutes)) * 60
        seg_ovl = max(0, int(args.segment_overlap)) * 60
        seg_step = max(60, seg_len - seg_ovl)
        # Track unresolved agenda items by item_id
        unresolved = {a.item_id: a for a in agenda}
        t0 = 0
        seg_idx = 0
        while t0 < max_s and unresolved:
            t1 = min(max_s, t0 + seg_len)
            subwins = [w for w in windows if int(w.get("start_s") or 0) < t1 and int(w.get("end_s") or 0) > t0]
            # Tighten within [t0, t1]
            for w in subwins:
                w["start_s"] = max(int(w["start_s"]), t0)
                w["end_s"] = min(int(w["end_s"]), t1)
            if subwins:
                # Call LLM with unresolved items for this segment
                items_here = list(unresolved.values())
                dump = f"{dump_base}_seg_{seg_idx}" if dump_base else None
                al = call_llm(client, args.model, items_here, subwins, dump_base=dump)
                segs = al.get("segments") or []
                # Accept any with start/end and mark resolved
                for s in segs:
                    iid = str(s.get("item_id") or "")
                    if iid and s.get("start_s") is not None and s.get("end_s") is not None and iid in unresolved:
                        segments.append(s)
                        unresolved.pop(iid, None)
            t0 += seg_step
            seg_idx += 1
    elif args.per_item:
        # Align each item with its own top-K windows
        for idx, it in enumerate(agenda):
            # Score all windows for this item
            scored = [
                (score_window(it.title, it.description, w.get("text") or ""), w)
                for w in windows
            ]
            scored.sort(key=lambda x: x[0], reverse=True)
            top_k = [w for s, w in scored[: max(1, args.per_item_topk)] if s > 0]
            if not top_k:
                # fallback: earliest few windows
                top_k = windows[: max(1, args.per_item_topk)]
            # Call LLM for this single item
            dump = f"{dump_base}_item_{it.item_id}" if dump_base else None
            al = call_llm(client, args.model, [it], top_k, dump_base=dump)
            segs = al.get("segments") or []
            segments.extend(segs)
    else:
        alignment = call_llm(client, args.model, agenda, windows, dump_base=dump_base)
        segments = alignment.get("segments") or []
    # Post-validate and clamp
    segments = clamp_and_validate(segments, windows)

    result_obj = {"segments": segments[: min(12, len(segments))]}
    # Print detailed JSON for debugging
    print(json.dumps(result_obj, indent=2))
    if out_path:
        try:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result_obj, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"WARN: failed to write --out file: {e}")

    # Show 5 clickable samples
    print_samples(args.meeting_id, base_url, segments, limit=5)


if __name__ == "__main__":
    main() 