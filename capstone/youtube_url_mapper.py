#!/usr/bin/env python3
"""
YouTube URL Mapper for Capstone Project
Maps meeting IDs to actual YouTube URLs from downloaded metadata
"""

import os
import json
from typing import Dict, Optional
from pathlib import Path

class YouTubeURLMapper:
    """Maps meeting IDs to YouTube URLs with metadata"""
    
    def __init__(self):
        """Initialize the URL mapper"""
        base = Path(__file__).resolve().parent
        default_dir = base / "downloads" / "town_meetings_youtube"
        self.data_dir = Path(os.getenv("YOUTUBE_DOWNLOAD_DIR", str(default_dir)))
        self.url_mapping = {}
        self._load_url_mapping()
        self._load_external_mapping()
    
    def _load_url_mapping(self):
        """Load YouTube URLs from downloaded metadata files"""
        if not self.data_dir.exists():
            print(f"Warning: YouTube data directory not found: {self.data_dir}")
            return
        
        # Find all info files
        info_files = list(self.data_dir.glob("*.info.json"))
        
        for info_file in info_files:
            try:
                with open(info_file, 'r', encoding='utf-8') as f:
                    info_data = json.load(f)
                
                # Extract meeting ID from filename
                meeting_id = info_file.stem.replace(".info", "")
                
                # Store URL mapping with robust fallbacks
                url = info_data.get("url") or info_data.get("webpage_url") or ""
                if not url:
                    vid = info_data.get("id")
                    if vid:
                        url = f"https://www.youtube.com/watch?v={vid}"
                self.url_mapping[meeting_id] = {
                    "url": url,
                    "title": info_data.get("title", ""),
                    "upload_date": info_data.get("upload_date", ""),
                    "description": info_data.get("description", "")
                }
                
                print(f"✅ Loaded URL mapping for: {meeting_id}")
                
            except Exception as e:
                print(f"❌ Failed to load URL mapping for {info_file}: {e}")

    def _load_external_mapping(self):
        """Optionally load mapping from an external JSON file path provided via env."""
        path = os.getenv("YOUTUBE_URL_MAP_JSON")
        if not path:
            return
        try:
            p = Path(path)
            if p.exists():
                with p.open('r', encoding='utf-8') as f:
                    data = json.load(f)
                # Expect format: { meeting_id: {"url": "...", ...}, ... } or { meeting_id: "url" }
                for mid, val in (data or {}).items():
                    if isinstance(val, dict):
                        url = val.get("url") or ""
                        meta = {"url": url, "title": val.get("title", ""), "upload_date": val.get("upload_date", ""), "description": val.get("description", "")}
                    else:
                        url = str(val)
                        meta = {"url": url, "title": "", "upload_date": "", "description": ""}
                    if url:
                        self.url_mapping[str(mid)] = meta
                print(f"✅ Merged external YouTube URL mapping from {p}")
        except Exception as e:
            print(f"❌ Failed to load external YouTube URL mapping: {e}")
    
    def get_youtube_url(self, meeting_id: str) -> Optional[str]:
        """Get YouTube URL for a meeting ID"""
        if meeting_id in self.url_mapping:
            return self.url_mapping[meeting_id]["url"]
        return None
    
    def get_meeting_info(self, meeting_id: str) -> Optional[Dict]:
        """Get complete meeting information"""
        return self.url_mapping.get(meeting_id)
    
    def get_all_meetings(self) -> Dict[str, Dict]:
        """Get all meeting information"""
        return self.url_mapping
    
    def get_meeting_count(self) -> int:
        """Get total number of meetings with URLs"""
        return len(self.url_mapping)

# Global mapper instance
youtube_mapper = YouTubeURLMapper()

def get_youtube_url(meeting_id: str) -> Optional[str]:
    """Global function to get YouTube URL"""
    return youtube_mapper.get_youtube_url(meeting_id)

def get_meeting_info(meeting_id: str) -> Optional[Dict]:
    """Global function to get meeting info"""
    return youtube_mapper.get_meeting_info(meeting_id)

if __name__ == "__main__":
    # Test the URL mapper
    print(f"📊 Loaded {youtube_mapper.get_meeting_count()} YouTube URL mappings")
    
    # Show first few mappings
    meetings = youtube_mapper.get_all_meetings()
    for i, (meeting_id, info) in enumerate(list(meetings.items())[:3]):
        print(f"\n{i+1}. {meeting_id}")
        print(f"   Title: {info.get('title', 'N/A')}")
        print(f"   URL: {info.get('url', 'N/A')}")
        print(f"   Date: {info.get('upload_date', 'N/A')}") 