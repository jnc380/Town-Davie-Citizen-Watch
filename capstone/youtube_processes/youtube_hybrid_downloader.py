#!/usr/bin/env python3
"""
Hybrid YouTube Downloader: Official API for discovery + Unofficial API for transcripts

UPDATED: More aggressive timing, only save info files with transcripts, stop on IP blocking
"""

import os
import sys
import json
import argparse
import time
import random
from datetime import datetime, date
from typing import List, Dict, Optional
from pathlib import Path
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from dotenv import load_dotenv

load_dotenv()

CHANNEL_NAME = "Town of Davie TV"
MEETING_KEYWORDS = ["council meeting", "budget", "assessment", "meeting"]
MIN_DATE = date(2024, 1, 1)
YOUTUBE_API_KEY = "AIzaSyBOLBCZLnxTZb0DXjqJoHZvqHwRu8o2lk0"

def matches_keywords(title: str, keywords: List[str]) -> bool:
    title_lower = title.lower()
    return any(keyword.lower() in title_lower for keyword in keywords)

class YouTubeHybridDownloader:
    def __init__(self, output_dir: str = "../downloads/town_meetings_youtube", max_videos: int = 200, transcript_delay: int = 30):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.max_videos = max_videos
        self.transcript_delay = transcript_delay  # More aggressive: 30s instead of 300s
        self.tracking_file = self.output_dir / "last_downloaded.json"
        self.youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        self.channel_id = "UClAfziZQ5Gra8DrGRzyq37w"

    def get_channel_videos(self) -> List[Dict]:
        print(f"📺 Getting videos from {CHANNEL_NAME} using YouTube Data API...")
        videos = []
        
        try:
            channels_response = self.youtube.channels().list(
                id=self.channel_id,
                part='contentDetails'
            ).execute()
            
            if not channels_response['items']:
                print("❌ Error: Channel not found")
                return videos
            
            uploads_playlist_id = channels_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
            next_page_token = None
            total_videos_checked = 0
            
            while True:
                playlist_response = self.youtube.playlistItems().list(
                    playlistId=uploads_playlist_id,
                    part='snippet,contentDetails',
                    maxResults=50,
                    pageToken=next_page_token
                ).execute()
                
                for item in playlist_response['items']:
                    total_videos_checked += 1
                    snippet = item['snippet']
                    title = snippet.get('title', '')
                    published_at = snippet.get('publishedAt', '')
                    video_id = snippet.get('resourceId', {}).get('videoId', '')
                    
                    if published_at:
                        try:
                            video_date = datetime.strptime(published_at[:10], "%Y-%m-%d").date()
                        except:
                            video_date = None
                    else:
                        video_date = None
                    
                    if total_videos_checked <= 20:
                        print(f"  📹 {title} ({published_at[:10] if published_at else 'Unknown'})")
                    
                    if matches_keywords(title, MEETING_KEYWORDS):
                        if video_date and video_date >= MIN_DATE:
                            video_info = {
                                'title': title,
                                'url': f'https://www.youtube.com/watch?v={video_id}',
                                'upload_date': published_at[:10] if published_at else '',
                                'description': snippet.get('description', ''),
                                'channel_name': CHANNEL_NAME,
                                'id': video_id,
                                'thumbnail': snippet.get('thumbnails', {}).get('default', {}).get('url', ''),
                                'uploader': CHANNEL_NAME,
                                'webpage_url': f'https://www.youtube.com/watch?v={video_id}',
                                'tags': [],
                            }
                            videos.append(video_info)
                            print(f"  ✅ MATCH: {title}")
                        else:
                            print(f"  ⏰ SKIP (date): {title} ({published_at[:10] if published_at else 'Unknown'})")
                    else:
                        if total_videos_checked <= 20:
                            print(f"  ❌ NO MATCH: {title}")
                
                next_page_token = playlist_response.get('nextPageToken')
                if not next_page_token or total_videos_checked >= self.max_videos:
                    break
                
                time.sleep(1)
            
        except HttpError as e:
            print(f"❌ Error getting videos: {e}")
        
        print(f"✅ Found {len(videos)} meeting videos from {total_videos_checked} total videos checked")
        return videos

    def fetch_transcript_unofficial(self, video_id: str) -> Optional[List[Dict]]:
        try:
            transcript = YouTubeTranscriptApi().fetch(video_id)
            transcript_list = []
            for entry in transcript:
                transcript_list.append({
                    'text': entry.text,
                    'start': entry.start,
                    'duration': entry.duration
                })
            print(f"✅ Successfully fetched transcript for {video_id}")
            return transcript_list
        except (TranscriptsDisabled, NoTranscriptFound):
            print(f"⚠️ No transcript available for video: {video_id}")
            return None
        except Exception as e:
            error_msg = str(e).lower()
            if "ip" in error_msg or "block" in error_msg or "blocking" in error_msg:
                print(f"🚫 IP BLOCKING DETECTED for {video_id}: {e}")
                print("🛑 Stopping script due to IP blocking...")
                sys.exit(1)  # Exit immediately on IP blocking
            else:
                print(f"❌ Error fetching transcript for {video_id}: {e}")
                return None

    def save_transcript(self, transcript: List[Dict], video_title: str):
        safe_title = "".join(c for c in video_title if c.isalnum() or c in (' ', '-', '_')).rstrip().replace(' ', '_')
        transcript_file = self.output_dir / f"{safe_title}_transcript.json"
        with open(transcript_file, 'w', encoding='utf-8') as f:
            json.dump(transcript, f, indent=2, ensure_ascii=False)
        print(f"💾 Transcript saved to: {transcript_file}")

    def save_metadata(self, video_info: Dict):
        safe_title = "".join(c for c in video_info['title'] if c.isalnum() or c in (' ', '-', '_')).rstrip().replace(' ', '_')
        metadata_file = self.output_dir / f"{safe_title}.info.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(video_info, f, indent=2, ensure_ascii=False)
        print(f"💾 Metadata saved to: {metadata_file}")

    def needs_transcript(self, video_title: str) -> bool:
        safe_title = "".join(c for c in video_title if c.isalnum() or c in (' ', '-', '_')).rstrip().replace(' ', '_')
        transcript_file = self.output_dir / f"{safe_title}_transcript.json"
        return not transcript_file.exists()

    def process_videos(self):
        videos = self.get_channel_videos()
        
        if not videos:
            print("❌ No videos found matching criteria")
            return
        
        print(f"🔍 {len(videos)} videos to process.")
        
        processed_videos = set()
        for metadata_file in self.output_dir.glob("*.info.json"):
            video_title = metadata_file.stem.replace(".info", "")
            processed_videos.add(video_title)
        
        new_videos = []
        for video in videos:
            safe_title = "".join(c for c in video['title'] if c.isalnum() or c in (' ', '-', '_')).rstrip().replace(' ', '_')
            if safe_title not in processed_videos:
                new_videos.append(video)
            else:
                print(f"⏭️ Skipping already processed: {video['title']}")
        
        print(f"📥 Processing {len(new_videos)} new videos")
        
        # NEW APPROACH: Only process videos that need transcripts
        videos_needing_transcripts = []
        for video in new_videos:
            if self.needs_transcript(video['title']):
                videos_needing_transcripts.append(video)
        
        print(f"📝 {len(videos_needing_transcripts)} videos need transcripts")
        
        # Process transcripts first, only save metadata if transcript succeeds
        if videos_needing_transcripts:
            print(f"\n🔄 Starting transcript processing for {len(videos_needing_transcripts)} videos...")
            print(f"⏰ Using {self.transcript_delay}s delay between transcript requests")
            
            successful_count = 0
            for i, video in enumerate(videos_needing_transcripts, 1):
                print(f"\n[{i}/{len(videos_needing_transcripts)}] Processing transcript: {video['title']}")
                
                transcript = self.fetch_transcript_unofficial(video['id'])
                if transcript:
                    # Only save metadata if transcript was successful
                    self.save_transcript(transcript, video['title'])
                    self.save_metadata(video)  # Save metadata only after successful transcript
                    successful_count += 1
                    print(f"✅ SUCCESS: Saved transcript and metadata for: {video['title']}")
                else:
                    print(f"⚠️ No transcript available for: {video['title']} (skipping metadata save)")
                
                if i < len(videos_needing_transcripts):
                    print(f"⏸️ Waiting {self.transcript_delay}s before next transcript request...")
                    time.sleep(self.transcript_delay)
            
            print(f"\n🎉 Successfully processed {successful_count} videos with transcripts!")
            print(f"📊 Total videos in database: {len(videos)}")
        else:
            print("✅ All videos already have transcripts!")

def main():
    parser = argparse.ArgumentParser(description="Hybrid YouTube downloader: Official API for discovery + Unofficial API for transcripts.")
    parser.add_argument('--max-videos', type=int, default=200, help='Maximum number of videos to process (default: 200)')
    parser.add_argument('--output-dir', default='../downloads/town_meetings_youtube', help='Output directory for downloads and transcripts (default: ../downloads/town_meetings_youtube)')
    parser.add_argument('--transcript-delay', type=int, default=30, help='Delay between transcript requests in seconds (default: 30 = 30 seconds)')
    args = parser.parse_args()
    
    downloader = YouTubeHybridDownloader(
        output_dir=args.output_dir, 
        max_videos=args.max_videos,
        transcript_delay=args.transcript_delay
    )
    downloader.process_videos()

if __name__ == "__main__":
    main()
