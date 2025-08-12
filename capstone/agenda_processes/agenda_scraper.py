#!/usr/bin/env python3
"""
Agenda Scraper for Town of Davie Council Meetings

Scrapes agenda pages from Novus Agenda system and downloads all PDF attachments.
"""

import os
import sys
import json
import argparse
import time
import requests
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import re

class AgendaScraper:
    def __init__(self, output_dir: str = "../downloads/agendas"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.base_url = "https://davie.novusagenda.com/agendapublic/"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })

    def extract_meeting_info(self, soup: BeautifulSoup) -> Dict:
        """Extract meeting metadata from the agenda page."""
        meeting_info = {
            'meeting_type': 'Unknown',
            'meeting_date': 'Unknown',
            'location': 'Unknown',
            'time': 'Unknown'
        }
        
        # Look for meeting details in the page
        text_content = soup.get_text()
        
        # Extract meeting type
        if 'REGULAR MEETING' in text_content:
            meeting_info['meeting_type'] = 'Regular Council Meeting'
        elif 'CRA' in text_content:
            meeting_info['meeting_type'] = 'CRA Meeting'
        elif 'WORKSHOP' in text_content:
            meeting_info['meeting_type'] = 'Workshop'
        elif 'SPECIAL' in text_content:
            meeting_info['meeting_type'] = 'Special Meeting'
        
        # Extract date (look for multiple date patterns)
        date_patterns = [
            r'(\w+ \d{1,2}, \d{4})',  # "January 15, 2025"
            r'(\d{1,2}/\d{1,2}/\d{4})',  # "1/15/2025"
            r'(\w+ \d{1,2} \d{4})',  # "January 15 2025"
            r'(\d{1,2}-\d{1,2}-\d{4})',  # "1-15-2025"
        ]
        
        for pattern in date_patterns:
            date_match = re.search(pattern, text_content)
            if date_match:
                meeting_info['meeting_date'] = date_match.group(1)
                break
        
        # If still no date found, look for date in page title or headers
        if meeting_info['meeting_date'] == 'Unknown':
            # Look in page title
            title = soup.find('title')
            if title:
                title_text = title.get_text()
                for pattern in date_patterns:
                    date_match = re.search(pattern, title_text)
                    if date_match:
                        meeting_info['meeting_date'] = date_match.group(1)
                        break
        
        # Extract location
        location_pattern = r'Location: ([^,]+)'
        location_match = re.search(location_pattern, text_content)
        if location_match:
            meeting_info['location'] = location_match.group(1).strip()
        
        # Extract time
        time_pattern = r'(\d{1,2}:\d{2} [AP]M)'
        time_match = re.search(time_pattern, text_content)
        if time_match:
            meeting_info['time'] = time_match.group(1)
        
        return meeting_info

    def extract_pdf_links(self, soup: BeautifulSoup, base_url: str) -> List[Dict]:
        """Extract all PDF links from the agenda page."""
        pdf_links = []
        
        # Find all links that contain CoverSheet.aspx
        links = soup.find_all('a', href=True)
        
        for link in links:
            href = link.get('href')
            if 'CoverSheet.aspx' in href:
                # Get the link text (agenda item description)
                link_text = link.get_text(strip=True)
                
                # Construct full URL
                full_url = urljoin(base_url, href)
                
                # Extract ItemID and MeetingID from URL
                item_id_match = re.search(r'ItemID=(\d+)', href)
                meeting_id_match = re.search(r'MeetingID=(\d+)', href)
                
                item_id = item_id_match.group(1) if item_id_match else 'unknown'
                meeting_id = meeting_id_match.group(1) if meeting_id_match else 'unknown'
                
                pdf_links.append({
                    'url': full_url,
                    'text': link_text,
                    'item_id': item_id,
                    'meeting_id': meeting_id,
                    'filename': f"item_{item_id}_{meeting_id}.pdf"
                })
        
        return pdf_links

    def extract_document_download_link(self, coversheet_url: str) -> Optional[str]:
        """Extract the main document download link from a CoverSheet page."""
        try:
            print(f"🔍 Following CoverSheet link: {coversheet_url}")
            response = self.session.get(coversheet_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for the MAIN document link (usually the first one)
            # Common patterns for main document links
            main_document_links = []
            
            # Look for links with document-related text or extensions
            for link in soup.find_all('a', href=True):
                href = link.get('href', '').lower()
                link_text = link.get_text().lower()
                
                # Check if it's a document link (PDF, DOCX, XLSX, PPTX, etc.)
                if ('.pdf' in href or 
                    '.docx' in href or 
                    '.xlsx' in href or
                    '.pptx' in href or
                    '.ppt' in href or
                    '.doc' in href or
                    '.xls' in href or
                    'attachmentviewer' in href or
                    'download' in link_text or
                    'document' in link_text):
                    full_url = urljoin(coversheet_url, link.get('href'))
                    main_document_links.append(full_url)
            
            # Also look for iframe sources that might contain documents
            for iframe in soup.find_all('iframe', src=True):
                src = iframe.get('src', '').lower()
                if any(ext in src for ext in ['.pdf', '.docx', '.xlsx', '.pptx', '.ppt']):
                    full_url = urljoin(coversheet_url, iframe.get('src'))
                    main_document_links.append(full_url)
            
            # Look for embedded document objects
            for obj in soup.find_all('object', data=True):
                data = obj.get('data', '').lower()
                if any(ext in data for ext in ['.pdf', '.docx', '.xlsx', '.pptx', '.ppt']):
                    full_url = urljoin(coversheet_url, obj.get('data'))
                    main_document_links.append(full_url)
            
            if main_document_links:
                print(f"✅ Found {len(main_document_links)} main document link(s)")
                return main_document_links[0]  # Return the first document link found
            else:
                print(f"⚠️ No main document links found in CoverSheet page")
                return None
                
        except Exception as e:
            print(f"❌ Error extracting document link from {coversheet_url}: {e}")
            return None

    def extract_nested_links(self, coversheet_url: str, item_id: str, meeting_id: str) -> List[Dict]:
        """Extract all nested document links from a CoverSheet page with metadata."""
        try:
            print(f"🔗 Extracting nested links from: {coversheet_url}")
            response = self.session.get(coversheet_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            nested_links = []
            
            # Look for all document links in the page
            for link in soup.find_all('a', href=True):
                href = link.get('href', '').lower()
                link_text = link.get_text().strip()
                
                # Check if it's a document link
                if ('.pdf' in href or 
                    '.docx' in href or 
                    '.xlsx' in href or
                    '.pptx' in href or
                    '.ppt' in href or
                    '.doc' in href or
                    '.xls' in href or
                    'attachmentviewer' in href):
                    
                    full_url = urljoin(coversheet_url, link.get('href'))
                    
                    # Determine file extension
                    if '.docx' in href:
                        file_ext = '.docx'
                    elif '.xlsx' in href:
                        file_ext = '.xlsx'
                    elif '.pptx' in href:
                        file_ext = '.pptx'
                    elif '.ppt' in href:
                        file_ext = '.ppt'
                    elif '.doc' in href:
                        file_ext = '.doc'
                    elif '.xls' in href:
                        file_ext = '.xls'
                    else:
                        file_ext = '.pdf'
                    
                    # Generate filename with parent item reference
                    filename = f"item_{item_id}_{meeting_id}_nested_{len(nested_links) + 1}{file_ext}"
                    
                    nested_links.append({
                        'url': full_url,
                        'text': link_text,
                        'filename': filename,
                        'parent_item_id': item_id,
                        'parent_meeting_id': meeting_id,
                        'file_extension': file_ext
                    })
            
            if nested_links:
                print(f"✅ Found {len(nested_links)} nested document link(s)")
            else:
                print(f"⚠️ No nested document links found")
                
            return nested_links
                
        except Exception as e:
            print(f"❌ Error extracting nested links from {coversheet_url}: {e}")
            return []

    def download_document(self, url: str, filepath: Path) -> bool:
        """Download a document file from URL."""
        try:
            print(f"📥 Downloading: {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Check if response is actually a document
            content_type = response.headers.get('content-type', '')
            if not any(doc_type in content_type.lower() for doc_type in ['pdf', 'word', 'excel', 'octet-stream']):
                print(f"⚠️ Warning: Response may not be a document (content-type: {content_type})")
                # Still try to save it, as some servers don't set content-type correctly
            
            # Save the document
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            print(f"✅ Saved: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error downloading {url}: {e}")
            return False

    def save_coversheet_html(self, coversheet_url: str, filepath: Path) -> bool:
        """Save the HTML content of a CoverSheet page."""
        try:
            print(f"📄 Saving HTML content from: {coversheet_url}")
            response = self.session.get(coversheet_url, timeout=30)
            response.raise_for_status()
            
            # Save the HTML content
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            print(f"✅ Saved HTML: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving HTML from {coversheet_url}: {e}")
            return False
    
    def is_meeting_downloaded(self, meeting_url: str) -> bool:
        """Check if a meeting has already been downloaded."""
        try:
            # Extract meeting ID from URL
            meeting_id = meeting_url.split('MeetingID=')[1].split('&')[0] if 'MeetingID=' in meeting_url else None
            if not meeting_id:
                return False
            
            # Look for existing meeting folders
            for folder in self.output_dir.iterdir():
                if folder.is_dir() and f"meeting_metadata.json" in [f.name for f in folder.iterdir()]:
                    # Check if this folder contains metadata for the same meeting
                    metadata_file = folder / "meeting_metadata.json"
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                            if metadata.get('meeting_url') == meeting_url:
                                print(f"⏭️ Meeting already downloaded: {folder.name}")
                                return True
                    except:
                        continue
            
            return False
            
        except Exception as e:
            print(f"❌ Error checking if meeting is downloaded: {e}")
            return False
    
    def get_meeting_urls(self, base_url: str = "https://davie.novusagenda.com/agendapublic/", start_date: str = "2025-01-01") -> List[str]:
        """Get a list of meeting URLs to process from start_date onwards."""
        meeting_urls = []
        
        print(f"🔍 Discovering meetings from {start_date} onwards...")
        
        # Search in a wider range to find all meeting types
        # Based on existing downloads, meetings are in the 570-620 range
        # But we also need to check 630-631 for the July 23, 2025 workshop meetings
        max_search_id = 635  # Search up to ID 635 to catch all meeting types including 630-631
        min_search_id = 570  # Search down to ID 570 to catch older meetings
        
        # Add a counter to prevent infinite loops
        consecutive_unknown_dates = 0
        max_unknown_dates = 20  # Stop after 20 consecutive unknown dates
        
        for meeting_id in range(max_search_id, min_search_id, -1):
            meeting_url = f"{base_url}MeetingView.aspx?MeetingID={meeting_id}"
            
            try:
                print(f"🔍 Checking meeting ID {meeting_id}...")
                response = self.session.get(meeting_url, timeout=10)
                
                if response.status_code == 200:
                    # Parse the meeting to get its date
                    soup = BeautifulSoup(response.content, 'html.parser')
                    meeting_info = self.extract_meeting_info(soup)
                    
                    if meeting_info and meeting_info.get('meeting_date') and meeting_info.get('meeting_date') != 'Unknown':
                        try:
                            # Parse the meeting date - handle malformed dates
                            meeting_date = meeting_info['meeting_date']
                            
                            # Clean up common malformed date patterns
                            # Remove prefixes like "FIRST", "SECOND", "HEARING", "MEETING" etc.
                            import re
                            cleaned_date = re.sub(r'^(FIRST|SECOND|HEARING|MEETING)', '', meeting_date).strip()
                            
                            # Try to parse the cleaned date
                            parsed_date = datetime.strptime(cleaned_date, '%B %d, %Y')
                            meeting_date_str = parsed_date.strftime('%Y-%m-%d')
                            
                            # Check if meeting is on or after start_date
                            if meeting_date_str >= start_date:
                                print(f"✅ Found meeting: {meeting_info['meeting_type']} - {meeting_date_str}")
                                meeting_urls.append(meeting_url)
                                consecutive_unknown_dates = 0  # Reset counter
                            else:
                                print(f"⏭️ Skipping meeting {meeting_id} - date {meeting_date_str} is before {start_date}")
                                # If we find a meeting before our start date, we can stop searching
                                # But only if we've found some meetings first
                                if len(meeting_urls) > 0:
                                    print(f"✅ Found {len(meeting_urls)} meetings, stopping search")
                                    break
                                
                        except ValueError as e:
                            print(f"⚠️ Could not parse date for meeting {meeting_id}: {meeting_info.get('meeting_date')}")
                            consecutive_unknown_dates += 1
                            
                            # If we have too many consecutive unknown dates, stop searching
                            if consecutive_unknown_dates >= max_unknown_dates:
                                print(f"🛑 Stopping search after {max_unknown_dates} consecutive unknown dates")
                                break
                            
                            # Add it anyway to be safe
                            meeting_urls.append(meeting_url)
                    else:
                        print(f"⚠️ No meeting info found for ID {meeting_id}")
                        consecutive_unknown_dates += 1
                        
                        # If we have too many consecutive unknown dates, stop searching
                        if consecutive_unknown_dates >= max_unknown_dates:
                            print(f"🛑 Stopping search after {max_unknown_dates} consecutive unknown dates")
                            break
                        
                else:
                    print(f"❌ Meeting ID {meeting_id} not found (status: {response.status_code})")
                    # If we get too many 404s in a row, we might have reached the end
                    if response.status_code == 404:
                        # Check if we've found any meetings recently
                        if len(meeting_urls) == 0:
                            print(f"⚠️ No meetings found yet, continuing search...")
                        else:
                            print(f"✅ Found {len(meeting_urls)} meetings, stopping search")
                            break
                            
            except Exception as e:
                print(f"❌ Error checking meeting ID {meeting_id}: {e}")
                continue
            
            # Small delay to be respectful
            time.sleep(1)
        
        print(f"🎯 Found {len(meeting_urls)} meetings from {start_date} onwards")
        return meeting_urls

    def scrape_meeting(self, meeting_url: str) -> Dict:
        """Scrape a single meeting agenda and download all PDFs."""
        print(f"🔍 Scraping meeting: {meeting_url}")
        
        try:
            # Fetch the agenda page
            response = self.session.get(meeting_url, timeout=30)
            response.raise_for_status()
            
            # Parse the HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract meeting information
            meeting_info = self.extract_meeting_info(soup)
            print(f"📋 Meeting: {meeting_info['meeting_type']} - {meeting_info['meeting_date']}")
            
            # Extract PDF links
            pdf_links = self.extract_pdf_links(soup, meeting_url)
            print(f"📄 Found {len(pdf_links)} PDF links")
            
            # Create meeting folder
            meeting_date = meeting_info['meeting_date']
            meeting_type = meeting_info['meeting_type'].replace(' ', '_').lower()
            
            # Parse date for folder naming
            try:
                parsed_date = datetime.strptime(meeting_date, '%B %d, %Y')
                folder_name = f"{parsed_date.strftime('%Y-%m-%d')}_{meeting_type}"
            except:
                folder_name = f"unknown_date_{meeting_type}"
            
            meeting_folder = self.output_dir / folder_name
            meeting_folder.mkdir(exist_ok=True)
            
            # Download all documents and nested links
            successful_downloads = 0
            nested_downloads = 0
            all_nested_links = []
            
            for i, pdf_link in enumerate(pdf_links, 1):
                print(f"\n[{i}/{len(pdf_links)}] Processing: {pdf_link['text'][:50]}...")
                
                # First, extract all nested links for this item
                nested_links = self.extract_nested_links(pdf_link['url'], pdf_link['item_id'], pdf_link['meeting_id'])
                all_nested_links.extend(nested_links)
                
                # If we found nested documents, save HTML content (no main document)
                if nested_links:
                    print(f"📄 Found {len(nested_links)} nested documents, saving HTML content...")
                    html_filename = f"{pdf_link['filename'].replace('.pdf', '')}.html"
                    html_filepath = meeting_folder / html_filename
                    
                    if self.save_coversheet_html(pdf_link['url'], html_filepath):
                        successful_downloads += 1
                        print(f"✅ Saved HTML content: {html_filename}")
                else:
                    # No nested documents, check for main document
                    actual_doc_url = self.extract_document_download_link(pdf_link['url'])
                    
                    if actual_doc_url:
                        # Determine file extension from URL
                        if '.docx' in actual_doc_url.lower():
                            file_ext = '.docx'
                        elif '.xlsx' in actual_doc_url.lower():
                            file_ext = '.xlsx'
                        elif '.pptx' in actual_doc_url.lower():
                            file_ext = '.pptx'
                        elif '.ppt' in actual_doc_url.lower():
                            file_ext = '.ppt'
                        elif '.doc' in actual_doc_url.lower():
                            file_ext = '.doc'
                        elif '.xls' in actual_doc_url.lower():
                            file_ext = '.xls'
                        else:
                            file_ext = '.pdf'  # Default to PDF
                        
                        # Update filename with correct extension
                        base_filename = pdf_link['filename'].replace('.pdf', '')
                        doc_filename = f"{base_filename}{file_ext}"
                        doc_filepath = meeting_folder / doc_filename
                        
                        if self.download_document(actual_doc_url, doc_filepath):
                            successful_downloads += 1
                    else:
                        print(f"⚠️ No documents found for: {pdf_link['text'][:50]}...")
                        # Save the HTML content instead
                        html_filename = f"{pdf_link['filename'].replace('.pdf', '')}.html"
                        html_filepath = meeting_folder / html_filename
                        
                        if self.save_coversheet_html(pdf_link['url'], html_filepath):
                            successful_downloads += 1
                            print(f"✅ Saved HTML content: {html_filename}")
                
                # Small delay between downloads
                time.sleep(2)
            
            # Download all nested links
            print(f"\n📥 Downloading {len(all_nested_links)} nested documents...")
            for i, nested_link in enumerate(all_nested_links, 1):
                print(f"\n[{i}/{len(all_nested_links)}] Downloading nested: {nested_link['text'][:50]}...")
                
                nested_filepath = meeting_folder / nested_link['filename']
                
                if self.download_document(nested_link['url'], nested_filepath):
                    nested_downloads += 1
                
                # Small delay between nested downloads
                time.sleep(1)
            
            # Save meeting metadata (after processing all links)
            metadata = {
                'meeting_info': meeting_info,
                'meeting_url': meeting_url,
                'pdf_links': pdf_links,
                'nested_links': all_nested_links,
                'scraped_at': datetime.now().isoformat()
            }
            
            metadata_file = meeting_folder / "meeting_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Metadata saved: {metadata_file}")
            
            print(f"\n🎉 Successfully downloaded {successful_downloads}/{len(pdf_links)} main documents")
            print(f"📥 Successfully downloaded {nested_downloads}/{len(all_nested_links)} nested documents")
            print(f"📁 Meeting folder: {meeting_folder}")
            
            return {
                'meeting_info': meeting_info,
                'total_main_documents': len(pdf_links),
                'successful_main_downloads': successful_downloads,
                'total_nested_documents': len(all_nested_links),
                'successful_nested_downloads': nested_downloads,
                'meeting_folder': str(meeting_folder)
            }
            
        except Exception as e:
            print(f"❌ Error scraping meeting: {e}")
            return {}

def main():
    parser = argparse.ArgumentParser(description="Scrape Town of Davie agenda pages and download PDFs.")
    parser.add_argument('--meeting-url', 
                       help='URL of the meeting agenda to scrape (if not provided, will process multiple meetings)')
    parser.add_argument('--output-dir', 
                       default='../downloads/agendas',
                       help='Output directory for downloaded PDFs')
    parser.add_argument('--start-date', 
                       default='2025-01-01',
                       help='Start date for meeting discovery (YYYY-MM-DD format)')
    args = parser.parse_args()
    
    scraper = AgendaScraper(output_dir=args.output_dir)
    
    if args.meeting_url:
        # Process single meeting
        result = scraper.scrape_meeting(args.meeting_url)
        if result:
            print(f"\n✅ Scraping completed successfully!")
            print(f"📊 Summary: {result['successful_main_downloads']}/{result['total_main_documents']} main documents downloaded")
            print(f"📊 Summary: {result['successful_nested_downloads']}/{result['total_nested_documents']} nested documents downloaded")
        else:
            print(f"\n❌ Scraping failed!")
    else:
        # Process multiple meetings
        meeting_urls = scraper.get_meeting_urls(start_date=args.start_date)
        print(f"🚀 Processing {len(meeting_urls)} meetings from {args.start_date} onwards...")
        
        successful_meetings = 0
        skipped_meetings = 0
        
        for i, meeting_url in enumerate(meeting_urls, 1):
            print(f"\n{'='*60}")
            print(f"📅 Processing meeting {i}/{len(meeting_urls)}: {meeting_url}")
            print(f"{'='*60}")
            
            # Check if already downloaded
            if scraper.is_meeting_downloaded(meeting_url):
                print(f"⏭️ Skipping - already downloaded")
                skipped_meetings += 1
                continue
            
            # Process the meeting
            result = scraper.scrape_meeting(meeting_url)
            if result:
                successful_meetings += 1
                print(f"✅ Meeting {i} completed successfully!")
            else:
                print(f"❌ Meeting {i} failed!")
            
            # Small delay between meetings
            time.sleep(3)
        
        print(f"\n🎉 Batch processing completed!")
        print(f"✅ Successful meetings: {successful_meetings}")
        print(f"⏭️ Skipped meetings: {skipped_meetings}")
        print(f"📁 Total processed: {successful_meetings + skipped_meetings}")

if __name__ == "__main__":
    main() 