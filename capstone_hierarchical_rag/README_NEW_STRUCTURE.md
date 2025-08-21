# Capstone Project - New Folder Structure

## 📁 Organization

```
capstone/
├── youtube_processes/              # YouTube-related scripts and configs
│   ├── youtube_hybrid_downloader.py
│   ├── .env                       # YouTube API credentials
│   └── CREDENTIALS_SETUP.md      # Setup instructions
├── downloads/                     # All downloaded data
│   └── town_meetings_youtube/    # Town meeting transcripts and metadata
├── .gitignore                    # Security (excludes downloads and credentials)
├── README.md                     # Main project documentation
├── .cursorrules                  # IDE rules
└── CLEANUP_SUMMARY.md           # Cleanup documentation
```

## 🎯 Purpose

### `youtube_processes/`
- Contains all YouTube-related scripts and configuration
- Isolated from main project for better organization
- Includes API credentials and setup instructions

### `downloads/`
- Central location for all downloaded data
- `town_meetings_youtube/` contains:
  - 52 transcript files (meeting content)
  - 56 metadata files (titles, dates, descriptions)
  - Complete dataset ready for RAG system

## 🚀 Usage

To run the YouTube downloader:
```bash
cd youtube_processes
python youtube_hybrid_downloader.py
```

The script will automatically save data to `../downloads/town_meetings_youtube/`

## 🔒 Security

- `.env` file is in `youtube_processes/` and excluded from git
- `downloads/` folder is excluded from git to avoid committing large data files
- Credentials and sensitive data are properly isolated

## 📊 Data Summary

- **52 transcript files** available
- **56 metadata files** available  
- **Complete coverage** (Jan 2024 - Jul 2025)
- **All meeting types** (Council, CRA, Budget, Special Assessment)

Ready for RAG system development! 🚀 