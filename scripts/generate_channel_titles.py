"""
Fetch all videos from joshycodes YouTube channel, get transcripts, and generate recommended titles.
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from pytubefix import Channel, YouTube
from tqdm import tqdm
from youtube_transcript_api import (NoTranscriptFound, TranscriptsDisabled,
                                    YouTubeTranscriptApi)
from youtube_transcript_api.proxies import WebshareProxyConfig

# Import title generation function
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_titles import generate_title

load_dotenv()

# Configuration
CHANNEL_NAME = "joshycodes"
CHANNEL_URL = f"https://www.youtube.com/@{CHANNEL_NAME}"
DATA_DIR = './data'
OUTPUT_FILE = f'{DATA_DIR}/joshycodes_titles.json'
MAX_WORKERS = 5  # Lower for API rate limits

# Global, thread-safe clients
PROXY_CONFIG = WebshareProxyConfig(
    proxy_username=os.getenv("PROXY_USERNAME"),
    proxy_password=os.getenv("PROXY_PASSWORD"),
)
TRANSCRIPT_API = YouTubeTranscriptApi(proxy_config=PROXY_CONFIG)


def get_channel_videos(channel_url: str) -> list[dict]:
    """
    Get all video URLs from a YouTube channel.
    
    Args:
        channel_url: YouTube channel URL (e.g., https://www.youtube.com/@joshycodes)
        
    Returns:
        List of video info dictionaries with url, video_id, title
    """
    print(f"\nFetching videos from channel: {channel_url}")
    
    try:
        channel = Channel(channel_url)
        videos = []
        
        print(f"Found channel: {channel.channel_name}")
        print(f"Getting video list...")
        
        # Get all videos from the channel
        for video in channel.videos:
            try:
                videos.append({
                    "url": video.watch_url,
                    "video_id": video.video_id,
                    "title": video.title
                })
            except Exception as e:
                print(f"  - ERROR getting video info: {e}")
                continue
        
        print(f"Found {len(videos)} videos")
        return videos
        
    except Exception as e:
        print(f"ERROR fetching channel videos: {e}")
        return []


def process_single_video(video_info: dict) -> dict | None:
    """
    Process a single video: fetch transcript and generate title.
    
    Args:
        video_info: Dictionary with url, video_id, title
        
    Returns:
        Video data with transcript and generated title, or None if failed
    """
    try:
        video_id = video_info['video_id']
        
        # Fetch the transcript
        try:
            transcript_list = TRANSCRIPT_API.fetch(video_id)
            full_transcript = " ".join(snippet.text for snippet in transcript_list).strip()
        except (NoTranscriptFound, TranscriptsDisabled):
            print(f"  - No transcript available for: {video_info['title']}")
            return None
        
        if not full_transcript or len(full_transcript) < 200:
            print(f"  - Transcript too short for: {video_info['title']}")
            return None
        
        # Generate recommended title using fine-tuned model
        print(f"  - Generating title for: {video_info['title']}")
        recommended_title = generate_title(full_transcript, temperature=0.3)
        
        if not recommended_title:
            print(f"  - Failed to generate title for: {video_info['title']}")
            return None
        
        return {
            "url": video_info['url'],
            "video_id": video_id,
            "original_title": video_info['title'],
            "recommended_title": recommended_title,
            "transcript_length": len(full_transcript),
            "transcript_preview": full_transcript[:200] + "..." if len(full_transcript) > 200 else full_transcript
        }
        
    except Exception as e:
        print(f"  - ERROR processing {video_info.get('title', 'unknown')}: {e}")
        return None


def process_channel_videos(channel_url: str) -> list[dict]:
    """
    Get all videos from channel, fetch transcripts, and generate titles.
    
    Args:
        channel_url: YouTube channel URL
        
    Returns:
        List of processed video data
    """
    # Get all videos from channel
    videos = get_channel_videos(channel_url)
    
    if not videos:
        print("No videos found!")
        return []
    
    print(f"\nProcessing {len(videos)} videos...")
    print(f"Using {MAX_WORKERS} workers\n")
    
    all_results = []
    
    # Process videos concurrently
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_video = {executor.submit(process_single_video, video): video for video in videos}
        
        progress_bar = tqdm(as_completed(future_to_video), total=len(videos), desc="Processing Videos")
        
        for future in progress_bar:
            result = future.result()
            if result:
                all_results.append(result)
    
    return all_results


def save_results(results: list[dict], output_file: str):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"Total videos processed: {len(results)}")
    print(f"{'='*60}\n")
    
    # Print summary
    print("\nTitle Recommendations:")
    print(f"{'='*60}")
    for i, video in enumerate(results, 1):
        print(f"\n{i}. {video['original_title']}")
        print(f"   → Recommended: {video['recommended_title']}")
        print(f"   Video: {video['url']}")
    print(f"{'='*60}\n")


def main():
    """Main function."""
    print(f"\n{'='*60}")
    print("joshycodes Title Generator")
    print(f"{'='*60}")
    
    # Process all videos
    results = process_channel_videos(CHANNEL_URL)
    
    if results:
        save_results(results, OUTPUT_FILE)
    else:
        print("No videos processed successfully.")


if __name__ == "__main__":
    main()




