import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from pytubefix import YouTube
from tqdm import tqdm
from youtube_transcript_api import (NoTranscriptFound, TranscriptsDisabled,
                                    YouTubeTranscriptApi)
from youtube_transcript_api.proxies import WebshareProxyConfig

# --- Configuration ---
load_dotenv()
DATA_DIR = './data'
MAX_WORKERS = 10  # Scalability knob. Tune based on your network/CPU. 10 is a safe start.

# --- Global, thread-safe clients ---
PROXY_CONFIG = WebshareProxyConfig(
    proxy_username=os.getenv("PROXY_USERNAME"),
    proxy_password=os.getenv("PROXY_PASSWORD"),
)
TRANSCRIPT_API = YouTubeTranscriptApi(proxy_config=PROXY_CONFIG)


def process_single_url(url: str) -> dict | None:
    """
    Worker function to process a single YouTube URL.
    Returns simplified data structure with url, video_id, title, and full_transcript.
    """
    try:
        # Extract video_id first
        video_id = YouTube(url).video_id
        
        # Fetch the transcript
        transcript_list = TRANSCRIPT_API.fetch(video_id)
        
        # Get the title
        video_title = YouTube(url).title

        # Combine transcript snippets into full text
        full_transcript = " ".join(snippet.text for snippet in transcript_list)

        return {
            "url": url,
            "video_id": video_id,
            "title": video_title,
            "full_transcript": full_transcript.strip()
        }
    except (NoTranscriptFound, TranscriptsDisabled):
        return None
    except Exception as e:
        print(f"  - ERROR: Failed to process URL {url}: {e}")
        return None


def fetch_transcripts(youtube_urls: list[str]) -> list[dict]:
    """
    Fetches video data concurrently using a thread pool for maximum speed.
    Returns a list of video data dictionaries.
    """
    all_videos_data = []
    
    # Use ThreadPoolExecutor to run process_single_url concurrently
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create a future for each URL
        future_to_url = {executor.submit(process_single_url, url): url for url in youtube_urls}
        
        # Use tqdm to show a progress bar as futures complete
        progress_bar = tqdm(as_completed(future_to_url), total=len(youtube_urls), desc="Processing Videos")
        
        for future in progress_bar:
            result = future.result()
            if result:  # Only append successful results
                all_videos_data.append(result)
                
    return all_videos_data


# --- Main Entry Point ---
if __name__ == "__main__":
    try:
        with open(f'{DATA_DIR}/urls.txt', 'r') as f:
            video_links = [line.strip() for line in f if line.strip()]
            video_links = list(set(video_links))  # Remove duplicates
    except FileNotFoundError:
        print(f"Error: The file {DATA_DIR}/urls.txt was not found.")
        exit()

    print(f"Found {len(video_links)} URLs. Starting concurrent processing with {MAX_WORKERS} workers...")
    
    videos_data = fetch_transcripts(video_links)

    output_filename = f'{DATA_DIR}/training_data.json'
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(videos_data, f, indent=4)
        print(f"\nSuccessfully saved {len(videos_data)} videos to {output_filename}")
    except Exception as e:
        print(f"\nError saving to file: {e}")





