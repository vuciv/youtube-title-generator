#!/usr/bin/env python3
"""
Filter videos from category_28_videos.csv using Gemini LLM to determine
which videos should be kept based on quality criteria.
"""

import pandas as pd
import sys
import os
from pathlib import Path
from tqdm import tqdm
import time
from typing import Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.utils.gemini_client import get_text_response

# Configuration
CSV_FILE = Path(__file__).parent.parent / 'category_28_videos.csv'
OUTPUT_FILE = Path(__file__).parent.parent / 'category_28_videos_filtered.csv'

FILTERING_CRITERIA = """
You are helping filter a YouTube dataset for training a high-CTR title generation model. 
Your task is to determine if a video should be KEPT or REMOVED based on the following criteria:

REMOVE these types of videos (contamination):
1. **Corporate Event/Ads**: Apple Events, Samsung Galaxy Unpacked, Microsoft Surface launches, etc.
   - These got high views because of Brand Power/News, not pure title skill
   - Examples: "Apple Event — October 13, Introducing iPad Air", "Galaxy Unpacked", "Introducing Windows 11"

2. **Pure Livestreams/Broadcasts**: SpaceX Starlink missions, NASA broadcasts, mission replays
   - These are highly searchable news bulletins that teach generic functional titles
   - Examples: "Starlink Mission", "DART Impact", "Replay - New Shepard Mission NS-13 Webcast"

3. **"Deal Guy" Content**: Amazon Prime Day deals, Black Friday deals, top X deals lists
   - This is pure listicle commerce, not educational content
   - Examples: "Top 50 Amazon Prime Day Deals 2020 🤑", "Top 10 Target Black Friday Deals 2021"

4. **Simple "Official" Videos**: Basic product announcements without curiosity/conflict
   - Examples: "The new MacBook Pro", "Introducing Apple Vision Pro"

 KEEP these types of videos (gold standard):
1. **Conflict/Controversy**: Titles that generate curiosity through debate or controversy
   - Examples: "NVIDIA just made EVERYTHING ELSE obsolete", "iPhone vs Android - Which can survive a CAR? 🚗"

2. **Absurdity/Engineering Feat**: Titles that highlight unusual or impressive engineering
   - Examples: "I Invented Three New Incredible Ways to Die", "4000° PLASMA LIGHTSABER BUILD"

3. **The Big Question/Hidden Truth**: Titles that pose interesting questions or reveal hidden insights
   - Examples: "Is The Metric System Actually Better?", "How Humans Lost Their Fur"

Respond with ONLY: "KEEP" or "REMOVE" followed by a brief reason (one sentence).
"""


def should_keep_video(title: str, channel_title: str, tags: str) -> Tuple[bool, str]:
    """
    Uses Gemini to determine if a video should be kept.
    Returns (should_keep: bool, reason: str)
    """
    tags_str = tags if tags and tags != "[None]" else "No tags"
    
    prompt = f"""
    {FILTERING_CRITERIA}

    Video Title: "{title}"
    Channel: {channel_title}
    Tags: {tags_str}

    Should this video be KEPT or REMOVED?
    """
    
    try:
        response = get_text_response(prompt)
        response_upper = response.upper()
        if "REMOVE" in response_upper or response_upper.startswith("REMOVE"):
            return False, response
        elif "KEEP" in response_upper or response_upper.startswith("KEEP"):
            return True, response
        else:
            return True, f"UNCLEAR: {response}"
    except Exception as e:
        print(f"ERROR processing video '{title}': {e}", file=sys.stderr)
        return True, f"ERROR: {e}"


def main():
    # Check if CSV file exists
    if not CSV_FILE.exists():
        print(f"Error: CSV file not found at {CSV_FILE}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Reading CSV file: {CSV_FILE}", file=sys.stderr)
    
    try:
        # Read the CSV
        df = pd.read_csv(CSV_FILE, low_memory=False)
        
        print(f"Found {len(df)} videos to filter", file=sys.stderr)
        print("Starting Gemini filtering (this may take a while)...", file=sys.stderr)
        
        # Track decisions
        keep_decisions = []
        reasons = []
        
        # Process each video with progress bar
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Filtering videos"):
            title = row['title']
            channel_title = row['channelTitle']
            tags = row['tags']
            
            should_keep, reason = should_keep_video(title, channel_title, tags)
            keep_decisions.append(should_keep)
            reasons.append(reason)
            print(f"Video '{title}' should be {should_keep} because {reason}", file=sys.stderr)
            
            # Small delay to avoid rate limiting
            time.sleep(0.1)
        
        # Filter the dataframe
        df['keep'] = keep_decisions
        df['reason'] = reasons
        
        kept_df = df[df['keep'] == True].copy()
        removed_df = df[df['keep'] == False].copy()
        
        # Remove the helper columns before saving
        kept_df = kept_df.drop(columns=['keep', 'reason'])
        
        print(f"\nFiltering complete!", file=sys.stderr)
        print(f"  - Kept: {len(kept_df)} videos", file=sys.stderr)
        print(f"  - Removed: {len(removed_df)} videos", file=sys.stderr)
        
        # Save filtered results
        kept_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nSaved filtered videos to: {OUTPUT_FILE}", file=sys.stderr)
        
        # Optionally save removed videos with reasons for review
        removed_file = Path(__file__).parent.parent / 'category_28_videos_removed.csv'
        removed_df.to_csv(removed_file, index=False)
        print(f"Saved removed videos (with reasons) to: {removed_file}", file=sys.stderr)
        
    except Exception as e:
        print(f"Error processing CSV: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

