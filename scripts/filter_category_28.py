#!/usr/bin/env python3
"""
Filter videos from US_youtube_trending_data.csv where categoryId == 28
and output specified columns to stdout as CSV.
"""

import pandas as pd
import sys
from pathlib import Path

# Configuration
CSV_FILE = Path(__file__).parent.parent / 'data' / 'kaggle' / 'extracted' / 'US_youtube_trending_data.csv'
OUTPUT_COLUMNS = [
    'video_id',
    'title',
    'publishedAt',
    'channelId',
    'channelTitle',
    'categoryId',
    'trending_date',
    'tags',
    'view_count',
    'likes',
    'dislikes',
    'comment_count'
]

def main():
    # Check if CSV file exists
    if not CSV_FILE.exists():
        print(f"Error: CSV file not found at {CSV_FILE}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Reading CSV file: {CSV_FILE}", file=sys.stderr)
    
    try:
        # Read CSV in chunks for memory efficiency
        chunk_size = 100000
        filtered_chunks = []
        
        for chunk in pd.read_csv(CSV_FILE, chunksize=chunk_size, low_memory=False):
            # Filter for categoryId == 28
            filtered_chunk = chunk[chunk['categoryId'] == 28]
            
            if not filtered_chunk.empty:
                # Select only the specified columns
                filtered_chunk = filtered_chunk[OUTPUT_COLUMNS]
                filtered_chunks.append(filtered_chunk)
        
        if not filtered_chunks:
            print("No videos found with categoryId == 28", file=sys.stderr)
            sys.exit(0)
        
        # Combine all chunks
        result_df = pd.concat(filtered_chunks, ignore_index=True)
        
        # Remove duplicates based on video_id, keeping the first occurrence
        result_df = result_df.drop_duplicates(subset='video_id', keep='first')
        
        print(f"Found {len(result_df)} unique videos with categoryId == 28", file=sys.stderr)
        
        # Output to stdout as CSV
        result_df.to_csv(sys.stdout, index=False)
        
    except Exception as e:
        print(f"Error processing CSV: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

