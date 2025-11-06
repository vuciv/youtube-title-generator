# YouTube Title Generation Model

A pipeline for training a fine-tuned model that generates high-CTR YouTube titles based on video transcripts. The model learns from trending Science & Technology videos to capture curiosity-driven title patterns.

## Overview

This project fine-tunes an OpenAI model to generate compelling YouTube titles by learning from successful trending videos. The pipeline filters a large Kaggle dataset, uses Gemini AI for quality filtering, fetches transcripts, and trains a specialized title generation model.

**Important**: Datasets are everything. The quality and relevance of your training data directly determines the quality of your fine-tuned model. Choose your category carefully based on your niche to ensure the model learns from the right patterns.

## Workflow

### 1. **Start with Kaggle Dataset**
   - Begin with the `US_youtube_trending_data.csv` from Kaggle
   - This dataset contains trending video data including titles, channels, views, and metadata

### 2. **Filter by Category**
   - Run `scripts/filter_category_28.py` to extract videos from category 28 (Science & Technology)
   - **Choose your category based on your niche**: You can modify the script to filter by any category ID that matches your content type
   - Check `data/categories.json` for all available YouTube categories and their IDs
   - Common categories: 20 (Gaming), 24 (Entertainment), 27 (Education), 28 (Science & Technology), etc.
   - Outputs filtered CSV with unique videos matching the target category

### 3. **Gemini Quality Filtering**
   - Run `scripts/filter_videos_with_gemini.py` to further refine the dataset
   - Uses Gemini AI to filter out contamination (corporate events, livestreams, deal content)
   - Keeps only videos with curiosity-driven elements: conflict, controversy, absurdity, or "big questions"
   - Outputs `category_28_videos_filtered.csv` with high-quality training examples

### 4. **Final Manual Pass** (Recommended)
   - Review the filtered dataset and manually remove:
     - **Low-quality titles**: Titles that don't actually demonstrate good title writing (even if they passed Gemini filtering)
     - **Biggest channels**: Videos from channels with millions of subscribers that get views regardless of title quality (e.g., Apple, Samsung, major news outlets)
     - These channels don't teach good title patterns because their views come from brand recognition, not title skill
   - This manual curation step is crucial for dataset quality - remember: **datasets are everything**

### 5. **Fetch Transcripts**
   - Run `scripts/fetch_transcripts.py` to download video transcripts
   - Processes videos concurrently for efficiency
   - Requires a list of YouTube URLs in `data/urls.txt`
   - Outputs `data/training_data.json` with transcripts and titles

### 6. **Fine-tune Model**
   - Run `scripts/train_title_model.py` to prepare training data and start fine-tuning
   - Prepares JSONL format for OpenAI fine-tuning
   - Uploads training file and creates fine-tuning job
   - Model learns to generate titles that capture curiosity gaps and compelling hooks

## Setup

### Prerequisites
- Python 3.8+
- OpenAI API key
- Google Gemini API key (for filtering)
- Proxy credentials (for transcript fetching, optional)

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
# Create a .env file with:
# OPENAI_API_KEY=your_key_here
# GEMINI_API_KEY=your_key_here
# PROXY_USERNAME=your_proxy_user (optional)
# PROXY_PASSWORD=your_proxy_pass (optional)
```

## Usage

### Training Pipeline

```bash
# 1. Filter by category (assumes data/kaggle/extracted/US_youtube_trending_data.csv exists)
python scripts/filter_category_28.py > category_28_videos.csv

# 2. Filter with Gemini
python scripts/filter_videos_with_gemini.py

# 3. (Optional but recommended) Do a final manual pass to remove:
#    - Low-quality titles that don't demonstrate good title writing
#    - Videos from the biggest channels (they get views from brand recognition, not title skill)

# 4. Prepare URLs file (create data/urls.txt with YouTube URLs, one per line)
# Then fetch transcripts
python scripts/fetch_transcripts.py

# 5. Train the model
python scripts/train_title_model.py
```

### Generating Titles

```bash
# Generate titles from a transcript
python scripts/generate_titles.py "your transcript text here"

# Generate title recommendations for an entire channel
python scripts/generate_channel_titles.py
```

## Project Structure

```
yt-data/
├── data/
│   ├── US_youtube_trending_data.csv    # Kaggle dataset
│   ├── categories.json                 # YouTube category IDs and names
│   ├── training_data.json              # Transcripts and titles
│   └── urls.txt                        # YouTube URLs for transcript fetching
├── scripts/
│   ├── filter_category_28.py           # Category filtering
│   ├── filter_videos_with_gemini.py    # Quality filtering with Gemini
│   ├── fetch_transcripts.py            # Transcript fetching
│   ├── train_title_model.py            # Model training
│   ├── generate_titles.py              # Title generation
│   ├── generate_channel_titles.py      # Batch title generation
│   └── utils/
│       └── gemini_client.py            # Gemini API client
├── category_28_videos_filtered.csv     # Filtered dataset
└── requirements.txt                    # Python dependencies
```

## Key Features

- **Quality Filtering**: Uses Gemini AI to identify videos with curiosity-driven titles
- **Concurrent Processing**: Efficient transcript fetching with thread pooling
- **Fine-tuning**: Custom OpenAI model trained on high-CTR title patterns
- **Batch Processing**: Generate title recommendations for entire channels

## Notes

- **Datasets are everything**: The quality of your training data directly impacts model performance. Spend time curating the right category and filtering criteria for your niche
- **Category Selection**: Choose a category that matches your content niche. Check `data/categories.json` to see all available YouTube categories (e.g., Gaming=20, Entertainment=24, Education=27, Science & Technology=28)
- **Final Manual Pass**: After Gemini filtering, do a manual review to remove low-quality titles and videos from the biggest channels. These channels (e.g., Apple, Samsung, major news outlets) get views from brand recognition, not title skill, so they don't teach good title patterns
- The filtering criteria focuses on keeping videos with conflict, controversy, absurdity, or "big questions"
- Removes corporate events, livestreams, and deal content that don't teach good title patterns
- Model is fine-tuned on `gpt-4.1-mini-2025-04-14` base model
- Training data is sampled (default: 800 examples) for cost efficiency

## Results

The fine-tuned model has shown significant improvements in click-through rates, with some videos achieving 15%+ CTR from subscribers by learning patterns from successful trending titles.

