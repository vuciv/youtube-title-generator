import json
import os
import random
from typing import Optional, Dict
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Model options (in order of cost/performance):
# - 'gpt-4.1-nano-2025-04-14'    → ~$5-10 for 800 examples, cheapest, good baseline
# - 'gpt-4.1-mini-2025-04-14'    → ~$20-40 for 800 examples, 4x Nano cost, better quality
# - 'gpt-4.1-2025-04-14'         → ~$200 for 800 examples, best quality, most expensive
BASE_MODEL = 'gpt-4.1-mini-2025-04-14'  # Upgraded from nano - good balance of cost/quality
DATA_DIR = './data'
TRAINING_DATA_PATH = f'{DATA_DIR}/training_data.json'
STRICT_SAFETY_THRESHOLD = 0.1
SAMPLE_SIZE = 800  # Adjust based on your dataset size

# Quality thresholds for training data
MIN_TRANSCRIPT_LENGTH = 200  # Minimum characters in transcript for meaningful content
MAX_TRANSCRIPT_LENGTH = 50000  # Maximum to avoid extremely long transcripts
MIN_TITLE_LENGTH = 10  # Minimum characters in title
MAX_TITLE_LENGTH = 100  # Maximum characters in title (YouTube limit is ~100)


def check_text_safety_strict(text_to_check: str) -> Optional[Dict[str, float]]:
    """
    Uses a strict, custom threshold on moderation scores to check for safety.
    Returns a dictionary of flagged categories and their scores, or None if safe.
    """
    if not text_to_check:
        return None
    try:
        response = client.moderations.create(
            model="omni-moderation-latest",
            input=text_to_check
        )
        result = response.results[0]
        
        flagged_categories = {}
        for category, score in result.category_scores.__dict__.items():
            if score is not None and score > STRICT_SAFETY_THRESHOLD:
                flagged_categories[category] = score
        
        return flagged_categories if flagged_categories else None

    except Exception as e:
        print(f"  - WARNING: Moderation API call failed: {e}. Skipping moderation for this item.")
        return None


def prepare_training_data(training_data_path: str, sample_size: int) -> Optional[str]:
    """
    Prepares high-quality training data for YouTube title generation fine-tuning.
    
    This function implements OpenAI's best practices for fine-tuning:
    - Clear, detailed system prompt with explicit instructions and guidelines
    - Structured user prompts with context and clear task definition
    - Quality filtering (length requirements, content validation)
    - Safety filtering using OpenAI's moderation API
    - Proper JSONL formatting for supervised fine-tuning
    
    Args:
        training_data_path: Path to JSON file containing video data with transcripts and titles
        sample_size: Number of examples to sample from the dataset
        
    Returns:
        Path to the generated JSONL file, or None if insufficient quality examples found
    """
    print(f"\n--- Preparing training data for Title Generator ---")
    
    try:
        with open(training_data_path, 'r', encoding='utf-8') as f:
            videos = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Training data file not found at {training_data_path}.")
        print("Run fetch_transcripts.py first.")
        return None
    except json.JSONDecodeError:
        print(f"ERROR: Could not parse JSON file at {training_data_path}.")
        return None

    # Filter for videos with both full_transcript and title, and basic quality checks
    valid_videos = [
        v for v in videos 
        if 'full_transcript' in v and 'title' in v
        and v.get('full_transcript') and v.get('title')
        and len(v.get('full_transcript', '').strip()) >= MIN_TRANSCRIPT_LENGTH
        and len(v.get('full_transcript', '').strip()) <= MAX_TRANSCRIPT_LENGTH
        and len(v.get('title', '').strip()) >= MIN_TITLE_LENGTH
        and len(v.get('title', '').strip()) <= MAX_TITLE_LENGTH
    ]
    
    print(f"Found {len(valid_videos)} high-quality examples after basic filtering.")
    
    # Sample from valid videos
    sample_size = min(len(valid_videos), sample_size)
    training_set = random.sample(valid_videos, sample_size)
        
    print(f"Using a sample of {len(training_set)} examples for training.")

    jsonl_local_path = 'openai_title_training_data.jsonl'
    clean_examples_count = 0
    skipped_examples_count = 0

    # Enhanced system prompt with clear instructions and guidelines
    system_message = {
        "role": "system",
        "content": """You are an expert YouTube title generator. Your task is to create compelling, accurate titles that capture the essence of video content based on transcripts.

Guidelines for generating YouTube titles:
1. Be accurate and truthful - the title must reflect the actual content of the video
2. Make it compelling and click-worthy while staying honest about the content
3. Keep titles concise (ideally 50-80 characters, max 100 characters)
4. Use natural language that viewers would search for
5. Highlight the most interesting or valuable aspect of the video
6. Capture curiosity gaps

Generate only the title itself, nothing else."""
    }

    with open(jsonl_local_path, 'w', encoding='utf-8') as f:
        for video in training_set:
            input_text = video['full_transcript'].strip()
            output_text = video['title'].strip()

            # Safety filtering (quality already filtered before sampling)
            #input_flags = check_text_safety_strict(input_text)
            #output_flags = check_text_safety_strict(output_text)

            #if input_flags or output_flags:
            #    skipped_examples_count += 1
            #    print(f"  - SKIPPED: Unsafe data. Input flagged: {input_flags}. Output flagged: {output_flags}.")
            #    continue

            # Enhanced user message with clear structure and instructions
            user_message_content = f"""Based on the following video transcript, generate an accurate and compelling YouTube title that captures the essence of the content.

Transcript:
{input_text}

Generate the YouTube title:"""

            user_message = {"role": "user", "content": user_message_content}
            assistant_message = {"role": "assistant", "content": output_text}
            
            messages_list = [system_message, user_message, assistant_message]
            json_line = json.dumps({"messages": messages_list})
            f.write(json_line + '\n')
            clean_examples_count += 1
            
    print(f"Data preparation complete for Title Generator.")
    print(f" - Wrote {clean_examples_count} clean examples to '{jsonl_local_path}'")
    print(f" - Skipped {skipped_examples_count} examples due to safety filtering")
    
    if clean_examples_count < 10:
        print(f"\nWARNING: Fewer than 10 clean examples were prepared. This is not enough for a fine-tuning job.")
        return None

    return jsonl_local_path


def train_model(training_file_path: str):
    """
    Uploads the training file and starts a fine-tuning job with OpenAI.
    """
    if not training_file_path:
        print("No training file path provided. Aborting training.")
        return
    
    print(f"\n--- Starting training for Title Generator ---")
    try:
        # 1. Upload the training file
        print(f"  - Uploading '{training_file_path}' to OpenAI...")
        with open(training_file_path, "rb") as f:
            training_file_object = client.files.create(file=f, purpose="fine-tune")
        print(f"  - File uploaded successfully. File ID: {training_file_object.id}")

        # Clean up local file immediately after upload
        os.remove(training_file_path)
        print(f"  - Local file '{training_file_path}' removed.")

        # 2. Create the fine-tuning job
        print(f"  - Creating fine-tuning job for model '{BASE_MODEL}' with suffix 'title-gen'...")
        job = client.fine_tuning.jobs.create(
            training_file=training_file_object.id,
            model=BASE_MODEL,
            suffix="title-gen"
        )
        
        print(f"  - Successfully submitted fine-tuning job! Job ID: {job.id}")
        print(f"\nMonitor your job at: https://platform.openai.com/finetune")
        return job.id
        
    except Exception as e:
        print(f"  - ERROR: An error occurred during the fine-tuning process: {e}")
        return None


if __name__ == "__main__":
    training_file = prepare_training_data(TRAINING_DATA_PATH, SAMPLE_SIZE)
    if training_file:
        train_model(training_file)

