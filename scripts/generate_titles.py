"""
Generate YouTube titles using the fine-tuned model.
Uses the exact same prompts as training for consistency.
"""
import os
import sys
from openai import OpenAI
from dotenv import load_dotenv
from typing import Tuple

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Fine-tuned model ID
FINE_TUNED_MODEL = 'ft:gpt-4.1-mini-2025-04-14:personal:title-gen:CWNBrUzt'

# Exact same system prompt as training
SYSTEM_PROMPT = """You are an expert YouTube title generator. Your task is to create compelling, accurate titles that capture the essence of video content based on transcripts.

Guidelines for generating YouTube titles:
1. Be accurate and truthful - the title must reflect the actual content of the video
2. Make it compelling and click-worthy while staying honest about the content
3. Keep titles concise (ideally 50-80 characters, max 100 characters)
4. Use natural language that viewers would search for
5. Highlight the most interesting or valuable aspect of the video
6. Capture curiosity gaps

Generate only the title itself, nothing else."""


def generate_title(transcript: str, temperature: float = 0.7) -> str:
    """
    Generate a YouTube title using the fine-tuned model.
    Uses the exact same prompts as training.
    
    Args:
        transcript: The video transcript text
        temperature: Temperature for generation (0.7 for variety, 0.0 for deterministic)
        
    Returns:
        Generated title string, or None if error
    """
    # Exact same user message format as training
    user_message_content = f"""Based on the following video transcript, generate an accurate and compelling YouTube title that captures the essence of the content.

Transcript:
{transcript}

Generate the YouTube title:"""
    
    try:
        response = client.chat.completions.create(
            model=FINE_TUNED_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message_content}
            ],
            temperature=temperature,
            max_tokens=100
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"ERROR generating title: {e}")
        return None


def generate_multiple_titles(transcript: str, num_titles: int = 5) -> list[str]:
    """
    Generate multiple title variations for the same transcript.
    Calls the fine-tuned model multiple times to get diverse suggestions.
    
    Args:
        transcript: The video transcript text
        num_titles: Number of title variations to generate (default: 5)
        
    Returns:
        List of generated titles
    """
    print(f"\n{'='*60}")
    print(f"Generating {num_titles} title variations...")
    print(f"{'='*60}\n")
    
    titles = []
    for i in range(num_titles):
        print(f"Generating title {i+1}/{num_titles}...")
        title = generate_title(transcript, temperature=0.7)
        if title:
            titles.append(title)
            print(f"  ✓ {title}\n")
        else:
            print(f"  ✗ Failed to generate title\n")
    
    print(f"{'='*60}")
    print(f"Generated {len(titles)} titles:")
    print(f"{'='*60}")
    for i, title in enumerate(titles, 1):
        print(f"{i}. {title}")
    print(f"{'='*60}\n")
    
    return titles


def main():
    """Main function to handle transcript input."""
    if len(sys.argv) > 1:
        # Transcript provided as command line argument
        transcript = " ".join(sys.argv[1:])
    else:
        # Example transcript for testing
        print("No transcript provided. Using example transcript...")
        transcript = """
1. Introduction: The Big Idea
The Large Language Model is one of the most powerful ideas in technology. We've seen it pass the bar exam, ace medical licensing tests, and even discover new drug-like molecules from scratch.
I see it used for all these world-changing, genius-level purposes. But using it, I wondered about something simpler, something more... selfish.
Can it make me go viral?
2. The Problem: How to Teach "A Good Title"
How do we even begin to make a machine learn something as subjective as what makes a YouTube title good?
We can't just give it a textbook of "YouTube Best Practices." (turn to the side, and say “a book like that doesn’t exist, but in my opinion, it’s one that makes you curious”)
We also can't code every condition that actually makes a human curious, because "clickability" is a gut feeling, an abstract concept.
And so what we need is an approach that can recognize patterns. One that can actually look at thousands of titles that worked and say, "Oh, that's the pattern. Let me learn from that."
Much like I try to do every time I stare at a blank "Upload" page for an hour.
And I think this type of problem is exactly where fine-tuning comes in.

3. The Concept: What is Fine-Tuning?
Now, it's actually a very simple concept.
To demonstrate, let's imagine a world-class musician. This musician is a technical genius. They’ve spent their life learning every scale, every chord, and all of music theory. They can read and play any piece of sheet music you put in front of them, perfectly.
This musician is our "base model" AI. It knows the "rules" of language inside and out.
Now, what happens if I ask this musician to play improvisational jazz?
[Visual: A classical musician playing a very stiff, robotic "jazz" tune, technically correct but with no soul.]
They can try. They'll follow the rules of jazz, but it's going to sound stiff. It's not in their "style."
So, fine-tuning is when I take this genius musician, I put them in a room, and I make them listen to nothing but Miles Davis albums for a month.
We're not teaching them how to play the piano. They already know that. We're teaching them how to swing. We're forcing them to internalize a new, specific style until it becomes second nature.
Theoretically, they stop sounding like a classical player trying to play jazz, and they just start sounding like a jazz player.
What we're building is the exact same premise. Except instead of a musician, we have a massive AI.
And instead of Miles Davis albums... we have a playlist of all of MrBeast's greatest hits.
4. The Build: Getting the "Sheet Music"
This leads us to our first problem. What is the perfect dataset of "good" titles?
I could have just scraped a few of my favorite channels, but that's a tiny, biased dataset. That’s just my opinion.
If we want this to work, we need to know what millions of people are actually clicking on. We need data. Lots of data.
And after some searching, I found my answer on Kaggle.
[Visual: Screen recording of the Kaggle dataset page you sent]
A "YouTube Trending Video Dataset." It's a massive, daily-updated archive of the top 200 trending videos, from 11 different countries, stretching back for years.
It has everything: the video title, channel title, publish time, tags, views, likes, dislikes, and comment count.
But that was the next problem. It had everything.
[Visual: Rapid montage of different video types: a music video, a sports highlight, a movie trailer, a vlog]
The dataset was 4.5 gigabytes of noise. I needed to filter this massive file down to only the videos that actually matched the content I make.
After a quick look, I found my category: ID 28, "Science & Technology."
So, I wrote this Python script.
[Visual: The Python script from the user's prompt scrolls by on screen]
Its job was simple. It read that massive 4.5-gigabyte file, but in small chunks so my computer wouldn't... well, melt.
It went through every single entry and did two things:
Find only the videos marked categoryId == 28.
Strip out all the duplicates.
After the script ran, I was left with 1,425 unique, trending, "Science & Technology" videos.
5. The Build: Cleaning the "Menu"
I was ready to start training, oh so I thought.
But when I started scrolling through... I saw a new problem.
[Visual: Scrolling through the CSV, highlighting titles like "Apple Event," "Galaxy Unpacked," and "Starlink Mission"]
My dataset was contaminated.
It was full of corporate events, rocket launches, and product announcements. These videos didn't trend because of a clever title; they trended because they're news.
Apple could title their video "Gray Box" and it would get 10 million views. If my AI learned from this, it would just learn to write boring, official-sounding titles.
I needed to clean the data.
My first thought was to do it manually. [Visual: You, with a huge cup of coffee, looking at the 1,425-row spreadsheet in despair]. Just... go through them, one by one, and mark them "KEEP" or "REMOVE."
It seemed incredibly tedious. And then I thought...
...why would I do the boring, repetitive work... when I could make an AI do it for me?
6. The "AI Bouncer"
So, I did something better. I wrote a new Python script.
But this one was different. It imported Gemini, a large language model, to act as the "bouncer" for my dataset.
[Visual: The Python script's FILTERING_CRITERIA scrolls by, highlighting the 🗑️ REMOVE and 💎 KEEP sections]
I gave the AI a very specific job. I wrote a prompt that defined my "gold standard."
I told it:
🗑️ REMOVE corporate ads, livestreams, and "deal guy" content.
💎 KEEP videos with conflict, absurd engineering, or "big questions."
Then, I just let it run.
[Visual: Terminal/console view, showing the tqdm progress bar zipping across as the script logs its decisions]
The script looped through all 1,425 videos. It fed the title, channel, and tags to Gemini. And Gemini sent back its decision, over and over.
"REMOVE: This is a corporate event." "KEEP: This title shows clear conflict." "REMOVE: This is a simple broadcast."
It was like having a super-fast, perfectly consistent intern. It filtered the entire dataset in the time it would have taken me to get through the first 50.
When it was done, it saved a new, clean file: category_28_videos_filtered.csv.
This was the dataset. This was what we would use for training.
7. The Result: Did It Work?
So... did it work?
The first test was on a new video I was editing. I fed the transcript to my new model. It gave me a title. I used it.
The video got a 15% click-through rate from my subscribers.
[Visual: A simple, clean graph showing a 15% CTR bar, towering over your channel's "average" 5-8% bar]
That is... insanely high. Higher than I have ever seen.
The AI wasn't just good. It had perfectly captured the style I was aiming for.
But I wasn't done. Now I had this powerful new tool. What if I unleashed it... on my entire channel?
8. The Finale: "Fixing" My Old Videos
I wrote one last script.
This one was the final boss. It connected to my YouTube channel, fetched all of my old videos, and ripped every single transcript.
Then, one by one, it fed them to my new specialist AI.
And in return... it gave me a list of every title I should have used.
[Visual: The final JSON output scrolls by, showing the "Original" vs. "Recommended" titles. The following can be B-roll text on screen.]
Original: "I Can't Believe How Bad This AI Actually Is" AI Recommended: "I let ChatGPT control my browser." (Wow. That's just... so much better. It's what the video is actually about.)
Original: "AI Orders a Pizza in 17:32 [any% glitchless]" AI Recommended: "AI vs Human Pizza Speedrun" (The AI found the conflict, the real hook.)
Original: "Spotify's Shuffle is a Lie." AI Recommended: "The Truth About Spotify Shuffle" (It knew the original was already pretty good, but just made it cleaner.)
Original: "I Trained an AI to Make Me Laugh... It Worked Too Well" AI Recommended: "Building an AI to make me laugh" (It even... it even fixed the title for the original video. It's more direct. No fluff. I... I'm satisfied.)
[Music fades in]
It's not just a generic "Top 10" list. It learned my style. It learned curiosity.




"""
    
    print(f"\nUsing fine-tuned model: {FINE_TUNED_MODEL}")
    print(f"\nTranscript length: {len(transcript)} characters")
    print(f"\nTranscript preview:")
    print(f"{'-'*60}")
    print(transcript[:200] + "..." if len(transcript) > 200 else transcript)
    print(f"{'-'*60}")
    
    generate_multiple_titles(transcript, num_titles=5)


if __name__ == "__main__":
    main()




