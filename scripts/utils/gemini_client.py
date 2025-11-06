# scripts/utils/gemini_client.py
import os
from google import generativeai as genai
from dotenv import load_dotenv
from PIL import Image
import io
import base64

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash-lite')


def get_ai_thumbnail_description(image_bytes: bytes) -> str:
    """
    Uses the Gemini vision model to analyze an image and create a detailed
    prompt for generating a similar thumbnail, using the base64 method.
    """
    try:
        # Convert raw image bytes to a PIL Image object to ensure it's a valid image
        # and to get the format (e.g., JPEG, PNG)
        image = Image.open(io.BytesIO(image_bytes))
        image_format = image.format or 'JPEG'  # Default to JPEG if format is not detected

        # Convert the raw bytes to a base64 encoded string
        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        prompt = """
        You are a world-class YouTube thumbnail designer. Analyze the provided thumbnail image and create a detailed prompt for an AI image generator (like DALL-E 3 or Midjourney) to create a similar, high-impact thumbnail.

        Your description must capture:
        1.  **Style:** Is it photorealistic, illustrated, 3D rendered, cartoonish? Describe the overall aesthetic.
        2.  **Composition:** Describe the main subject, their pose, and expression. Where are they in the frame? What is in the background?
        3.  **Text:** If there is text, what does it say? Describe the font style (bold, sans-serif, etc.) and color.
        4.  **Color & Lighting:** Describe the dominant colors, the mood, and the lighting (e.g., dramatic, bright, high contrast).
        5.  **Overall Vibe:** What is the overall feeling? (e.g., chaotic, mysterious, exciting, funny).

        Generate a single, detailed paragraph that could be used as a prompt.
        """

        response = model.generate_content(
            [
                prompt,
                {
                    "inline_data": {
                        "mime_type": f"image/{image_format.lower()}",
                        "data": base64_image
                    }
                }
            ]
        )
        
        return response.text.strip()
    except Exception as e:
        return f"ERROR: Gemini vision call failed: {e}"


def get_text_response(prompt: str) -> str:
    """
    Uses Gemini to generate a text response from a text prompt.
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"ERROR: Gemini text call failed: {e}"

