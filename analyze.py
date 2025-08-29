import traceback
import google.generativeai as genai
from PIL import Image
import io
import os

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")
genai.configure(api_key=gemini_api_key)

def get_llm_response(image_data: bytes) -> str:
    # implement the call to the Gemini API here
    # docs: https://ai.google.dev/gemini-api/docs/text-generation
    try:
        image = Image.open(io.BytesIO(image_data))
        prompt_text = "Provide a detailed description of the image."
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content([prompt_text, image])
        return response.text
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()  
        return f"An error occured"

    
