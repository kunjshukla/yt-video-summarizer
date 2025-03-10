import streamlit as st
from dotenv import load_dotenv
import yt_dlp
import os
import torchaudio
import google.generativeai as gai
from googleapiclient.discovery import build
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Load environment variables
load_dotenv()
gai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Set up YouTube Data API
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# Set up Whisper model
whisper_model_name = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(whisper_model_name)
model = WhisperForConditionalGeneration.from_pretrained(whisper_model_name)

# Prompt for AI summary
prompt_text = """You are an advanced AI assistant specialized in video summarization. Your task is to summarize a YouTube video transcript into key insights and highlights.

**Instructions:**
1. Read the provided transcript carefully.
2. Extract the **core message** of the video while removing unnecessary details.
3. Structure the output in **two sections**:
   - **Summary:** A well-structured and concise summary of the video in 500-650 words.
   - **Key Highlights:** A list of the most important points covered in the video.

**Rules for the Summary:**
- Use **clear, professional, and engaging language**.
- Keep it **fact-based** and **contextually relevant**.
- Maintain the **original intent and tone** of the speaker.
- Format in **short paragraphs** to enhance readability.

**Rules for Key Highlights:**
- Use a **bullet-point format** for easy reading.
- Each point should be a **single, impactful sentence**.
- Capture the **most valuable information, facts, or takeaways**.

**Example Output:**

**📌 Summary:**  
This video discusses the latest advancements in AI and Machine Learning, particularly in Natural Language Processing (NLP). The speaker explains how transformer-based architectures, such as GPT-4 and Gemini, have revolutionized text generation and contextual understanding. A key focus is on fine-tuning LLMs for specific tasks like code generation and automated summarization. Additionally, the video covers real-world applications of AI in healthcare, finance, and education, demonstrating how businesses leverage these models to improve efficiency and decision-making.

**🚀 Key Highlights:**
- Transformer-based architectures like GPT-4 and Gemini have advanced NLP significantly.
- Fine-tuning LLMs helps achieve **task-specific optimizations**.
- AI is transforming **healthcare, finance, and education** with automation.
- Ethical AI development remains a key challenge in the industry.
- Future AI trends include **multimodal learning** and real-time conversational agents.

Now, based on these instructions, summarize the given transcript.
"""

def get_video_id(youtube_url):
    """Extract video ID from YouTube URL."""
    try:
        ydl_opts = {}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            return info.get("id", None)
    except Exception as e:
        st.error(f"Failed to fetch video details: {e}")
        return None

def get_youtube_captions(video_id):
    """Fetch video captions from YouTube Data API."""
    try:
        captions = youtube.captions().list(part="snippet", videoId=video_id).execute()
        return True if "items" in captions and captions["items"] else False
    except Exception as e:
        print(f"Error fetching captions: {e}")
        return False

def fetch_transcript(video_id):
    """Retrieve captions transcript if available via YouTube API."""
    try:
        response = youtube.videos().list(part="snippet", id=video_id).execute()
        if "items" in response:
            return response["items"][0]["snippet"]["description"]  # Some videos include transcripts in the description
    except Exception as e:
        print(f"Error fetching transcript: {e}")
    return None

def download_audio(youtube_url, output_path="audio.mp3"):
    """Download YouTube audio using yt_dlp."""
    ydl_opts = {
        "format": "bestaudio/best",
        "extract_audio": True,
        "audio_format": "mp3",
        "outtmpl": output_path,
        "quiet": True
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        return output_path
    except Exception as e:
        print(f"Failed to download audio: {e}")
        return None

def whisper_transcribe_audio(youtube_url):
    """Transcribe YouTube audio using Whisper AI."""
    audio_path = download_audio(youtube_url)

    if not audio_path or not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load the audio file
    speech_array, sampling_rate = torchaudio.load(audio_path)
    input_features = processor(speech_array, sampling_rate=sampling_rate, return_tensors="pt").input_features

    # Generate transcription
    with torch.no_grad():
        predicted_ids = model.generate(input_features)
        transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    return transcript

def get_video_transcript(youtube_url):
    """Fetch transcript from YouTube API or transcribe using Whisper."""
    video_id = get_video_id(youtube_url)
    if not video_id:
        return None

    if get_youtube_captions(video_id):
        transcript = fetch_transcript(video_id)
        if transcript:
            st.success("Fetched transcript from YouTube.")
            return transcript

    # If no captions, use Whisper for transcription
    st.info("No captions found. Using Whisper AI for transcription.")
    return whisper_transcribe_audio(youtube_url)

def summarize_transcript(transcript_text):
    """Summarize the transcript using Gemini AI."""
    if not transcript_text:
        return "No transcript available for summarization."

    model = gai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt_text + transcript_text)

    try:
        return response.candidates[0].content.parts[0].text
    except Exception as e:
        st.error(f"Error in AI Summarization: {e}")
        return None

# Streamlit UI
st.title("YouTube Video Summarizer")
youtube_link = st.text_input("Enter the YouTube video link: ")

if youtube_link:
    video_id = get_video_id(youtube_link)
    if video_id:
        st.image(f"https://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)

if st.button("Summarize"):
    if not youtube_link:
        st.error("Please enter a valid YouTube video link.")
    else:
        transcript_text = get_video_transcript(youtube_link)

        if transcript_text:
            summary = summarize_transcript(transcript_text)
            if summary:
                st.write(summary)
            else:
                st.error("Failed to generate summary.")
