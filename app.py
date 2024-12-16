import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled
load_dotenv()
import json
from pytube import YouTube
import os
import google.generativeai as gai
from youtube_transcript_api import YouTubeTranscriptApi

gai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

prompt_text = """You are Yotube video summarizer. You will be taking the transcript text
and summarizing the entire video and providing the important summary in points
within 250 words. Please provide the summary of the text given here:"""

def extract_transcript_details(yt_video_url, lang="hi"):
    try:
        video_id = yt_video_url.split("=")[1]


        # transcript_text = YouTubeTranscriptApi.get_transcript(video_id)
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
        except TranscriptsDisabled as e:
            print(f"Error: Subtitles are disabled for the video. Transcript cannot be retrieved. Video ID: {e.video_id}")


        
        transcript = ""

        
        
        for i in transcript_text:
            transcript += " " + i["text"]        
        return transcript

    except Exception as e:
        raise e




def download_audio(youtube_link, output_path="audio.mp3"):
    yt = YouTube(youtube_link)
    audio_stream = yt.streams.filter(only_audio=True).first()
    audio_stream.download(filename=output_path)
    print(f"Audio downloaded to {output_path}")
    return output_path



def generate_gemini_content(transcript_text, prompt_text):

    
    model = gai.GenerativeModel('gemini-pro')
    print("Model loaded")
    combined_text = prompt_text + transcript_text
    response = model.generate_content(combined_text)
    print("Response received" ,response)
        
    # Access the generated text
    candidate = response.candidates[0]  # Access the first candidate
    content = candidate.content.parts[0].text  # Extract the text part
        
    return content  # Return the summary text directly
    
st.title("Youtube Video Summarizer")
youtube_link = st.text_input("Enter the Youtube video link: ")

if youtube_link:
    video_id = youtube_link.split("=")[1]
    print(video_id)
    st.image(f"https://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)

if st.button("Summarize"):
    transcript_text = extract_transcript_details(youtube_link)

    if transcript_text:
        summary = generate_gemini_content(transcript_text, prompt_text)
        st.write(summary)
