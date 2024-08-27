import streamlit as st
from dotenv import load_dotenv

load_dotenv()
import json
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


        transcript_text = YouTubeTranscriptApi.get_transcript(video_id)


        
        transcript = ""

        print(transcript_text)
        
        for i in transcript_text:
            transcript += " " + i["text"]        
        return transcript

    except Exception as e:
        raise e


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
    st.write(transcript_text)

    if transcript_text:
        summary = generate_gemini_content(transcript_text, prompt_text)
        st.markdown("##summary")
        st.write(summary)