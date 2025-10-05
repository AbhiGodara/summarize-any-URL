import streamlit as st
import validators
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
import os
from dotenv import load_dotenv
import sys
import whisper
import yt_dlp
import time

load_dotenv()
groq_api_key=os.getenv("GROQ_API_KEY")
os.environ["LANGSMITH_TRACING"]="false"
llm=ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)

prompt = """
    Provide a concise and comprehensive summary of the following text in English, 
    capturing the key points and main ideas in about 300 words.
    Text: {text}
"""
prompt_template=PromptTemplate(input_variables=["text"], template=prompt)

st.set_page_config(page_title="Langchain: Summarize your URL", page_icon="🦜")
st.title("🦜 Summarize anything with your URL")

url=st.text_input("URL", label_visibility="collapsed", placeholder="Enter your URL here")

def get_transcript_with_whisper(url, progress_bar):
    audio_filename="downloaded_audion.mp3"
    # yt-dlp options to download best audio and convert to mp3
    ydl_opts = {
        'format': 'bestaudio/best',
        # 'ffmpeg_location': 'C:/ProgramData/chocolatey/bin',
        'outtmpl': audio_filename.replace('.mp3', '.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
    }

    try:
        progress_bar.progress(20, text="Downloading audio from YouTube...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        st.error(f"Error downloading audio: {e}")
        return None

    # Transcribe with Whisper using the 'small' model
    try:
        progress_bar.progress(40, text="Loading Whisper model...")
        model = whisper.load_model("small")
        
        progress_bar.progress(60, text="Translating audio to English... (This may take a moment)")
        # *** Key Change: Set task to 'translate' to get English output ***
        result = model.transcribe(audio_filename, task='translate', fp16=False)
        transcript_text = result["text"]
        
        os.remove(audio_filename)
        return transcript_text
    except Exception as e:
        st.exception(f"Error during transcription: {e}")
        if os.path.exists(audio_filename):
            os.remove(audio_filename)
        return None


if st.button("summarize!"):
    if not url.strip():
        st.error("Please enter your URL!!")
    elif not validators.url(url):
        st.error("please enter a valid URL. it can be any YT video url or any website url")
    else:
        progress_bar = st.progress(0, text="Starting...")
        try:
            with st.spinner("Summarizing..."):
                #loading the data from url
                content_text=""
                if "youtube.com" in url or "youtube.be" in url:
                    try:
                        content_text=get_transcript_with_whisper(url, progress_bar)
                    except Exception as e:
                        st.error(f"Error while fetching transcript: {e}")
                else:
                    progress_bar.progress(33, text="Fetching content from Website...")
                    loader=UnstructuredURLLoader(urls=[url], ssl_verify=False, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                    data=loader.load()
                    if data:
                        content_text=data[0].page_content if data else ""
                if content_text:
                    progress_bar.progress(80, text="Content Loaded. Summarizing...")
                    chain=load_summarize_chain(llm,chain_type="stuff", prompt=prompt_template)
                    docs=[Document(page_content=content_text)]
                    output_summary=chain.run(docs)
                    progress_bar.progress(100, text="Summary complete!")
                    time.sleep(1) # Brief pause to show "Complete!"
                    progress_bar.empty() # Remove the progress bar
                    st.subheader("Summary:")
                    st.write(output_summary)
                else:
                    progress_bar.empty()
                    st.error("Failed to get any content from the URL")
        except Exception as e:
            progress_bar.empty()
            st.exception(f"an Unexpected error accured:{e}")