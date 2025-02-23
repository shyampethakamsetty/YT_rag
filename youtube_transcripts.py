import requests
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from dotenv import load_dotenv
import os

load_dotenv()
YOUTUBE_API_KEY  = os.getenv("YOUTUBE_API_KEY")

def search_youtube(topic, max_results=5):
    search_url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={topic}&maxResults={max_results}&type=video&order=date&key={YOUTUBE_API_KEY}"
    response = requests.get(search_url).json()
    video_ids = [item["id"]["videoId"] for item in response.get("items", [])]
    return get_video_transcripts(video_ids)


def get_video_transcripts(video_ids):
    transcripts = []
    for video_id in video_ids:
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            try:
                transcript = transcript_list.find_transcript(["en"])
                print(f"✅ Transcripts found for: {video_id}")
            except NoTranscriptFound:
                try:
                    transcript = transcript_list.find_generated_transcript(["en"])
                except NoTranscriptFound:
                    print(f"❌ No English transcript (manual or auto-generated) for: {video_id}")
                    continue
            transcript_text = " ".join([t["text"] for t in transcript.fetch()])
            transcripts.append(transcript_text)
        except TranscriptsDisabled:
            print(f"❌ Transcripts are disabled for: {video_id}")
    return " ".join(transcripts)