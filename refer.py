#=============================================== IMPORT =============================================================

import streamlit as st
import json
import requests
import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import psycopg2
from nltk.tokenize import sent_tokenize
import openai
from crewai import Agent, Task, Crew  # Import CrewAI

#===================================================================================================================

#============================================== LOADING ENV VARIABLES ==================================================

load_dotenv()
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
OPENAI_API_KEY = os.getenv("TOGETHER_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

#====================================================================================================================

MAX_CHAR = 1500 
CHAR_OVERLAP = 50  

#============================================== DATABASE =========================================================

def get_db_connection():
    return psycopg2.connect(SUPABASE_DB_URL)

#==================================================================================================================

#========================================== SEARCHING YOUTUBE TRANSCRIPTS ===========================================

def search_youtube(topic, max_results=20):
    if not YOUTUBE_API_KEY:
        st.error("YouTube API key not found. Please check your .env file.")
        st.stop()
    search_url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={topic}&maxResults={max_results}&type=video&order=date&key={YOUTUBE_API_KEY}"
    response = requests.get(search_url).json()
    return [item["id"]["videoId"] for item in response.get("items", [])]

#===================================================================================================================

#========================================== PULL TRANSCRIPTS ======================================================

def get_video_transcripts(video_ids):
    transcripts = []
    for video_id in video_ids:
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            try:
                transcript = transcript_list.find_transcript(["en"])
            except NoTranscriptFound:
                try:
                    transcript = transcript_list.find_generated_transcript(["en"])
                except NoTranscriptFound:
                    print(f"No English transcript (manual or auto-generated) for: {video_id}")
                    continue
            transcript_text = " ".join([t["text"] for t in transcript.fetch()])
            transcripts.append(transcript_text)
        except TranscriptsDisabled:
            print(f"Transcripts are disabled for: {video_id}")
    return " ".join(transcripts) 

#====================================================================================================================

#========================================= CHUNKING =================================================================

def chunk_text(text, max_char, char_overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_char, len(text))
        chunks.append(text[start:end])
        start += max_char - char_overlap
    return chunks

#======================================================================================================================

#============================================ QUERY TO EMBEDDING ===================================================

def get_together_embedding(text):
    url = "https://api.together.xyz/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"model": "togethercomputer/m2-bert-80M-8k-retrieval", "input": text}
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["data"][0]["embedding"]
    else:
        return None

#=====================================================================================================================

#==================================== STORING VECTOR EMBEDDINGS =======================================================

def store_embeddings(text):
    chunks = chunk_text(text, MAX_CHAR, CHAR_OVERLAP)
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            for chunk in chunks:
                embedding = get_together_embedding(chunk)
                if embedding:
                    cursor.execute(
                        """
                        INSERT INTO youtube_transcripts (transcript_chunk, embedding)
                        VALUES (%s, %s)
                        """,
                        (chunk, json.dumps(embedding))
                    )
            conn.commit()

#===================================================================================================================

#=========================================== PULL RELEVANT RECORDS ==================================================

def search_relevant_transcripts(query, top_k=5):
    query_embedding = get_together_embedding(query)
    if not query_embedding:
        return []
    search_query = """
        SELECT transcript_chunk, 1 - (embedding <=> %s) AS similarity
        FROM youtube_transcripts
        ORDER BY similarity DESC
        LIMIT %s;
    """
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(search_query, (json.dumps(query_embedding), top_k))
            results = cursor.fetchall()
    return [row[0] for row in results]

#===================================================================================================================

#=============================================== CREWAI AGENTS ==========================================================

research_agent = Agent(
    role='Research Agent',
    goal='Search and scrape YouTube transcripts based on the given topic',
    backstory='An expert in searching and extracting relevant information from YouTube videos.',
    verbose=True
)

chat_agent = Agent(
    role='Chat Agent',
    goal='Provide accurate and concise answers to user queries based on the context provided',
    backstory='A knowledgeable assistant that uses the context from YouTube transcripts to answer questions.',
    verbose=True
)

#=====================================================================================================================

#================================================ STREAMLIT UI =============================================================

st.title("YouTube Topic Chat")
topic = st.text_input("Enter topic:")

if st.button("Search & Scrape"):
    if not topic.strip():
        st.warning("Please enter a topic before searching.")
        st.stop()
    
    def research_transcripts(topic):
     video_ids = search_youtube(topic)
     return get_video_transcripts(video_ids)

    research_task = Task(
    description=f'Search and scrape YouTube transcripts for the topic: {topic}',
    agent=research_agent,
    expected_output='A concatenated string of all relevant YouTube transcripts.',
    function=research_transcripts,  # This tells CrewAI to run this function!
    function_kwargs={'topic': topic}
)

    
    crew = Crew(
        agents=[research_agent],
        tasks=[research_task],
        verbose=True
    )
    
    result = crew.kickoff().tasks_output[0].raw
    
    if result:
        store_embeddings(result)
        st.success("Transcripts processed and stored!")
    else:
        st.warning("No transcripts found.")

st.subheader("Chat with the RAG Agent")
query = st.text_input("Ask a question:")
if st.button("Get Answer"):
    if not query.strip():
        st.warning("Please enter a question.")
        st.stop()
    
    relevant_chunks = search_relevant_transcripts(query)
    if not relevant_chunks:
        st.warning("No relevant transcripts found in the vector store.")
        st.stop()
    
    context = " ".join(relevant_chunks)
    
    chat_task = Task(
        description=f'Answer the user query: {query} based on the context provided.',
        agent=chat_agent,
        expected_output='A concise and accurate answer to the user query.'
    )
    
    crew = Crew(
        agents=[chat_agent],
        tasks=[chat_task],
        verbose=True
    )
    
    response = crew.kickoff().tasks_output[0].raw
    st.write(response)
