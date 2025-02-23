import streamlit as st
from crewai import Agent, Task, Crew
import youtube_transcripts
import chunker
import text_embeddings
import vector_store
import os
from groq import Groq  # Assuming Groq API client is installed

groq_api_key = "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"
os.environ["GROQ_API_KEY"] = groq_api_key

# Streamlit UI
topic = st.text_input("Enter Topic:")
query = st.text_input("Enter Query:")

if st.button("Fetch Topic Data") and topic:
    # Define Agents with backstories
    youtube_scraper = Agent(
        name="YouTubeScraper", 
        role="A seasoned web crawler who tirelessly scours YouTube for transcripts", 
        goal="Find and extract accurate transcripts for a given topic.",
        backstory="An AI-powered researcher who has spent years gathering data from online videos to make knowledge more accessible.",
        function=youtube_transcripts.search_youtube
    )
    text_chunker = Agent(
        name="TextChunker", 
        role="A meticulous librarian who organizes transcripts into digestible parts", 
        goal="Break down transcripts into meaningful chunks for efficient processing.",
        backstory="Once a book archivist, now an AI that ensures knowledge is structured and easily retrievable.",
        function=chunker.chunk_text
    )
    embedder = Agent(
        name="Embedder", 
        role="A cutting-edge AI researcher who generates embeddings using Groq API", 
        goal="Transform text into vector embeddings for efficient semantic search.",
        backstory="A former NLP scientist, now an AI tasked with making textual data machine-readable.",
        function=text_embeddings.getEmbeddings
    )
    vector_db = Agent(
        name="VectorStore", 
        role="A data archivist who ensures embeddings are securely stored and retrievable", 
        goal="Store and retrieve embeddings effectively from the database.",
        backstory="An AI librarian who safeguards knowledge for future retrieval.",
        function=vector_store.insert_into_supabase
    )

    # Define Tasks
    scrape_task = Task(description="Scrape YouTube transcripts for topic", agent=youtube_scraper, expected_output="transcripts")
    chunk_task = Task(description="Chunk transcripts", agent=text_chunker, expected_output="chunked_text")
    embedding_task = Task(description="Generate embeddings", agent=embedder, expected_output="embeddings")
    store_task = Task(description="Store embeddings into Supabase", agent=vector_db, expected_output="stored")

    # Crew execution
    crew = Crew(agents=[youtube_scraper, text_chunker, embedder, vector_db], tasks=[scrape_task, chunk_task, embedding_task, store_task])
    crew.kickoff(inputs={"topic": topic})
    st.success("Topic data processed and stored successfully!")

if st.button("Fetch Relevant Context") and query:
    query_agent = Agent(
        name="QueryAgent", 
        role="An AI detective who retrieves the most relevant transcripts for user queries", 
        goal="Find and return the most useful information based on the query.",
        backstory="An AI investigator trained to dig through vast amounts of text and find the best insights.",
        function=vector_store.get_relevant_transcripts
    )
    retrieve_task = Task(description="Retrieve relevant transcripts for the query", agent=query_agent, expected_output="context")
    crew = Crew(agents=[query_agent], tasks=[retrieve_task])
    results = crew.kickoff(inputs={"query": query})
    st.write("Relevant Context:")
    st.write(results[retrieve_task] if retrieve_task in results else "No relevant data found")

