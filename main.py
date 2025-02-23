import streamlit as st
import youtube_transcripts
import chunker
import text_embeddings
import vector_store
from agent import AI_Agent

st.title("YouTube Transcript Search & AI Chat")

search_query = st.text_input("Enter a YouTube search term:")

if st.button("Search & Store"):
    with st.spinner("Fetching and processing transcripts..."):
        text = youtube_transcripts.search_youtube(search_query)
        text_chunks = chunker.chunk_text(text)
        embeddings = text_embeddings.getEmbeddings(text_chunks)
        vector_store.insert_into_supabase(text_chunks, embeddings)
        st.success("Transcripts stored successfully!")

query = st.text_input("Enter your query:")

if st.button("Get Answer"):
    with st.spinner("Retrieving relevant transcripts..."):
        relevant_transcripts = vector_store.get_relevent_transcripts(query)
        context = "\n\n".join(relevant_transcripts)
        agent = AI_Agent()
        response = agent.answer(query, context)
        st.write("### AI Response:")
        st.write(response)
