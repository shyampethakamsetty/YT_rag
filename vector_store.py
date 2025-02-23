import text_embeddings
from supabase import create_client
from dotenv import load_dotenv
import os

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")


supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)


def insert_into_supabase(text_chunks,text_embeddings):
    data = [  
              {"content": chunk, "embedding": embedding} 
              for chunk, embedding in zip(text_chunks, text_embeddings)
    ]
    supabase_client.table("vectors").insert(data).execute()

def get_relevent_transcripts(query_text):
    query_embedding = text_embeddings.getEmbeddings([query_text])[0]
    response = supabase_client.rpc(
    "match_vectors",
    {"query_embedding": query_embedding, "match_threshold": 0.8, "match_count": 5}
    ).execute()
    return [match["content"] for match in response.data]
