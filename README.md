
## **YouTube Financial Sentiment Analysis**  

### **Overview**  
This project is designed to retrieve, process, and analyze financial sentiment from various sources, starting with **YouTube transcripts**. It leverages **vector databases**, embeddings, and agents for data retrieval and organization.  

### **Features**  
- **YouTube Transcripts Retrieval**: Extracts English transcripts (or auto-generated ones) along with metadata and comments.  
- **Text Processing & Chunking**: Processes large transcripts into manageable chunks.  
- **Embeddings & Vector Storage**: Converts text into embeddings and stores them in a **vector database (Qdrant)** for efficient retrieval.  
- **Graph-Based Insights**: Plans to integrate **Neo4j** for analyzing market relationships, sentiment, and discussions on financial symbols.  
- **Multi-Source Expansion**: Future updates will include sentiment analysis from **Twitter (X), Reddit, and financial blogs**.  

### **Folder & File Structure**  
- **`main.py`** – Entry point for running the project.  
- **`youtube_transcripts.py`** – Extracts YouTube video transcripts, metadata, and comments.  
- **`text_embeddings.py`** – Generates embeddings from text data using **Cohere embeddings**.  
- **`vector_store.py`** – Handles storage and retrieval from the **Qdrant vector database**.  
- **`chunker.py`** – Splits large texts into meaningful chunks for better processing.  
- **`agent.py`** – Implements **CrewAI agents** for retrieving and storing data efficiently.  
- **`requirements.txt`** – Lists dependencies required for the project.  

### **Installation & Setup**  
1. **Clone the repository**  
   ```sh
   git clone <repo-url>
   cd <repo-folder>
   ```
2. **Install dependencies**  
   ```sh
   pip install -r requirements.txt
   ```
3. **Run the main script**  
   ```sh
   python main.py
   ```

### **Future Enhancements**  
✅ **Integrate Twitter (X), Reddit, and blogs** for sentiment analysis.  
✅ **Graph-based relationships in Neo4j** for stock trends & discussions.  
✅ **Stock price retrieval** from yFinance into a time-series database.  

### **Contributing**  
Feel free to submit pull requests or raise issues for suggestions!  
