Neural-Document-Retriever 📋
Industry-grade Multi-Document RAG with Hybrid Search & Neural Reranking

Neural-Doc-Retriever (formerly PaperTrail) is a sophisticated Retrieval-Augmented Generation (RAG) platform built for deep analysis of technical PDFs. It moves beyond "naive RAG" by implementing a multi-stage retrieval pipeline used in production-level search engines.

🚀 Key Features
Dual-Engine Hybrid Search: Combines semantic vector search (ChromaDB) with exact keyword matching (BM25).

RRF Fusion: Merges search results using Reciprocal Rank Fusion for optimal context retrieval.

Neural Reranking: Integrates Cohere Rerank 3.5 to filter and prioritize the most relevant document chunks.

Powered by Gemini: Utilizes Gemini 2.0 Flash for high-speed, 1-million-token context window generation and Gemini Embeddings for asymmetric document-query mapping.

Semantic Chunking: Structure-aware parsing that respects paragraph boundaries and document hierarchy.

Hard Citations: Every answer includes verifiable source mapping: [Source: filename, p.N].

🛠️ Tech Stack
LLM: Google Gemini 2.0 Flash

Embeddings: Google text-embedding-004

Reranker: Cohere Rerank 3.5

Vector Store: ChromaDB

Frontend: Streamlit

Parsing: PyMuPDF + Tesseract OCR

📦 Installation
Clone the repository:

Bash
git clone https://github.com/Bhoomi-jain/Neural-Doc-Retriever.git
cd Neural-Doc-Retriever
Set up Virtual Environment:

Bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
Configure Environment Variables:
Create a .env file in the root directory:

Code snippet
GOOGLE_API_KEY=your_gemini_api_key
COHERE_API_KEY=your_cohere_api_key
Install System Dependencies (for OCR):

Bash
sudo apt update
sudo apt install tesseract-ocr
🖥️ Usage
Run the application:

Bash
streamlit run app.py
Upload your PDFs via the sidebar.

Activate the documents you wish to query.

Chat with your documents. The system will rewrite your queries for better retrieval and provide cited answers.

🛡️ Privacy & Security
This project is configured with a .gitignore to prevent the accidental leakage of API keys. Ensure your .env file is never committed to version control.

Developed by Bhoomi Jain
