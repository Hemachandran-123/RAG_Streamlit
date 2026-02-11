# RAG_Streamlit# 📄 Multi-Document RAG Q&A System

AI-powered document question-answering system using Retrieval-Augmented Generation (RAG) architecture.

## 🚀 Features

- **Upload documents** (PDF, DOCX, TXT)
- **Ask questions** in natural language
- **Get accurate answers** with source citations
- **View source chunks** for transparency
- **Multi-query retrieval** for better context

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Embeddings:** BAAI/bge-small-en-v1.5
- **LLM:** Qwen3-4B-Instruct (via Hugging Face API)
- **Vector Database:** ChromaDB
- **NLP Processing:** spaCy, Unstructured

## 📦 Installation

### Prerequisites
- Python 3.8+
- Hugging Face account (free)

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/YOUR_USERNAME/rag-qa-system.git
cd rag-qa-system
```

2. **Create virtual environment:**
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

4. **Set up environment variables:**

Create a `.env` file:
```
HF_TOKEN=your_huggingface_token_here
```

5. **Run the app:**
```bash
streamlit run app.py
```

Open browser at `http://localhost:8501`

## 💡 How It Works

1. **Document Processing:** Upload a document → Extract text → Smart chunking
2. **Embedding Creation:** Convert chunks to vectors using BAAI embeddings
3. **Vector Storage:** Store in ChromaDB for semantic search
4. **Query Processing:** Multi-query expansion for better retrieval
5. **Answer Generation:** LLM generates answers grounded in retrieved context

## 📊 Architecture
```
User Upload → Text Extraction → Chunking → Embeddings → ChromaDB
                                                              ↓
User Query → Multi-Query → Retrieval → LLM → Answer + Citations
```

## 🎯 Use Cases

- Research paper analysis
- Legal document review
- Medical report Q&A
- Customer support knowledge base
- Educational content exploration

## 🔧 Configuration

Key parameters in `app.py`:
- `EMBEDDING_MODEL`: BAAI/bge-small-en-v1.5
- `LLM_MODEL`: Qwen/Qwen3-4B-Instruct-2507
- Chunk size: 1000 characters
- Chunk overlap: 200 characters

## 📝 Project Structure
```
rag-qa-system/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (not in repo)
├── .gitignore            # Git ignore rules
└── README.md             # Project documentation
```

## 🚀 Deployment

Deploy on Streamlit Cloud:
1. Push to GitHub
2. Go to https://share.streamlit.io
3. Connect repository
4. Add `HF_TOKEN` to Secrets
5. Deploy!

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License - feel free to use for your projects!

## 👤 Author

**Hemachandran M**
- GitHub: https://github.com/Hemachandran-123

## 🙏 Acknowledgments

- Hugging Face for free model hosting
- ChromaDB for vector storage
- Streamlit for easy deployment
