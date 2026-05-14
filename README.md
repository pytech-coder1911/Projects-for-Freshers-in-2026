# 5 DS Projects for Freshers — PyTech Coder
## Complete Source Code

---

## Project Structure

```
ds_projects/
├── requirements.txt
├── project1_churn/
│   ├── churn_model.py      ← train XGBoost + SHAP + MLflow
│   └── churn_app.py        ← Streamlit dashboard
├── project2_yolo/
│   └── detect.py           ← YOLOv8 webcam + FastAPI
├── project3_bert/
│   └── sentiment.py        ← DistilBERT fine-tune + Gradio
├── project4_rag/
│   └── rag_assistant.py    ← RAG + LangChain + Streamlit chat
└── project5_agent/
    └── data_agent.py       ← Autonomous agent + Gradio UI
```

---

## Quick Start

```bash
# 1. Clone / download this folder
# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

# 3. Install all dependencies
pip install -r requirements.txt
```

---

## Project 1 — Customer Churn Prediction
**Stack:** XGBoost · SHAP · MLflow · Streamlit

```bash
# Download dataset from Kaggle
# https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# Place telco_churn.csv in project1_churn/

cd project1_churn
python churn_model.py          # train model
streamlit run churn_app.py     # launch dashboard
```

---

## Project 2 — Real-Time Object Detection
**Stack:** YOLOv8 · OpenCV · FastAPI

```bash
cd project2_yolo
python detect.py               # live webcam detection
python detect.py api           # start FastAPI server
# API docs: http://localhost:8000/docs
```

---

## Project 3 — BERT Sentiment Analyzer
**Stack:** DistilBERT · HuggingFace · Gradio

```bash
cd project3_bert
# Prepare CSV with columns: text, label (0/1/2)
# Edit train_from_csv path in sentiment.py
python sentiment.py            # launches Gradio demo

# Deploy free on HuggingFace Spaces:
# 1. Create account at huggingface.co
# 2. New Space → Gradio → upload sentiment.py + requirements
```

---

## Project 4 — RAG Research Assistant
**Stack:** LangChain · ChromaDB · Ollama · Streamlit

```bash
# Install Ollama: https://ollama.com
ollama pull llama3
ollama pull nomic-embed-text

cd project4_rag
mkdir documents
# Copy your PDFs into documents/

streamlit run rag_assistant.py  # web UI
python rag_assistant.py         # CLI mode
```

---

## Project 5 — Autonomous Data Agent
**Stack:** smolagents · Pandas · Plotly · Gradio

```bash
cd project5_agent
python data_agent.py              # Gradio UI
python data_agent.py eda data.csv # Quick EDA on any CSV
```

---

## Resources

| Project | Dataset / Tool |
|---------|----------------|
| Churn   | Kaggle Telco Churn |
| YOLO    | Roboflow.com (free annotation) |
| BERT    | Amazon Reviews / Twitter Sentiment |
| RAG     | Any PDF (research papers, docs) |
| Agent   | Any CSV dataset |

---

*Built for PyTech Coder — 2026 DS Fresher Guide*
