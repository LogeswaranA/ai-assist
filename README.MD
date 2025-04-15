# PLUGIN AI ASSIST

IT HAS KNOWLEDGE ABOUT PLUGIN ECO SYSTEM AND COMPLETE DOCS & KNOWLEDGEBYTES. ASK ANYTHING ABOUT PLUGIN AND IT ANSWERS PRECISELY

# How to prepare knowledge for the agent

- * Use [firecrawl](https://llmstxt.firecrawl.dev/)
- * paste the URL, that you want skimthrough and generate LLMText file
- * Download that file into this directory
- * copy the content from the file and append that into llms.txt
- * run the app.py to re-generate the faiss index & pkl file

## INPUTS REQUIRED
create .env
add OPENAI_API_KEY
```
OPENAI_API_KEY=
```

## Installation

```
python3 -m venv envs
source env/bin/activate
pip install -r requirements.txt
```

## Execute backend

```
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Execute Frontend

```
streamlit run run.py    
```