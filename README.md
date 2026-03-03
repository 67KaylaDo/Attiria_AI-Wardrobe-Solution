# Attiria – AI Outfit Recommender

Attiria is an AI-powered personal styling assistant built with Streamlit, LangChain, and Google Gemini.

It generates personalized outfit recommendations based on:
- Body type
- Skin tone
- Style preference
- Occasion
- Budget

---

## Tech Stack

- Python 3.11
- Streamlit
- LangChain
- Google Gemini API
- Pandas
- python-dotenv

---

## Project Structure

GenAI_project_lang/
│
├── assets/
│   ├── attiria_logo.jpeg
│   └── hero_banner.jpeg
│
├── streamlit_app_lang.py
├── recommender_lang.py
├── llm_clients_lang.py
├── catalog.csv
├── requirements.txt
└── README.md

---

## Run Locally

1. Create virtual environment:

python -m venv .venv
source .venv/bin/activate  # Mac/Linux

2. Install dependencies:

pip install -r requirements.txt

3. Create a .env file:

GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=models/gemini-1.5-flash

4. Run the app:

streamlit run streamlit_app_lang.py

---

## Deploy on Streamlit Cloud

1. Push project to GitHub
2. Deploy via https://share.streamlit.io
3. Add secrets in Settings → Secrets:

GEMINI_API_KEY="your_api_key_here"
GEMINI_MODEL="models/gemini-1.5-flash"

---

## Notes

- Image generation may require paid quota.
- Text-only mode works on free tier.
- .env file should NOT be committed to GitHub.

---

