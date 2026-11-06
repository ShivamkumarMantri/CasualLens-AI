# Deployment checklist

## Local
- Python 3.10+
- Install `requirements.txt`
- Set `GOOGLE_API_KEY` as an environment variable or enter it in the app.
- Run `streamlit run app.py`.

## GitHub
```bash
git init
git add .
git commit -m "Initial CausalLens AI research prototype"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/CausalLens-AI.git
git push -u origin main
```

## Streamlit Community Cloud
- Select the GitHub repository and `app.py`.
- Add secrets in the Streamlit app settings:
```toml
GOOGLE_API_KEY = "your-key"
GEMINI_MODEL = "gemini-2.5-flash"
```
- Do not commit secrets or real personal datasets.

## Before public release
Add authentication, rate limiting, upload size limits, data retention controls, robust categorical encoding, logging without sensitive values, and managed storage if persistence is required.
