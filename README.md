# Email Spam Detector

**Model** finds high-confidence spam → **LLM** confirms real phishing/scams only → mark spam + trash.

Trained on ~72k emails (Kaggle + Enron + SpamAssassin + custom). Default threshold `SPAM_THRESHOLD=0.90` so the LLM only sees strong spam hits.

## Setup

```bash
uv sync
cp .env.example .env   # OPENAI_API_KEY, EMAIL_ADDRESS, EMAIL_PASSWORD
```

Gmail: use an [App Password](https://myaccount.google.com/apppasswords).

## Train / Run

```bash
uv run python machine_learning.py   # LR vs KNN on the big dataset
uv run python main.py               # poll every 60s
```
