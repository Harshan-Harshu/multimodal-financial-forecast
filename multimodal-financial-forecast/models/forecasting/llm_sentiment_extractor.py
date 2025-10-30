from transformers import pipeline

def extract_sentiment(texts):
    sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    results = sentiment_pipeline(texts)
    return [r['label'] for r in results]
