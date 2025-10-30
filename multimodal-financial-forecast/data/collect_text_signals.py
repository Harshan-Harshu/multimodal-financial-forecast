import os
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

NEWS_API_KEY = "6e2fd3190a1a4b7aa695197ea2c4edfd"  
TEXT_DATA_PATH = "data/raw/text/"
os.makedirs(TEXT_DATA_PATH, exist_ok=True)

def fetch_news(query="crypto OR bitcoin OR stock", from_days_ago=1):
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=from_days_ago)

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "from": start_date.strftime('%Y-%m-%d'),
        "to": end_date.strftime('%Y-%m-%d'),
        "language": "en",
        "sortBy": "publishedAt",
        "apiKey": NEWS_API_KEY,
        "pageSize": 100,
    }

    response = requests.get(url, params=params)
    data = response.json()

    if "articles" not in data:
        raise Exception(f"❌ Error fetching news: {data}")

    articles = data["articles"]
    news_df = pd.DataFrame([{
        "source": a["source"]["name"],
        "title": a["title"],
        "description": a["description"],
        "publishedAt": a["publishedAt"],
        "url": a["url"]
    } for a in articles])

    file_path = os.path.join(TEXT_DATA_PATH, f"news_{start_date.date()}_{end_date.date()}.csv")
    news_df.to_csv(file_path, index=False)
    print(f"✅ News saved to {file_path}")

    return news_df


def fetch_reddit_stub():
    print("🛑 Reddit data collection not implemented yet.")
    print("🔧 To implement, use PRAW (Reddit API) or Pushshift.io API.")
    return pd.DataFrame()


def fetch_tweets_stub():
    print("🛑 Twitter/X data collection requires paid API access.")
    print("🔧 Use Tweepy or Twitter API v2 with Bearer Token if available.")
    return pd.DataFrame()


if __name__ == "__main__":
    print("📥 Collecting financial text signals...")

    news_df = fetch_news()

    reddit_df = fetch_reddit_stub()
    twitter_df = fetch_tweets_stub()
