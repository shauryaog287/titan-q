import feedparser
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")

print("--- 1. Waking up the FinBERT NLP AI (This takes a moment...) ---")
# Load FinBERT from HuggingFace (it will download the weights if it's the first time)
nlp = pipeline("sentiment-analysis", model="ProsusAI/finbert")

print("--- 2. Connecting to Live Indian Financial News Feed ---")
# We use a live RSS feed searching for Indian Stock Market news in the last 24 hours
rss_url = "https://news.google.com/rss/search?q=NSE+OR+BSE+OR+Indian+Stock+Market+when:1d&hl=en-IN&gl=IN&ceid=IN:en"
feed = feedparser.parse(rss_url)

print(f"Found {len(feed.entries)} live articles. Analyzing the top 50...")
print("\n========================================================")
print(" LIVE MARKET SENTIMENT SCANNER")
print("========================================================\n")

for i, entry in enumerate(feed.entries[:50]):
    headline = entry.title
    # The AI reads the headline and outputs a score
    result = nlp(headline)[0]
    sentiment = result['label'].upper()
    confidence = result['score'] * 100
    
    if sentiment == "POSITIVE":
        icon = "[+]"
    elif sentiment == "NEGATIVE":
        icon = "[-]"
    else:
        icon = "[=]"
        
    print(f"{icon} {sentiment} ({confidence:.1f}% confidence)")
    print(f"Headline: {headline}")
    print("-" * 55)
