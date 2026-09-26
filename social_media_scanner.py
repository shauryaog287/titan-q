import time
import json
import logging
from transformers import pipeline
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SocialMediaScanner:
    def __init__(self):
        logging.info("Loading FinBERT for Social Media Sentiment...")
        self.nlp = pipeline("text-classification", model="ProsusAI/finbert")
        
        self.options = Options()
        self.options.add_argument("--headless")
        self.options.add_argument("--window-size=1920,1080")
        self.options.add_argument("--disable-blink-features=AutomationControlled")
        self.options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36")
        
        # Share the exact same driver manager setup as the NSE Scraper
        self.service = Service(ChromeDriverManager().install())
        
    def scan_reddit(self, subreddit="IndianStreetBets", limit=15):
        logging.info(f"Stealth scraping r/{subreddit} for retail sentiment using Old Reddit...")
        url = f"https://old.reddit.com/r/{subreddit}/new/"
        
        driver = webdriver.Chrome(service=self.service, options=self.options)
        try:
            driver.set_page_load_timeout(30)
            driver.get(url)
            time.sleep(3)
            
            soup = BeautifulSoup(driver.page_source, "html.parser")
            posts = soup.find_all("p", class_="title")[:limit]
            
            scored_posts = []
            for post in posts:
                title_elem = post.find("a", class_="title")
                if not title_elem:
                    continue
                
                title = title_elem.text
                full_text = title[:512] 
                
                if len(full_text.strip()) < 10:
                    continue
                    
                result = self.nlp(full_text)[0]
                scored_posts.append({
                    "title": title,
                    "sentiment": result['label'],
                    "confidence": round(result['score'], 3),
                    "url": f"https://reddit.com{title_elem.get('href', '')}"
                })
                
            return scored_posts
            
        except Exception as e:
            logging.error(f"Failed to scrape Old Reddit via Selenium: {e}")
            return []
        finally:
            driver.quit()

if __name__ == "__main__":
    print("=============================================")
    print(" HEDGE FUND SECRET: SOCIAL MEDIA SENTIMENT")
    print("=============================================")
    scanner = SocialMediaScanner()
    
    for sub in ["IndianStreetBets", "DalalStreetTalks"]:
        print(f"\n--- Scanning r/{sub} ---")
        results = scanner.scan_reddit(subreddit=sub, limit=5)
        
        for r in results:
            if r['sentiment'] == 'positive':
                color = "BULLISH"
            elif r['sentiment'] == 'negative':
                color = "BEARISH"
            else:
                color = "NEUTRAL"
                
            print(f"{color} ({r['confidence'] * 100:.1f}%) | {r['title']}")
