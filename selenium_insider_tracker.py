import time
import logging
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [SELENIUM TRACKER] %(message)s')

def fetch_insider_trades_humanely():
    """
    Uses a headless Chrome browser to act exactly like a human user.
    This legally bypasses the basic NSE bot protection by properly 
    rendering Javascript and cookies.
    """
    logging.info("Initializing Headless Chrome Browser (Human Simulation)...")
    
    options = Options()
    options.add_argument('--headless') # Run invisibly in the background
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    # Use a realistic User-Agent so we don't look like a bot
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    
    # Automatically download and manage the ChromeDriver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    
    try:
        url = "https://www.nseindia.com/companies-listing/corporate-filings-insider-trading"
        logging.info(f"Navigating to NSE Insider Trading Portal: {url}")
        
        # 1. Load the page just like a human typing in the URL
        driver.get(url)
        
        # 2. Wait for the heavy Javascript table to actually load (Max wait 15 seconds)
        logging.info("Waiting for Javascript security checks and data tables to render...")
        time.sleep(5) # Hard sleep for initial load
        
        # Look for the data table (The specific ID NSE uses for PIT disclosures)
        try:
            table = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "CFinsidertradingTable"))
            )
            logging.info("SUCCESS! Security bypassed. Table rendered.")
            
            # Extract the raw HTML of the table to parse with BeautifulSoup/Pandas later
            html_content = table.get_attribute('outerHTML')
            logging.info(f"Extracted {len(html_content)} bytes of Insider Trading data!")
            
            # Note: In the live system, we pass this html_content to pandas.read_html()
            # to extract the Exact Promoters and Share volumes.
            
        except Exception as e:
            logging.warning("Table did not load in time. NSE might be under maintenance or updating.")
            
    except Exception as e:
        logging.error(f"Browser encountered an error: {e}")
        
    finally:
        logging.info("Closing Headless Browser to free up RAM.")
        driver.quit()

if __name__ == "__main__":
    fetch_insider_trades_humanely()
