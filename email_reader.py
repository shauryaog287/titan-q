import imaplib
import email
import logging
from bs4 import BeautifulSoup
from transformers import pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class PrivateIntelligenceReader:
    """
    Phase 2: The Ear (Email Intelligence Module)
    Securely connects to a dedicated Hedge Fund IMAP inbox to read
    private newsletters, tip-offs, and PR wire alerts in real-time.
    """
    def __init__(self, email_address: str, app_password: str):
        self.email_address = email_address
        self.app_password = app_password
        self.imap_server = "imap.gmail.com"
        
        logging.info("Loading FinBERT for Private Email Sentiment...")
        self.nlp = pipeline("text-classification", model="ProsusAI/finbert")
        
    def scan_inbox(self, max_emails=5):
        logging.info("Connecting to secure IMAP server...")
        try:
            # Connect to IMAP
            mail = imaplib.IMAP4_SSL(self.imap_server)
            mail.login(self.email_address, self.app_password)
            mail.select('inbox')
            
            # Search for all unread emails
            status, messages = mail.search(None, 'UNSEEN')
            email_ids = messages[0].split()
            
            logging.info(f"Found {len(email_ids)} unread intelligence alerts.")
            
            alerts = []
            for e_id in email_ids[-max_emails:]:
                _, msg_data = mail.fetch(e_id, '(RFC822)')
                for response_part in msg_data:
                    if isinstance(response_part, tuple):
                        msg = email.message_from_bytes(response_part[1])
                        subject = msg['subject']
                        sender = msg['from']
                        
                        body = ""
                        if msg.is_multipart():
                            for part in msg.walk():
                                if part.get_content_type() == "text/plain":
                                    body = part.get_payload(decode=True).decode()
                                    break
                        else:
                            body = msg.get_payload(decode=True).decode()
                            
                        # Clean up HTML if any
                        clean_body = BeautifulSoup(body, "html.parser").text.strip()
                        
                        # Grade the intelligence
                        analysis_text = f"{subject}. {clean_body}"[:512]
                        sentiment = self.nlp(analysis_text)[0]
                        
                        alerts.append({
                            "sender": sender,
                            "subject": subject,
                            "sentiment": sentiment['label'],
                            "confidence": sentiment['score']
                        })
            
            mail.logout()
            return alerts
            
        except Exception as e:
            logging.error(f"IMAP connection failed. Check App Passwords: {e}")
            return []

if __name__ == "__main__":
    print("=============================================")
    print(" HEDGE FUND SECRET: PRIVATE INTELLIGENCE IMAP")
    print("=============================================")
    
    # Example usage (Replace with actual App Password)
    # reader = PrivateIntelligenceReader("your_fund_email@gmail.com", "xxxx_xxxx_xxxx_xxxx")
    # alerts = reader.scan_inbox()
    # for alert in alerts:
    #     print(f"[{alert['sentiment'].upper()}] {alert['subject']} (from {alert['sender']})")
    
    print("Email reader module initialized. Waiting for IMAP credentials.")
