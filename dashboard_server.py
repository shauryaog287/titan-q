import os
import json
from http.server import HTTPServer, BaseHTTPRequestHandler

DASHBOARD_HTML = r"C:\Users\jiten\titan-q-dashboard\index.html"
STATE_FILE = r"D:\Quant_Project\dashboard_state.json"

class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Enable CORS
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        
        if self.path == '/':
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            if os.path.exists(DASHBOARD_HTML):
                with open(DASHBOARD_HTML, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.wfile.write(b"Dashboard HTML not found.")
                
        elif self.path == '/api/live_state':
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                default_data = json.dumps({
                    "current_pnl": 500000.0,
                    "toxicity_score": 0.0,
                    "quantum_assets": [],
                    "recent_trades": []
                })
                self.wfile.write(default_data.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass # Suppress logs to keep command prompt clean

if __name__ == "__main__":
    print("==================================================")
    print("TITAN-Q DASHBOARD SERVER RUNNING")
    print("Open your browser to: http://localhost:5000")
    print("==================================================")
    server = HTTPServer(('0.0.0.0', 5000), DashboardHandler)
    server.serve_forever()
