import os
import json
from http.server import HTTPServer, BaseHTTPRequestHandler

DASHBOARD_HTML = r"C:\Users\jiten\titan-q-dashboard\index.html"
STATE_FILE = r"D:\Quant_Project\dashboard_state.json"

# Localhost only. Binding 0.0.0.0 exposed the account state to the whole LAN.
HOST, PORT = "127.0.0.1", 5000

DEFAULT_STATE = {
    "current_pnl": 0.0,
    "toxicity_score": 0.0,
    "quantum_assets": [],
    "recent_trades": []
}


class ExclusiveHTTPServer(HTTPServer):
    # HTTPServer sets SO_REUSEADDR, which on Windows lets a second process bind a
    # port that is already in use. Turn it off so a second server fails to start.
    allow_reuse_address = False


class DashboardHandler(BaseHTTPRequestHandler):
    def _send(self, code, content_type, body):
        self.send_response(code)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-type', content_type)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == '/':
            if os.path.exists(DASHBOARD_HTML):
                with open(DASHBOARD_HTML, 'rb') as f:
                    self._send(200, 'text/html', f.read())
            else:
                self._send(404, 'text/plain', b"Dashboard HTML not found.")

        elif self.path == '/api/live_state':
            body = None
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, 'rb') as f:
                    body = f.read()
            if not body:
                body = json.dumps(DEFAULT_STATE).encode('utf-8')
            self._send(200, 'application/json', body)

        else:
            self._send(404, 'text/plain', b"Not found.")

    def log_message(self, format, *args):
        pass # Suppress logs to keep command prompt clean


if __name__ == "__main__":
    try:
        server = ExclusiveHTTPServer((HOST, PORT), DashboardHandler)
    except OSError:
        # The scheduled task starts a server every day; if yesterday's is still
        # running it already serves the same page and the same state file.
        print("Dashboard already running at http://localhost:%d -- nothing to do." % PORT)
        raise SystemExit(0)
    print("==================================================")
    print("TITAN-Q DASHBOARD SERVER RUNNING")
    print("Open your browser to: http://localhost:%d" % PORT)
    print("==================================================")
    server.serve_forever()
