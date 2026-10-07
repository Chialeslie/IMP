from flask import Flask, render_template_string

app = Flask(__name__)

# Simple HTML template for your ground station dashboard
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Robot Ground Station</title>
    <style>
        body { font-family: Arial, sans-serif; background: #121212; color: #e0e0e0; text-align: center; padding-top: 50px; }
        .card { background: #1e1e1e; padding: 20px; margin: 20px auto; width: 60%; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.3); }
    </style>
</head>
<body>
    <h1>Robot Ground Station Dashboard</h1>
    <div class="card">
        <h3>System Status</h3>
        <p>Status: <span style="color: #4CAF50;">Online & Connected via Tailscale</span></p>
    </div>
    <div class="card">
        <h3>Telemetry & Video Feeds</h3>
        <p>Ready to integrate ESP32-CAM streams and GIGA GPS data.</p>
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

if __name__ == '__main__':
    # host='0.0.0.0' allows external access over Tailscale
    app.run(host='0.0.0.0', port=5000, debug=True)