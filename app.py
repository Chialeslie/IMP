from flask import Flask, Response, render_template_string, request, jsonify
import requests
import time
import serial
import threading
from datetime import datetime

app = Flask(__name__)

# Persistent path for your Arduino Giga
SERIAL_PORT = "/dev/ttyACM0"

# Global variables
current_lat = 1.3521
current_lon = 103.8198
current_heading = 0
current_roll = 0.0
current_pitch = 0.0
robot_mode = "AUTONOMOUS"
arduino_connected = False
last_gps_time = "Never"

# Initialize Serial connection using the stable path
try:
    arduino = serial.Serial(SERIAL_PORT, 115200, timeout=1)
    time.sleep(2)  # Allow connection to settle
    arduino_connected = True
    print("Connected to Arduino Giga via Serial")
except Exception as e:
    print(f"Warning: Could not connect to Arduino: {e}")
    arduino = None
    arduino_connected = False

def send_to_arduino(command_string):
    if arduino and arduino.is_open:
        try:
            arduino.write((command_string + '\n').encode('utf-8'))
        except Exception as e:
            print(f"Serial write error: {e}")

# Background thread to continuously read incoming data from Arduino (GPS + Heading + IMU strings)
def serial_reader_thread():
    global current_lat, current_lon, current_heading, current_roll, current_pitch, last_gps_time, arduino_connected
    while True:
        if arduino and arduino.is_open:
            try:
                arduino_connected = True
                if arduino.in_waiting > 0:
                    line = arduino.readline().decode('utf-8', errors='ignore').strip()
                    if line.startswith("G:"):
                        parts = line[2:].split(',')
                        if len(parts) >= 2:
                            current_lat = float(parts[0])
                            current_lon = float(parts[1])
                            if len(parts) >= 3:
                                current_heading = int(float(parts[2]))
                            if len(parts) >= 4:
                                current_roll = float(parts[3])
                            if len(parts) >= 5:
                                current_pitch = float(parts[4])
                            last_gps_time = datetime.now().strftime("%H:%M:%S")
            except Exception as e:
                print(f"Serial read error: {e}")
                arduino_connected = False
        else:
            arduino_connected = False
        time.sleep(0.05)

# Start background serial listener thread
reader_thread = threading.Thread(target=serial_reader_thread, daemon=True)
reader_thread.start()

ESP32_CAMS = {
    "cam1": "http://192.168.10.103:81/stream",
    "cam2": "http://192.168.10.200:81/stream",
    "cam3": "http://192.168.10.155:81/stream",
    "cam4": "http://192.168.10.121:81/stream"
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Robot Control Station Dashboard</title>
    <!-- Include NippleJS for the virtual joystick -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/nipplejs/0.10.1/nipplejs.min.js"></script>
    <!-- Include Leaflet CSS and JS for dynamic map updates -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: #121212; 
            color: #e0e0e0; 
            margin: 0; 
            padding: 0;
            display: flex;
            flex-direction: column;
            height: 100vh;
            overflow: hidden;
        }
        
        /* Top Master Header Panel */
        #top-header-panel {
            width: 100%;
            background: #181818;
            border-bottom: 2px solid #333;
            padding: 12px 20px;
            box-sizing: border-box;
            box-shadow: 0 2px 8px rgba(0,0,0,0.4);
            z-index: 1100;
        }

        #top-header-panel h1 {
            color: #ffffff;
            margin: 0;
            font-size: 22px;
            text-align: center;
            letter-spacing: 0.5px;
        }

        /* Master Dashboard Container Layout */
        #dashboard-container {
            display: flex;
            width: 100%;
            height: calc(100vh - 55px);
            box-sizing: border-box;
            gap: 15px;
            padding: 15px;
        }

        /* LEFT SIDE: Scrollable Area (Telemetry, Map, Cameras) */
        #left-scrollable-area {
            flex: 1;
            height: 100%;
            overflow-y: auto;
            padding-right: 5px;
        }

        /* RIGHT SIDE: Fixed Control Panel Sidebar */
        #right-fixed-sidebar {
            width: 320px;
            min-width: 320px;
            height: 100%;
            background: #181818;
            border-radius: 8px;
            display: flex;
            flex-direction: column;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5);
            z-index: 1000;
            overflow-y: auto;
        }

        .card { background: #1e1e1e; padding: 12px 15px; margin-bottom: 12px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.5); text-align: left; }
        .card h3 { margin: 0 0 8px 0; border-bottom: 2px solid #333; padding-bottom: 4px; color: #2196F3; font-size: 15px; }

        /* Status Grid */
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
            gap: 10px;
            text-align: center;
        }
        .status-item {
            background: #2a2a2a;
            padding: 6px;
            border-radius: 6px;
            font-size: 13px;
        }
        .status-item p { margin: 0 0 2px 0; color: #aaa; }
        .status-item span { font-weight: bold; }

        .map-container { width: 100%; height: 280px; border-radius: 6px; overflow: hidden; }
        
        .camera-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 10px;
            margin-top: 5px;
        }
        .cam-box {
            background: #2a2a2a;
            padding: 8px;
            border-radius: 6px;
            text-align: center;
        }
        .cam-box p { margin: 0 0 5px 0; font-size: 13px; }
        .cam-box img {
            width: 100%;
            height: auto;
            max-height: 130px;
            object-fit: cover;
            border-radius: 4px;
            background: #000;
        }
        
        /* Mode Switcher Buttons */
        .mode-container {
            margin-bottom: 12px;
            text-align: center;
        }
        .mode-btn {
            padding: 6px 12px;
            font-size: 12px;
            font-weight: bold;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            margin: 0 4px;
            color: white;
            transition: all 0.2s;
        }
        #btn-auto { background: #d32f2f; }
        #btn-auto.active { background: #f44336; box-shadow: 0 0 8px #f44336; }
        #btn-manual { background: #455a64; }
        #btn-manual.active { background: #2196F3; box-shadow: 0 0 8px #2196F3; }

        /* Control Panel Layout inside Right Sidebar */
        .control-panel {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 50px;
            opacity: 0.3;
            pointer-events: none;
            transition: opacity 0.3s;
        }
        .control-panel.enabled {
            opacity: 1;
            pointer-events: auto;
        }
        
        .control-group {
            text-align: center;
            width: 100%;
        }
        .control-group p {
            margin: 0 0 5px 0;
            font-size: 12px;
            color: #aaa;
            font-weight: 600;
        }

        #joystick-container {
            width: 110px;
            height: 110px;
            position: relative;
            background: #2a2a2a;
            border-radius: 50%;
            margin: 0 auto;
        }
        
        /* D-Pad Button Grid Layout */
        .dpad-container {
            display: grid;
            grid-template-columns: repeat(3, 55px);
            grid-template-rows: repeat(3, 38px);
            gap: 5px;
            justify-content: center;
            align-items: center;
        }
        .ctrl-btn {
            background: #2196F3;
            color: white;
            border: none;
            font-size: 13px;
            font-weight: bold;
            border-radius: 6px;
            cursor: pointer;
            height: 100%;
            transition: background 0.2s;
        }
        .ctrl-btn:active { background: #0b7dda; }
        
        .btn-up { grid-column: 2; grid-row: 1; }
        .btn-left { grid-column: 1; grid-row: 2; }
        .btn-stop { grid-column: 2; grid-row: 2; background: #f44336; font-size: 10px; }
        .btn-stop:active { background: #d32f2f; }
        .btn-right { grid-column: 3; grid-row: 2; }
        .btn-down { grid-column: 2; grid-row: 3; }

        #command-status { font-style: italic; color: #aaa; margin: 15px 0 0 0; font-size: 11px; text-align: center; }
    </style>
</head>
<body>

    <!-- TOP TITLE PANEL -->
    <div id="top-header-panel">
        <h1>Robot Control System Dashboard</h1>
    </div>

    <div id="dashboard-container">
        <!-- LEFT SIDE: Scrollable Monitoring panels -->
        <div id="left-scrollable-area">
            <div class="card">
                <h3>System Status & Telemetry</h3>
                <div class="status-grid">
                    <div class="status-item">
                        <p>Network Link</p>
                        <span style="color: #4CAF50;">Online</span>
                    </div>
                    <div class="status-item">
                        <p>Arduino USB Link</p>
                        <span id="arduino-status" style="color: #ff9800;">Checking...</span>
                    </div>
                    <div class="status-item">
                        <p>Operating Mode</p>
                        <span id="mode-display" style="color: #f44336;">AUTONOMOUS</span>
                    </div>
                    <div class="status-item">
                        <p>Compass Heading</p>
                        <span id="heading-display" style="color: #E91E63;">0°</span>
                    </div>
                    <div class="status-item">
                        <p>IMU Roll</p>
                        <span id="roll-display" style="color: #00bcd4;">0.0°</span>
                    </div>
                    <div class="status-item">
                        <p>IMU Pitch</p>
                        <span id="pitch-display" style="color: #00bcd4;">0.0°</span>
                    </div>
                    <div class="status-item">
                        <p>Last GPS Update</p>
                        <span id="gps-time" style="color: #2196F3;">Never</span>
                    </div>
                </div>
            </div>

            <div class="card">
                <h3>Live GPS Tracking</h3>
                <div id="map" class="map-container"></div>
            </div>

            <div class="card">
                <h3>ESP32-CAM 360° Feeds</h3>
                <div class="camera-grid">
                    <div class="cam-box">
                        <p>Camera 1 (Front)</p>
                        <img src="/video_feed/cam1" alt="Cam 1 Offline">
                    </div>
                    <div class="cam-box">
                        <p>Camera 2 (Back)</p>
                        <img src="/video_feed/cam2" alt="Cam 2 Offline">
                    </div>
                    <div class="cam-box">
                        <p>Camera 3 (Left)</p>
                        <img src="/video_feed/cam3" alt="Cam 3 Offline">
                    </div>
                    <div class="cam-box">
                        <p>Camera 4 (Right)</p>
                        <img src="/video_feed/cam4" alt="Cam 4 Offline">
                    </div>
                </div>
            </div>
        </div>

        <!-- RIGHT SIDE: Fixed Robot Controls Sidebar -->
        <div id="right-fixed-sidebar">
            <div class="card" style="flex: 1; margin-bottom: 0; display: flex; flex-direction: column; justify-content: space-between; background: transparent; box-shadow: none;">
                <div>
                    <h3 style="border-bottom: none; text-align: center; font-size: 16px; margin-bottom: 12px;">Robot Controls</h3>
                    <div class="mode-container">
                        <button id="btn-auto" class="mode-btn active" onclick="setMode('AUTONOMOUS')">Auto</button>
                        <button id="btn-manual" class="mode-btn" onclick="setMode('MANUAL')">Manual</button>
                    </div>
                    
                    <div id="control-panel" class="control-panel">
                        <div class="control-group">
                            <p>Virtual Joystick</p>
                            <div id="joystick-container"></div>
                        </div>
                        <div class="control-group">
                            <p>Movement Buttons</p>
                            <div class="dpad-container">
                                <button class="ctrl-btn btn-up" onclick="sendAction('forward')">▲</button>
                                <button class="ctrl-btn btn-left" onclick="sendAction('left')">◄</button>
                                <button class="ctrl-btn btn-stop" onclick="sendAction('stop')">STOP</button>
                                <button class="ctrl-btn btn-right" onclick="sendAction('right')">►</button>
                                <button class="ctrl-btn btn-down" onclick="sendAction('backward')">▼</button>
                            </div>
                        </div>
                    </div>
                </div>
                <p id="command-status">Last command: None</p>
            </div>
        </div>
    </div>

    <script>
        // Initialize Leaflet Map dynamically
        var map = L.map('map').setView([{{ current_lat }}, {{ current_lon }}], 15);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            maxZoom: 19,
            attribution: '© OpenStreetMap'
        }).addTo(map);

        var robotMarker = L.marker([{{ current_lat }}, {{ current_lon }}]).addTo(map)
            .bindPopup('Robot Current Location');

        // Initialize NippleJS Virtual Joystick
        var joystick = nipplejs.create({
            zone: document.getElementById('joystick-container'),
            mode: 'static',
            position: { left: '50%', top: '50%' },
            color: 'green',
            size: 95
        });

        let lastSend = 0;
        joystick.on('move', function (evt, data) {
            let now = Date.now();
            if (now - lastSend > 100) {
                let angle = data.angle.degree;
                let force = data.force;
                sendDriveCommand(angle, force);
                lastSend = now;
            }
        });

        joystick.on('end', function (evt, data) {
            sendDriveCommand(0, 0);
        });

        // Periodic Status Fetcher to keep UI and Map Marker updated
        setInterval(() => {
            fetch('/status')
            .then(res => res.json())
            .then(data => {
                let arduinoEl = document.getElementById('arduino-status');
                if (data.arduino_connected) {
                    arduinoEl.innerText = "Connected";
                    arduinoEl.style.color = "#4CAF50";
                } else {
                    arduinoEl.innerText = "Disconnected";
                    arduinoEl.style.color = "#f44336";
                }
                document.getElementById('heading-display').innerText = data.heading + "°";
                document.getElementById('roll-display').innerText = data.roll + "°";
                document.getElementById('pitch-display').innerText = data.pitch + "°";
                document.getElementById('gps-time').innerText = data.last_gps_time;

                // Update Map Marker coordinates dynamically
                let newLatLng = [data.lat, data.lon];
                robotMarker.setLatLng(newLatLng);
                // Optionally pan map to robot location:
                // map.panTo(newLatLng);
            });
        }, 2000);

        function setMode(mode) {
            fetch('/control/mode', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode: mode })
            })
            .then(response => response.json())
            .then(data => {
                let display = document.getElementById('mode-display');
                let panel = document.getElementById('control-panel');
                let btnAuto = document.getElementById('btn-auto');
                let btnManual = document.getElementById('btn-manual');

                if (display) {
                    display.innerText = data.mode;
                }
                
                if (data.mode === 'MANUAL') {
                    if (display) display.style.color = '#2196F3';
                    panel.classList.add('enabled');
                    btnManual.classList.add('active');
                    btnAuto.classList.remove('active');
                } else {
                    if (display) display.style.color = '#f44336';
                    panel.classList.remove('enabled');
                    btnAuto.classList.add('active');
                    btnManual.classList.remove('active');
                }
                document.getElementById('command-status').innerText = "Mode switched to: " + data.mode;
            });
        }

        function sendDriveCommand(angle, force) {
            fetch('/control/drive', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ angle: angle, force: force })
            })
            .then(response => response.json());
        }

        function sendAction(actionType) {
            fetch('/control/action', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: actionType })
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('command-status').innerText = "Action -> " + actionType;
            });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, current_lat=current_lat, current_lon=current_lon)

@app.route('/status')
def system_status():
    return jsonify({
        "arduino_connected": arduino_connected,
        "mode": robot_mode,
        "last_gps_time": last_gps_time,
        "lat": current_lat,
        "lon": current_lon,
        "heading": current_heading,
        "roll": current_roll,
        "pitch": current_pitch
    })

@app.route('/control/mode', methods=['POST'])
def control_mode():
    global robot_mode
    data = request.json
    robot_mode = data.get('mode', 'AUTONOMOUS')
    
    cmd = f"MODE:{robot_mode}"
    send_to_arduino(cmd)
    print(f"System Mode Changed -> {robot_mode}")
    
    return jsonify({"status": "success", "mode": robot_mode})

@app.route('/control/drive', methods=['POST'])
def control_drive():
    if robot_mode != "MANUAL":
        return jsonify({"status": "ignored", "reason": "Robot is in Autonomous mode"})
    
    data = request.json
    angle = round(data.get('angle', 0))
    force = round(data.get('force', 0) * 100)
    
    cmd = f"D:{angle},{force}"
    send_to_arduino(cmd)
    return jsonify({"status": "success", "angle": angle, "force": force})

@app.route('/control/action', methods=['POST'])
def control_action():
    if robot_mode != "MANUAL" and request.json.get('action') != 'stop':
        return jsonify({"status": "ignored", "reason": "Robot is in Autonomous mode"})
    
    data = request.json
    action = data.get('action')
    
    cmd = f"A:{action}"
    send_to_arduino(cmd)
    return jsonify({"status": "success", "action": action})

@app.route('/video_feed/<cam_id>')
def video_feed(cam_id):
    if cam_id not in ESP32_CAMS:
        return "Camera not found", 404
    
    target_url = ESP32_CAMS[cam_id]
    
    def generate():
        while True:
            try:
                req = requests.get(target_url, stream=True, timeout=5)
                for chunk in req.iter_content(chunk_size=1024):
                    if chunk:
                        yield chunk
            except Exception:
                time.sleep(0.5)
                continue

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=1234567890')
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)