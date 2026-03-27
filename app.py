from flask import Flask, jsonify, render_template_string, send_file
import pywifi
import time
import subprocess
import json
from datetime import datetime
from tensorflow import keras
import numpy as np

model = keras.models.load_model("models/rssi_model.h5")
direction_model = keras.models.load_model("models/direction_model.h5")

app = Flask(__name__)
tracking_data = {}
last_scan_time = "Never"

# ---------------- DNN SIGNAL ----------------
def predict_signal(rssi):
    rssi_array = np.array([[rssi]])
    prediction = model.predict(rssi_array, verbose=0)
    class_id = np.argmax(prediction)
    if class_id == 0: return "Strong 🔥"
    elif class_id == 1: return "Medium ⚠️"
    else: return "Weak ❌"

# ---------------- AI DIRECTION ----------------
def predict_direction_ai(rssi_list):
    if not rssi_list: return "Out of range"
    current = rssi_list[-1]
    if current < -95: return "Out of range"
    if current > -50: return "Near"
    if current < -80: return "Far"
    if len(rssi_list) >= 2:
        prev = rssi_list[-2]
        return "Getting closer" if current > prev else "Moving away"
    return "Stable"

# ---------------- IMPROVED WIFI SCAN (Fixed Channel & Band) ----------------
def scan_wifi():
    global last_scan_time
    netsh_output = subprocess.check_output("netsh wlan show networks mode=bssid", shell=True).decode(errors="ignore")

    # Maps to store data from netsh
    bssid_map = {}
    band_map = {}
    channel_map = {}
    current_ssid = None

    for line in netsh_output.split("\n"):
        line = line.strip()
        if line.startswith("SSID"):
            current_ssid = line.split(":", 1)[1].strip()
        elif current_ssid:
            if "BSSID" in line:
                bssid_map[current_ssid] = line.split(":", 1)[1].strip()
            elif "Band" in line:
                band_map[current_ssid] = line.split(":", 1)[1].strip()
            elif "Channel" in line:
                channel_map[current_ssid] = line.split(":", 1)[1].strip()

    # pywifi for RSSI
    wifi = pywifi.PyWiFi()
    iface = wifi.interfaces()[0]
    iface.scan()
    time.sleep(2.5)
    results = iface.scan_results()

    network_list = []
    seen = set()

    for net in results:
        ssid = net.ssid if net.ssid else "Hidden"
        if ssid in seen: continue
        seen.add(ssid)

        rssi = net.signal
        bssid = net.bssid if hasattr(net, 'bssid') and net.bssid else bssid_map.get(ssid, "Unknown")
        band = band_map.get(ssid, "Unknown")
        channel = channel_map.get(ssid, "?")

        distance = round(10 ** ((-40 - rssi) / 30), 2)
        security = "WPA2-Personal"
        strength = "Strong" if rssi >= -55 else "Medium" if rssi >= -70 else "Weak"

        network_list.append({
            "SSID": ssid,
            "BSSID": bssid,
            "RSSI": rssi,
            "Distance": distance,
            "Security": security,
            "Band": band,
            "Channel": channel,
            "Strength": strength
        })

    last_scan_time = datetime.now().strftime("%H:%M:%S")
    return sorted(network_list, key=lambda x: x["RSSI"], reverse=True)

# ---------------- TRACK STRONG WIFI ----------------
def track_strong_wifi(networks):
    global tracking_data
    strong_list = []
    for net in networks:
        if net["RSSI"] > -78:
            ssid = net["SSID"]
            rssi = net["RSSI"]
            if ssid not in tracking_data:
                tracking_data[ssid] = []
            tracking_data[ssid].append(rssi)
            if len(tracking_data[ssid]) > 100:
                tracking_data[ssid] = tracking_data[ssid][-100:]
            direction = predict_direction_ai(tracking_data[ssid])
            strong_list.append({
                "SSID": ssid,
                "BSSID": net["BSSID"],
                "RSSI": rssi,
                "Distance": net["Distance"],
                "Direction": direction,
                "DNN": predict_signal(rssi)
            })
    return strong_list

@app.route('/data')
def data():
    networks = scan_wifi()
    strong = track_strong_wifi(networks)
    return jsonify({
        "all": networks,
        "strong": strong,
        "last_scan": last_scan_time
    })

@app.route('/reset')
def reset_tracking():
    global tracking_data
    tracking_data = {}
    return jsonify({"status": "ok"})

@app.route('/export')
def export_data():
    networks = scan_wifi()
    strong = track_strong_wifi(networks)
    export = {"timestamp": last_scan_time, "all": networks, "strong": strong}
    with open("rf_scan_export.json", "w") as f:
        json.dump(export, f, indent=2)
    return send_file("rf_scan_export.json", as_attachment=True)

# ---------------- UI ----------------
@app.route('/')
def home():
    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RF Scanner</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700&display=swap');
        body { background: #0a001f; color: #00f0ff; font-family: 'Orbitron', sans-serif; margin: 0; padding: 20px; }
        .container { max-width: 1480px; margin: auto; }
        h1 { text-align: center; font-size: 2.6rem; margin-bottom: 25px; text-shadow: 0 0 15px #00f0ff; }
        .card { background: rgba(20, 10, 50, 0.9); border: 2px solid #00f0ff; border-radius: 12px; padding: 20px; margin-bottom: 25px; box-shadow: 0 0 25px rgba(0, 240, 255, 0.4); }
        .strong-card { border-color: #c300ff; }
        button { background: transparent; color: #39ff14; border: 2px solid #39ff14; padding: 12px 30px; margin: 8px; font-size: 1.05rem; border-radius: 8px; cursor: pointer; transition: all 0.3s; }
        button:hover { background: #39ff14; color: #000; box-shadow: 0 0 20px #39ff14; }
        .status { text-align: center; font-size: 1.2rem; margin: 12px 0; }
        .timestamp { text-align: center; color: #aaa; font-size: 1rem; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th, td { padding: 14px 10px; text-align: left; border-bottom: 1px solid rgba(0, 240, 255, 0.3); font-size: 1.02rem; }
        th { background: rgba(0, 240, 255, 0.2); color: #00f0ff; }
        tr:hover { background: rgba(195, 0, 255, 0.15); }
        .graph-container { height: 380px; background: #0f0529; border-radius: 12px; padding: 15px; border: 1px solid #00f0ff; }
        .section-title { color: #c300ff; text-align: center; margin: 15px 0 10px; font-size: 1.4rem; }
        .no-data { text-align: center; color: #ff00aa; padding: 25px; font-style: italic; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔷 RF Scanner</h1>
        
        <div class="card">
            <div style="text-align:center;">
                <button onclick="loadData()">SCAN NOW</button>
                <button onclick="resetTracking()">RESET TRACKING</button>
                <button onclick="exportData()">EXPORT JSON</button>
            </div>
            <div class="status" id="status">Ready</div>
            <div class="timestamp" id="timestamp">Last scan: Never</div>
        </div>

        <div class="card">
            <div class="section-title">Available Networks</div>
            <table id="wifiTable"></table>
        </div>

        <div class="card strong-card">
            <div class="section-title">Strong Signal Tracking</div>
            <table id="strongTable"></table>
        </div>

        <div class="card">
            <div class="section-title">RSSI Live Graph</div>
            <div class="graph-container">
                <canvas id="rssiChart"></canvas>
            </div>
        </div>
    </div>

    <script>
        let chart;
        let colorIndex = 0;
        const colors = ['#00f0ff', '#ff00ff', '#39ff14', '#ffff00', '#ff8800'];

        function initChart() {
            chart = new Chart(document.getElementById('rssiChart'), {
                type: 'line',
                data: { labels: [], datasets: [] },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { labels: { color: '#00f0ff' } } },
                    scales: {
                        y: { min: -100, max: -30, grid: { color: 'rgba(0,240,255,0.2)' }, ticks: { color: '#00f0ff', callback: v => v + ' dBm' }},
                        x: { grid: { color: 'rgba(0,240,255,0.15)' }, ticks: { color: '#00f0ff' }}
                    }
                }
            });
        }

        async function loadData() {
            const status = document.getElementById("status");
            status.textContent = "Scanning...";
            status.style.color = "#ff00aa";

            try {
                const res = await fetch('/data');
                const data = await res.json();
                document.getElementById("timestamp").textContent = `Last scan: ${data.last_scan}`;

                // Available Networks Table
                let html = `<tr>
                    <th>SSID</th>
                    <th>Router MAC (BSSID)</th>
                    <th>RSSI</th>
                    <th>Distance</th>
                    <th>Security</th>
                    <th>Band</th>
                    <th>Channel</th>
                    <th>Strength</th>
                </tr>`;
                
                data.all.forEach(n => {
                    const rowStyle = n.Strength === "Strong" ? "background:rgba(57,255,20,0.15);" : 
                                    n.Strength === "Medium" ? "background:rgba(255,255,0,0.1);" : "";
                    html += `<tr style="${rowStyle}">
                        <td>${n.SSID}</td>
                        <td style="font-family:monospace; font-size:0.95rem;">${n.BSSID}</td>
                        <td>${n.RSSI}</td>
                        <td>${n.Distance} m</td>
                        <td>${n.Security}</td>
                        <td>${n.Band}</td>
                        <td>${n.Channel}</td>
                        <td>${n.Strength}</td>
                    </tr>`;
                });
                document.getElementById("wifiTable").innerHTML = html;

                // Strong Signal Tracking
                let strongHTML = `<tr>
                    <th>SSID</th>
                    <th>Router MAC (BSSID)</th>
                    <th>RSSI</th>
                    <th>Distance</th>
                    <th>Direction</th>
                    <th>DNN</th>
                </tr>`;
                let hasStrong = false;

                data.strong.forEach(n => {
                    hasStrong = true;
                    strongHTML += `<tr>
                        <td>${n.SSID}</td>
                        <td style="font-family:monospace;">${n.BSSID}</td>
                        <td>${n.RSSI}</td>
                        <td>${n.Distance} m</td>
                        <td>${n.Direction}</td>
                        <td>${n.DNN}</td>
                    </tr>`;

                    let dataset = chart.data.datasets.find(ds => ds.label === n.SSID);
                    const color = colors[colorIndex % colors.length];
                    colorIndex++;
                    if (!dataset) {
                        dataset = { label: n.SSID, data: [], borderColor: color, backgroundColor: color+'22', borderWidth: 3.5, tension: 0.4, pointRadius: 5 };
                        chart.data.datasets.push(dataset);
                    }
                    dataset.data.push(n.RSSI);
                    if (dataset.data.length > 25) dataset.data.shift();
                });

                if (!hasStrong) {
                    strongHTML += `<tr><td colspan="6" class="no-data">No strong signals detected</td></tr>`;
                }
                document.getElementById("strongTable").innerHTML = strongHTML;

                // Graph update
                if (chart.data.labels.length === 0 || chart.data.labels.length >= 25) {
                    chart.data.labels = Array.from({length: 25}, (_, i) => i+1);
                } else {
                    chart.data.labels.push(chart.data.labels.length + 1);
                    if (chart.data.labels.length > 25) chart.data.labels.shift();
                }
                chart.update();

                status.textContent = "Scan Complete";
                status.style.color = "#39ff14";

            } catch (e) {
                console.error(e);
                status.textContent = "Scan Failed";
                status.style.color = "#ff0055";
            }
        }

        async function resetTracking() {
            if (confirm("Clear tracking history?")) {
                await fetch('/reset');
                chart.data.datasets = [];
                chart.update();
                loadData();
            }
        }

        function exportData() {
            window.location.href = "/export";
        }

        window.onload = () => {
            initChart();
            loadData();
            setInterval(loadData, 60000);
        };
    </script>
</body>
</html>
""")

if __name__ == '__main__':
    print("🚀 RF Scanner Running - Channel & Band Fixed")
    app.run(debug=True)