import serial
import serial.tools.list_ports
import tkinter as tk
from tkinter import Canvas
import threading
import time

# === KONFIGURASI SERIAL ===
PORT = 'COM5'  # Sesuaikan dengan port ESP32 Anda
BAUDRATE = 115200
ser = None

# === GUI SETUP ===
root = tk.Tk()
root.title("HMI Lalu Lintas 4 Titik - MQTT Version (10ms Update)")

traffic_lights = {}
counters = {}
durasi_labels = {}

def create_traffic_light_frame(name, row, col):
    frame = tk.LabelFrame(root, text=f"Jalan {name}", padx=10, pady=10)
    frame.grid(row=row, column=col, padx=15, pady=15)

    canvas = Canvas(frame, width=60, height=180, bg='black')
    canvas.pack()

    light_red = canvas.create_oval(10, 10, 50, 50, fill='gray')
    light_yellow = canvas.create_oval(10, 65, 50, 105, fill='gray')
    light_green = canvas.create_oval(10, 120, 50, 160, fill='gray')

    # ✅ DESAIN ORIGINAL: Putih, Arial 18, tanpa perubahan warna
    counter_label = tk.Label(frame, text="00.00", font=("Arial", 18), fg='white', bg='black')
    counter_label.pack(pady=5)

    traffic_lights[name] = {
        'canvas': canvas,
        'lights': {
            'RED': light_red,
            'YELLOW': light_yellow,
            'GREEN': light_green
        }
    }
    counters[name] = counter_label

# Create traffic light frames (2x2 grid)
for i, titik in enumerate(["A", "B", "C", "D"]):
    create_traffic_light_frame(titik, i // 2, i % 2)

# === PANEL KANAN (Durasi & Status) ===
right_frame = tk.Frame(root)
right_frame.grid(row=0, column=2, rowspan=2, padx=15, pady=15, sticky='n')

# Durasi dari Server
durasi_frame = tk.LabelFrame(right_frame, text="Durasi dari Server", padx=10, pady=10)
durasi_frame.pack(fill='x', pady=5)

for titik in ["A", "B", "C", "D"]:
    label = tk.Label(durasi_frame, text=f"{titik} = -- s", font=("Arial", 14))
    label.pack(anchor='w', pady=5)
    durasi_labels[titik] = label

# Status Frame
status_frame = tk.LabelFrame(right_frame, text="Status Sistem", padx=10, pady=10)
status_frame.pack(fill='x', pady=5)

wifi_status_label = tk.Label(status_frame, text="WiFi: Menunggu...", 
                              font=("Arial", 11), fg='white', bg='gray', 
                              anchor='w', padx=10)
wifi_status_label.pack(fill='x', pady=3)

mqtt_status_label = tk.Label(status_frame, text="MQTT: Menunggu...", 
                              font=("Arial", 11), fg='white', bg='gray', 
                              anchor='w', padx=10)
mqtt_status_label.pack(fill='x', pady=3)

db_status_label = tk.Label(status_frame, text="Database: Menunggu...", 
                            font=("Arial", 11), fg='white', bg='gray', 
                            anchor='w', padx=10)
db_status_label.pack(fill='x', pady=3)

mode_status_label = tk.Label(status_frame, text="Mode: --", 
                              font=("Arial", 11), fg='white', bg='gray', 
                              anchor='w', padx=10)
mode_status_label.pack(fill='x', pady=3)

# Performance Frame
perf_frame = tk.LabelFrame(right_frame, text="Delay RTT", padx=10, pady=10)
perf_frame.pack(fill='x', pady=5)

rtt_label = tk.Label(perf_frame, text="RTT: -- ms", 
                     font=("Arial", 12, "bold"), fg='white', bg='gray', 
                     anchor='w', padx=10)
rtt_label.pack(fill='x', pady=3)

# === FUNGSI RESET ===
def reset_durasi_labels():
    """Reset durasi ke default saat disconnect"""
    for titik in durasi_labels:
        durasi_labels[titik].config(text=f"{titik} = -- s")

# === FUNGSI UPDATE STATUS ===
def update_status_messages(line):
    """Parse status messages dari ESP32"""
    
    # WiFi Status
    if "Terhubung ke WiFi" in line:
        wifi_status_label.config(text="WiFi: Connected ✓", bg='green')
    elif "Mencoba menghubungkan ke WiFi" in line:
        wifi_status_label.config(text="WiFi: Connecting...", bg='orange')
    elif "Gagal terhubung ke WiFi" in line:
        wifi_status_label.config(text="WiFi: Disconnected ✗", bg='red')
        reset_durasi_labels()
    
    # MQTT Status (dari info tambahan di countdown)
    if "MQTT:OK" in line:
        mqtt_status_label.config(text="MQTT: Connected ✓", bg='green')
    elif "MQTT:DISC" in line:
        mqtt_status_label.config(text="MQTT: Disconnected ✗", bg='red')
    
    # Database/Server Status
    if "Durasi dari server:" in line:
        db_status_label.config(text="Database: Updated ✓", bg='green')
    elif "Gagal parsing JSON dari server" in line:
        db_status_label.config(text="Database: Parse Error ✗", bg='red')
        reset_durasi_labels()
    elif "Gagal koneksi ulang. Masuk mode darurat." in line:
        db_status_label.config(text="Database: Connection Failed ✗", bg='red')
        mqtt_status_label.config(text="MQTT: Disconnected ✗", bg='red')
        reset_durasi_labels()
    elif "Gagal koneksi MQTT. Mode darurat aktif." in line:
        mqtt_status_label.config(text="MQTT: Connection Failed ✗", bg='red')
        db_status_label.config(text="Database: Offline ✗", bg='red')
        reset_durasi_labels()
    elif "Gagal mengirim request update" in line:
        db_status_label.config(text="Database: Request Failed ✗", bg='orange')
    
    # Mode Status
    if "Mode:NORMAL" in line:
        mode_status_label.config(text="Mode: NORMAL ✓", bg='green')
    elif "Mode:DARURAT" in line:
        mode_status_label.config(text="Mode: DARURAT ⚠", bg='red')
    
    # System Ready
    if "System Ready" in line:
        wifi_status_label.config(text="WiFi: Connected ✓", bg='green')
        mqtt_status_label.config(text="MQTT: Ready", bg='green')

def update_rtt(line):
    """Parse dan update RTT dari countdown line"""
    try:
        if "RTT:" in line:
            # Extract RTT value: "RTT:35ms"
            rtt_part = line.split("RTT:")[1].split("ms")[0].strip()
            rtt_value = int(rtt_part)
            
            # Update label dengan color coding
            if rtt_value == 0:
                rtt_label.config(text=f"RTT: -- ms", bg='gray')
            elif rtt_value < 500:
                rtt_label.config(text=f"RTT: {rtt_value} ms ✓", bg='green')
            elif rtt_value < 1000:
                rtt_label.config(text=f"RTT: {rtt_value} ms", bg='orange')
            else:
                rtt_label.config(text=f"RTT: {rtt_value} ms ⚠", bg='red')
    except Exception as e:
        pass  # Ignore parse errors

def update_display(line):
    """Parse dan update countdown & warna lampu dengan presisi 10ms (2 desimal)"""
    try:
        if "Countdown:" in line:
            # Extract countdown part: "Countdown: AHijau=27.45, BMerah=27.45, ..."
            countdown_part = line.split("Countdown:")[1].split("|")[0].strip()
            
            # Split by comma
            pairs = countdown_part.split(",")
            
            for p in pairs:
                p = p.strip()
                if "=" in p:
                    # Extract: "AHijau=27.45" -> titik='A', warna='Hijau', nilai=27.45
                    label_part, value_part = p.split("=")
                    
                    titik = label_part[0].upper()  # A, B, C, D
                    warna = label_part[1:].upper()  # HIJAU, MERAH, KUNING
                    
                    # ✅ Parse sebagai float untuk mendukung desimal
                    try:
                        nilai = float(value_part.strip())
                    except:
                        nilai = 0.0
                    
                    # ✅ Update counter dengan 2 desimal - TETAP PUTIH seperti original
                    if titik in counters:
                        counters[titik].config(text=f"{nilai:.2f}")
                    
                    # Update warna lampu
                    if titik in traffic_lights:
                        canvas = traffic_lights[titik]['canvas']
                        
                        # Reset semua lampu ke gray
                        for w in ["RED", "YELLOW", "GREEN"]:
                            canvas.itemconfig(traffic_lights[titik]['lights'][w], fill='gray')
                        
                        # Nyalakan lampu yang aktif
                        warna_map = {
                            "MERAH": "RED",
                            "KUNING": "YELLOW",
                            "HIJAU": "GREEN"
                        }
                        
                        if warna in warna_map:
                            color_name = warna_map[warna].lower()
                            canvas.itemconfig(
                                traffic_lights[titik]['lights'][warna_map[warna]],
                                fill=color_name
                            )
    except Exception as e:
        print("Error parsing countdown:", e)

def update_durasi_server(line):
    """Parse dan update durasi dari server dengan 2 desimal"""
    try:
        # Ekstrak bagian data saja (setelah "Durasi dari server:" jika ada)
        if "Durasi dari server:" in line:
            # Ambil bagian setelah ":"
            data_part = line.split("Durasi dari server:")[1].strip()
        else:
            data_part = line
        
        # Cek apakah semua titik ada
        if "A=" in data_part and "B=" in data_part and "C=" in data_part and "D=" in data_part:
            # Format: "A=27.45s, B=17.89s, C=26.12s, D=31.56s"
            # Hapus 's' dan split by comma
            clean = data_part.replace("s", "").strip()
            parts = clean.split(",")
            
            for part in parts:
                part = part.strip()  # Trim whitespace
                if "=" in part:
                    titik, nilai = part.split("=")
                    titik = titik.strip().upper()  # Clean whitespace
                    nilai = nilai.strip()
                    
                    if titik in durasi_labels:
                        try:
                            # ✅ Parse sebagai float dan tampilkan dengan 2 desimal
                            nilai_float = float(nilai)
                            durasi_labels[titik].config(text=f"{titik} = {nilai_float:.2f} s")
                        except:
                            # Fallback jika gagal parse
                            durasi_labels[titik].config(text=f"{titik} = {nilai} s")
    except Exception as e:
        print("Error parsing durasi server:", e)

# === SERIAL LOOP ===
def serial_loop():
    """Background thread untuk membaca serial port dengan optimasi untuk 10ms update"""
    global ser
    
    while True:
        # Coba buka serial port jika belum terbuka
        if ser is None or not ser.is_open:
            try:
                print(f"Mencoba membuka port serial {PORT}...")
                ser = serial.Serial(PORT, BAUDRATE, timeout=0.01)  # ✅ Timeout lebih pendek
                print(f"✅ Serial Terhubung di {PORT}")
                root.after(0, lambda: wifi_status_label.config(
                    text="Serial: Connected", bg='blue'))
            except Exception as e:
                print(f"Gagal membuka port serial: {e}")
                root.after(0, lambda: wifi_status_label.config(
                    text="Serial: Disconnected", bg='red'))
                time.sleep(2)
                continue
        
        # Baca data dari serial
        try:
            # ✅ Optimasi: Baca multiple lines untuk mengurangi overhead
            lines_to_process = []
            while ser.in_waiting:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    lines_to_process.append(line)
            
            # Process semua lines yang terbaca
            for line in lines_to_process:
                if line:  # Jangan proses line kosong
                    print(f"ESP32: {line}")
                    
                    # Update GUI (harus di main thread)
                    root.after(0, update_status_messages, line)
                    root.after(0, update_rtt, line)
                    root.after(0, update_display, line)
                    
                    # Update durasi jika ada
                    if "A=" in line and "B=" in line:
                        root.after(0, update_durasi_server, line)
                        
        except serial.SerialException as e:
            print(f"Serial error: {e}")
            try:
                ser.close()
            except:
                pass
            ser = None
            root.after(0, lambda: wifi_status_label.config(
                text="Serial: Disconnected", bg='red'))
            
        except Exception as e:
            print(f"Unexpected error: {e}")
        
        # ✅ Sleep time dikurangi untuk responsivitas 10ms
        time.sleep(0.01)  # 10ms polling interval

# === START BACKGROUND THREAD ===
threading.Thread(target=serial_loop, daemon=True).start()

# === START GUI ===
root.mainloop()
