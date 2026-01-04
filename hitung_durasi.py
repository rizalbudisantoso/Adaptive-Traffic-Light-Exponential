import mysql.connector
import time
from datetime import datetime
import uuid
import numpy as np
import csv
import os


# Konfigurasi database
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'atl_2025'
}


# Fungsi untuk ambil ID terakhir dari tabel
def get_last_id(table_name):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(f"SELECT id FROM {table_name} ORDER BY id DESC LIMIT 1")
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        return result[0] if result else 0
    except mysql.connector.Error as err:
        print(f"❌ Error membaca ID {table_name}: {err}")
        return 0


# Fungsi ambil data lengkap terakhir
def get_last_entry(table_name):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute(f"SELECT * FROM {table_name} ORDER BY id DESC LIMIT 1")
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        return result
    except mysql.connector.Error as err:
        print(f"❌ Error membaca data {table_name}: {err}")
        return None


# Fungsi hitung green time - DIPERBAIKI: return float 2 desimal
def hitung_lama_hijau(mobil, motor, bus, truck, wait, prev_green_time=10.0, jalur='X'):
    """
    Menghitung durasi lampu hijau dengan pembobotan eksponensial.
    Return: float dengan 2 digit desimal
    """
    # Bobot awal untuk setiap jenis kendaraan
    bobot_awal = {"Motor": 1.7, "Mobil": 2.5, "Bus": 3.4, "Truk": 3.45}
    # Parameter ruas jalan
    ruas_jalan = {'A': 4.5, 'B': 4.0, 'C': 4.0, 'D': 4.5}
    # Parameter eksponensial
    k = 0.15
    # Konstanta a berdasarkan ruas jalan
    a = ruas_jalan[jalur] / 3

    # Jumlah kendaraan
    jumlah_kendaraan = {"Mobil": mobil, "Motor": motor, "Bus": bus, "Truk": truck}
    total_kendaraan = mobil + motor + bus + truck

    if total_kendaraan == 0:
        print(f"⚠️ Jalur {jalur} tidak ada kendaraan.")
        return 0.00

    # Fungsi bobot efektif eksponensial kumulatif
    def bobot_efektif(w, n, k):
        if n == 0:
            return 0.0
        return w * (1 - np.exp(-k * n)) / (1 - np.exp(-k))

    # Hitung total durasi
    green_time = sum(
        bobot_efektif(bobot_awal[jenis], jumlah, k)
        for jenis, jumlah in jumlah_kendaraan.items()
    ) / a
    
    # ✅ PERBAIKAN: Bulatkan ke 2 desimal sejak awal
    green_time = round(green_time, 2)

    print(f"Durasi hijau jalur {jalur}: {green_time:.2f} detik")
    return green_time


# ✅ DIPERBAIKI: Simpan dengan 2 desimal, tetap ada minimum batas
def simpan_durasi(gta, gtb, gtc, gtd):
    """
    Menyimpan durasi ke database dengan 2 digit desimal.
    Tetap menerapkan batas minimum untuk keamanan.
    """
    # Terapkan batas minimum, tapi pertahankan desimal
    final_a = max(gta, 20.00)
    final_b = max(gtb, 15.00)
    final_c = max(gtc, 15.00)
    final_d = max(gtd, 20.00)
    
    # Pastikan format 2 desimal
    final_a = round(final_a, 2)
    final_b = round(final_b, 2)
    final_c = round(final_c, 2)
    final_d = round(final_d, 2)

    start_time = time.perf_counter()

    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # ✅ Gunakan %s untuk menyimpan float
        sql = "INSERT INTO algoritmadurasi (waktu, Barat, Selatan, Timur, Utara) VALUES (%s, %s, %s, %s, %s)"
        cursor.execute(sql, (now, final_a, final_b, final_c, final_d))
        conn.commit()
        cursor.close()
        conn.close()

        end_time = time.perf_counter()
        delay_sec = end_time - start_time

        print(f"💾 Disimpan: A={final_a:.2f} B={final_b:.2f} C={final_c:.2f} D={final_d:.2f}")
        print(f"⏱️ Delay insert DB: {delay_sec:.4f} detik")

        # Log ke CSV
        log_to_csv(
            timestamp=now,
            delay_sec=delay_sec,
            gta=final_a,
            gtb=final_b,
            gtc=final_c,
            gtd=final_d
        )

    except mysql.connector.Error as err:
        print(f"❌ Gagal simpan durasi: {err}")


# Fungsi logging ke CSV - DIPERBAIKI: format 2 desimal
def log_to_csv(timestamp, delay_sec, gta, gtb, gtc, gtd):
    """
    Menyimpan log perhitungan durasi + delay insert ke CSV dengan retry mechanism.
    """
    csv_dir = "DelayLogg"
    csv_name = "algorithm_durasi_log.csv"
    os.makedirs(csv_dir, exist_ok=True)

    csv_path = os.path.join(csv_dir, csv_name)

    headers = [
        "timestamp",
        "delay_sec",
        "GWayA",
        "GWayB",
        "GWayC",
        "GWayD",
        "total_green_time"
    ]

    max_retries = 5
    retry_delay = 0.2
    
    total_green = round(gta + gtb + gtc + gtd, 2)

    for attempt in range(max_retries):
        try:
            file_exists = os.path.isfile(csv_path)

            with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                if not file_exists:
                    writer.writerow(headers)
                
                # ✅ Format dengan 2 desimal untuk CSV
                writer.writerow([
                    timestamp,
                    f"{delay_sec:.4f}",
                    f"{gta:.2f}",
                    f"{gtb:.2f}",
                    f"{gtc:.2f}",
                    f"{gtd:.2f}",
                    f"{total_green:.2f}"
                ])
            
            print(f"✅ Log tersimpan ke CSV: {csv_name}")
            break
            
        except PermissionError as pe:
            if attempt < max_retries - 1:
                print(f"⚠️ File CSV sedang digunakan, mencoba lagi ({attempt + 1}/{max_retries})...")
                time.sleep(retry_delay)
            else:
                print(f"✗ Gagal menyimpan ke CSV setelah {max_retries} percobaan: {pe}")
                print(f"   Data: A={gta:.2f}, B={gtb:.2f}, C={gtc:.2f}, D={gtd:.2f}")
        
        except Exception as e:
            print(f"✗ Error tidak terduga saat menulis CSV: {e}")
            break


# Main Loop
if __name__ == "__main__":
    last_ids = {
        'waya': get_last_id('waya'),
        'wayb': get_last_id('wayb'),
        'wayc': get_last_id('wayc'),
        'wayd': get_last_id('wayd'),
    }

    print("⏳ Memulai pemantauan data... (CTRL+C untuk berhenti)")
    print(f"📁 Log akan tersimpan di: DelayLogg/algorithm_durasi_log.csv")
    print("-" * 60)

    try:
        while True:
            time.sleep(2)

            for name in ['waya', 'wayb', 'wayc', 'wayd']:
                current_id = get_last_id(name)
                if current_id > last_ids[name]:
                    last_ids[name] = current_id
                    print(f"📥 Tabel {name.upper()} bertambah data (ID: {current_id}).")
                    
                    data_a = get_last_entry('waya')
                    data_b = get_last_entry('wayb')
                    data_c = get_last_entry('wayc')
                    data_d = get_last_entry('wayd')

                    if data_a and data_b and data_c and data_d:
                        gta = hitung_lama_hijau(data_a['MobilA'], data_a['MotorA'], data_a['BusA'], data_a['TruckA'], data_a['WaitA'], jalur='A')
                        gtb = hitung_lama_hijau(data_b['MobilB'], data_b['MotorB'], data_b['BusB'], data_b['TruckB'], data_b['WaitB'], jalur='B')
                        gtc = hitung_lama_hijau(data_c['MobilC'], data_c['MotorC'], data_c['BusC'], data_c['TruckC'], data_c['WaitC'], jalur='C')
                        gtd = hitung_lama_hijau(data_d['MobilD'], data_d['MotorD'], data_d['BusD'], data_d['TruckD'], data_d['WaitD'], jalur='D')

                        simpan_durasi(gta, gtb, gtc, gtd)
                        print("🔁 Menunggu data baru...")
                        print("-" * 60)
                    else:
                        print("⚠️ Tidak dapat menghitung durasi: Data tidak lengkap.")

    except KeyboardInterrupt:
        print("\n⛔ Pemantauan dihentikan oleh pengguna.")
