import cv2
import numpy as np
import torch
import time
import argparse
import mysql.connector
import csv
import os
from ultralytics import YOLO
from ocsort.ocsort import OCSort
from collections import defaultdict, deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import queue
import logging

logging.getLogger("ultralytics").setLevel(logging.WARNING)

# --- Global ---
enable_hits = []
enable_active = False

kendaraan_minimal = 8
MIN_CYCLE_DURATION_SEC = 10.0
ENABLE_IDLE_TIMEOUT_FACTOR = 0.5

TRAJECTORY_MAXLEN = 60
WINDOW_DURATION = 3.0
THRESHOLD_PERCENTAGE = 0.10
Duplicated_IoU_ID = 0.3

frame_queue = queue.Queue(maxsize=2)
result_image_queue = queue.Queue(maxsize=2)
status_queue = queue.Queue(maxsize=10)

running = False
cap = None
model = None
tracker = None
trajectories = None
track_last_positions = None
vehicle_counter = 0
vehicle_types_count = None

vehicle_timestamps = deque()
first_vehicle_time = None
last_vehicle_time = None

counted_tracks = None


def insert_vehicle_count_to_db(count_total, mobil, motor, bis, truk, durasi_efisien, avg_fps):
    """Insert data ke database Way D (North Lane) + log delay + rata-rata FPS ke CSV."""
    start_time = time.perf_counter()
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="atl_2025"
        )
        cursor = conn.cursor()

        sql = """INSERT INTO wayd 
                 (MotorD, MobilD, TruckD, BusD, CountWayD, durasiefisienD) 
                 VALUES (%s, %s, %s, %s, %s, %s)"""

        cursor.execute(sql, (motor, mobil, truk, bis, count_total, durasi_efisien))
        conn.commit()  # commit wajib agar insert persisten [web:9]

        delay_sec = time.perf_counter() - start_time

        csv_dir = "DelayLogg"
        csv_name = "UTARA_db_delay_log.csv"
        os.makedirs(csv_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, csv_name)

        headers = [
            "timestamp", "delay_sec", "avg_fps",
            "count_total", "mobil", "motor", "bis", "truk", "durasi_efisien"
        ]

        max_retries = 5
        retry_delay = 0.2

        for attempt in range(max_retries):
            try:
                file_exists = os.path.isfile(csv_path)
                with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(headers)
                    writer.writerow([
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        f"{delay_sec:.4f}",
                        f"{avg_fps:.2f}",
                        count_total, mobil, motor, bis, truk, f"{durasi_efisien:.2f}"
                    ])
                break
            except PermissionError as pe:
                if attempt < max_retries - 1:
                    print(f"⚠️ File CSV sedang digunakan, mencoba lagi ({attempt + 1}/{max_retries})...")
                    time.sleep(retry_delay)
                else:
                    print(f"✗ Gagal menyimpan ke CSV setelah {max_retries} percobaan: {pe}")
                    print(f"   Data: Total={count_total}, Mobil={mobil}, Motor={motor}, FPS={avg_fps:.2f}")
            except Exception as e:
                print(f"✗ Error tidak terduga saat menulis CSV: {e}")
                break

        cursor.close()
        conn.close()

        print("✓ Data tersimpan ke database (North Lane):")
        print(f"  Total Kendaraan: {count_total}")
        print(f"  Mobil: {mobil} | Motor: {motor} | Bus: {bis} | Truk: {truk}")
        print(f"  Durasi Efektif: {durasi_efisien:.2f} detik")
        print(f"  Rata-rata FPS: {avg_fps:.2f}")
        print(f"  Delay insert DB (Way D): {delay_sec:.4f} detik")
        print("-" * 50)

    except mysql.connector.Error as err:
        print(f"✗ Database Error: {err}")
    except Exception as e:
        print(f"✗ Error umum: {e}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolo11n-15k.pt')
    parser.add_argument('--iou', type=float, default=0.5)
    parser.add_argument('--cam-index', type=int, default=6)
    return parser.parse_args()


counting_lines = [
    {"start": (100, 250), "end": (480, 420)},   # Line 0 - Bawah (counting)
    {"start": (0, 0), "end": (0, 0)},           # Line 1 - Atas (disabled)
    {"start": (230, 250), "end": (500, 400)}    # Line 2 - Enable
]

block_areas = [
    (0, 0, 220, 140),
    (0, 0, 135, 260),
    (475, 10, 505, 120),
]


def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    inter = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    return inter / (area1 + area2 - inter + 1e-6)


def id_to_color(id_int):
    return (int(id_int * 37) % 255, int(id_int * 17) % 255, int(id_int * 97) % 255)


def ccw(A, B, C):
    return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])


def line_intersect(p1, p2, a, b):
    return ccw(p1, a, b) != ccw(p2, a, b) and ccw(p1, p2, a) != ccw(p1, p2, b)


def check_direction_down(prev, curr):
    return curr[1] > prev[1]


def get_vehicle_count_in_window(timestamps, current_time, window_duration):
    count = 0
    window_start = current_time - window_duration
    for ts in timestamps:
        if window_start <= ts <= current_time:
            count += 1
    return count


def clean_old_timestamps(timestamps, current_time, max_duration=15.0):
    cutoff_time = current_time - max_duration
    while timestamps and timestamps[0] < cutoff_time:
        timestamps.popleft()


def processing_thread(args):
    global running, cap, model, tracker, trajectories, track_last_positions
    global vehicle_counter, vehicle_types_count
    global enable_hits, enable_active
    global first_vehicle_time, last_vehicle_time
    global counted_tracks, vehicle_timestamps
    global THRESHOLD_PERCENTAGE

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(args.weights).to(device)
    class_names = model.names

    conf_thresholds = {'motor': 0.25, 'mobil': 0.4, 'truk': 0.45, 'bis': 0.50}

    tracker = OCSort(det_thresh=0.1, iou_threshold=args.iou, delta_t=3, inertia=0.2)
    trajectories = defaultdict(lambda: deque(maxlen=TRAJECTORY_MAXLEN))
    track_last_positions = {}

    counted_tracks = defaultdict(lambda: {0: False, 1: False, 2: False})
    vehicle_timestamps = deque()

    cap = cv2.VideoCapture(args.cam_index)
    if not cap.isOpened():
        status_queue.put("ERROR: Kamera tidak terbuka!")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    vehicle_counter = 0
    vehicle_types_count = {k: 0 for k in ['mobil', 'motor', 'person', 'bis', 'truk', 'train']}
    first_vehicle_time = None
    last_vehicle_time = None

    model_fps_history = []
    cycle_fps_history = []
    frame_count = 0

    # watchdog enable idle
    enable_on_time = None
    last_count_time = None

    status_queue.put(("RESOLUTION", width, height))
    status_queue.put("RUNNING")

    print(f"\n{'='*50}")
    print("🎥 North Lane Vehicle Counter Started")
    print(f"{'='*50}")
    print(f"Minimal kendaraan: {kendaraan_minimal}")
    print(f"Min durasi siklus (DB): {MIN_CYCLE_DURATION_SEC}s")
    print(f"Enable idle timeout: {MIN_CYCLE_DURATION_SEC * ENABLE_IDLE_TIMEOUT_FACTOR:.2f}s")
    print(f"IoU Threshold: {Duplicated_IoU_ID}")
    print(f"{'='*50}\n")

    def reset_cycle_state(reason: str):
        nonlocal enable_on_time, last_count_time
        global vehicle_counter, vehicle_types_count, enable_active, enable_hits
        global first_vehicle_time, last_vehicle_time, counted_tracks, vehicle_timestamps

        print("\n" + "-" * 50)
        print(f"🔄 RESET SIKLUS ({reason})")
        print("-" * 50)

        vehicle_counter = 0
        vehicle_types_count = {k: 0 for k in vehicle_types_count}
        enable_active = False
        enable_hits = []
        first_vehicle_time = None
        last_vehicle_time = None
        counted_tracks = defaultdict(lambda: {0: False, 1: False, 2: False})
        vehicle_timestamps.clear()
        cycle_fps_history.clear()

        enable_on_time = None
        last_count_time = None

    def disable_enable_only(reason: str):
        nonlocal enable_on_time, last_count_time
        global enable_active, enable_hits, counted_tracks

        print("\n" + "-" * 50)
        print(f"🟡 DISABLE ENABLE ({reason})")
        print("-" * 50)

        enable_active = False
        enable_hits = []
        counted_tracks = defaultdict(lambda: {0: False, 1: False, 2: False})
        enable_on_time = None
        last_count_time = None

    while running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        frame_count += 1
        current_time = time.time()

        # block areas
        for (x1, y1, x2, y2) in block_areas:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)

        # enable idle-timeout: enable ON tapi belum ada hit counting
        if enable_active and vehicle_counter == 0 and enable_on_time is not None and last_count_time is None:
            idle_timeout = MIN_CYCLE_DURATION_SEC * ENABLE_IDLE_TIMEOUT_FACTOR
            if (current_time - enable_on_time) >= idle_timeout:
                disable_enable_only(f"Enable ON tanpa hit >= {idle_timeout:.2f}s")

        try:
            frame_queue.put_nowait(frame.copy())
        except queue.Full:
            pass

        model_start = time.time()
        results = model(frame, conf=0.1, verbose=False)[0]
        model_time = time.time() - model_start

        detections = []
        motor_boxes = []
        person_boxes = []
        detection_to_class = {}

        for i, box in enumerate(results.boxes.data.cpu().numpy()):
            x1, y1, x2, y2, conf, class_id = box
            class_id = int(class_id)
            class_name = class_names.get(class_id, "Unknown")

            if class_name in conf_thresholds and conf < conf_thresholds[class_name]:
                continue

            detection_to_class[i] = class_id

            if class_name == "motor":
                motor_boxes.append((x1, y1, x2, y2, conf, class_id, i))
            elif class_name == "person":
                person_boxes.append((x1, y1, x2, y2, conf, class_id, i))
            else:
                detections.append([x1, y1, x2, y2, conf])

        # gabung motor + orang dengan IoU custom
        for motor_box in motor_boxes:
            mx1, my1, mx2, my2, mconf, _, _ = motor_box
            for person_box in person_boxes[:]:
                px1, py1, px2, py2, _, _, _ = person_box
                if calculate_iou((mx1, my1, mx2, my2), (px1, py1, px2, py2)) > Duplicated_IoU_ID:
                    person_boxes.remove(person_box)
            detections.append([mx1, my1, mx2, my2, mconf])

        for pb in person_boxes:
            detections.append([pb[0], pb[1], pb[2], pb[3], pb[4]])

        detections_np = np.array(detections, dtype=np.float32) if detections else np.empty((0, 5))
        tracks = tracker.update(detections_np, (height, width), (height, width))

        track_class_map = {}
        for track in tracks:
            track_id = int(track[4])
            for j, det in enumerate(detections):
                if abs(det[0] - track[0]) < 5 and abs(det[1] - track[1]) < 5:
                    track_class_map[track_id] = detection_to_class.get(j, -1)
                    break

        vis_frame = frame.copy()

        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track[:5])
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            class_id = track_class_map.get(track_id, -1)
            class_name = class_names.get(class_id, "Unknown")
            trajectories[track_id].append(center)

            if track_id not in track_last_positions:
                track_last_positions[track_id] = {}

            for i, line in enumerate(counting_lines):
                prev = track_last_positions[track_id].get(i)
                a, b = line["start"], line["end"]

                if prev and line_intersect(prev, center, a, b):

                    # Enable Line (index 2)
                    if i == 2 and center[1] > prev[1]:
                        enable_hits.append(current_time)
                        enable_hits[:] = [t for t in enable_hits if current_time - t <= 1.0]
                        if len(enable_hits) >= 2 and not enable_active:
                            enable_active = True
                            enable_on_time = current_time
                            last_count_time = None
                            enable_hits = []
                            print("🟢 ENABLE AKTIF (double-hit line 2)")

                    # Counting Line 0
                    elif i == 0 and enable_active and not counted_tracks[track_id][0]:
                        if check_direction_down(prev, center):
                            vehicle_counter += 1
                            counted_tracks[track_id][0] = True
                            vehicle_timestamps.append(current_time)

                            last_count_time = current_time

                            if first_vehicle_time is None:
                                first_vehicle_time = current_time
                                print("🚦 Siklus dimulai - Kendaraan pertama terdeteksi")
                            last_vehicle_time = current_time

                            durasi_saat_ini = last_vehicle_time - first_vehicle_time if first_vehicle_time else 0.0
                            print(f"🚗 Kendaraan #{vehicle_counter} ({class_name}) | Line 0 | Durasi: {durasi_saat_ini:.2f}s")

                            if class_name == "mobil":
                                vehicle_types_count['mobil'] += 1
                            elif class_name in ["motor", "person"]:
                                vehicle_types_count['motor'] += 1
                            elif class_name == "bis":
                                vehicle_types_count['bis'] += 1
                            elif class_name in ["truk", "train"]:
                                vehicle_types_count['truk'] += 1

                    # Counting Line 1 (kalau suatu saat diaktifkan)
                    elif i == 1 and enable_active and not counted_tracks[track_id][1]:
                        if check_direction_down(prev, center):
                            vehicle_counter += 1
                            counted_tracks[track_id][1] = True
                            vehicle_timestamps.append(current_time)

                            last_count_time = current_time

                            if first_vehicle_time is None:
                                first_vehicle_time = current_time
                                print("🚦 Siklus dimulai - Kendaraan pertama terdeteksi")
                            last_vehicle_time = current_time

                            durasi_saat_ini = last_vehicle_time - first_vehicle_time if first_vehicle_time else 0.0
                            print(f"🚗 Kendaraan #{vehicle_counter} ({class_name}) | Line 1 | Durasi: {durasi_saat_ini:.2f}s")

                            if class_name == "mobil":
                                vehicle_types_count['mobil'] += 1
                            elif class_name in ["motor", "person"]:
                                vehicle_types_count['motor'] += 1
                            elif class_name == "bis":
                                vehicle_types_count['bis'] += 1
                            elif class_name in ["truk", "train"]:
                                vehicle_types_count['truk'] += 1

                track_last_positions[track_id][i] = center

            color = id_to_color(track_id)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID: {track_id} | {class_name}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(vis_frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(vis_frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            traj = list(trajectories[track_id])
            for j in range(1, len(traj)):
                cv2.line(vis_frame, traj[j - 1], traj[j], color, 2)
            cv2.circle(vis_frame, center, 3, color, -1)

        # ===== SLIDING WINDOW CHECK (FIX LOGIC) =====
        clean_old_timestamps(vehicle_timestamps, current_time)
        curr_win, prev_win = 0, 0
        ratio_percent = 0.0

        if enable_active and len(vehicle_timestamps) > 0:
            curr_win = get_vehicle_count_in_window(vehicle_timestamps, current_time, WINDOW_DURATION)
            prev_center_time = current_time - WINDOW_DURATION
            prev_win = get_vehicle_count_in_window(vehicle_timestamps, prev_center_time, WINDOW_DURATION)

            if prev_win > 0:
                ratio_percent = (curr_win / prev_win) * 100.0

            threshold_percent = THRESHOLD_PERCENTAGE * 100.0

            # FIX: cek siklus selesai dulu
            if prev_win > 0 and ratio_percent < threshold_percent:
                durasi_efisien = (last_vehicle_time - first_vehicle_time) if (first_vehicle_time and last_vehicle_time) else 0.0
                avg_cycle_fps = (sum(cycle_fps_history) / len(cycle_fps_history)) if cycle_fps_history else 0.0

                ok_count = vehicle_counter >= kendaraan_minimal
                ok_durasi = durasi_efisien >= MIN_CYCLE_DURATION_SEC

                print("\n" + "=" * 50)
                print("⏱️  SIKLUS SELESAI (North Lane - Ratio Trigger)")
                print("=" * 50)
                print(f"📊 Threshold Ratio: {threshold_percent:.1f}%")
                print(f"📊 Win Sekarang: {curr_win} kendaraan")
                print(f"📊 Win Sebelumnya: {prev_win} kendaraan")
                print(f"📉 Ratio Sekarang: {ratio_percent:.1f}%")
                print(f"📦 Count total: {vehicle_counter}")
                print(f"🕒 Durasi efektif: {durasi_efisien:.2f}s (min {MIN_CYCLE_DURATION_SEC:.2f}s)")

                if ok_count and ok_durasi:
                    insert_vehicle_count_to_db(
                        count_total=vehicle_counter,
                        mobil=vehicle_types_count['mobil'],
                        motor=vehicle_types_count['motor'],
                        bis=vehicle_types_count['bis'],
                        truk=vehicle_types_count['truk'],
                        durasi_efisien=durasi_efisien,
                        avg_fps=avg_cycle_fps
                    )
                    reset_cycle_state("Lolos syarat -> Insert DB")
                else:
                    alasan = []
                    if not ok_count:
                        alasan.append(f"Count<{kendaraan_minimal}")
                    if not ok_durasi:
                        alasan.append(f"Durasi<{MIN_CYCLE_DURATION_SEC:.2f}s")
                    reset_cycle_state("Tidak lolos syarat (" + ", ".join(alasan) + ")")

            # baru cek lampu merah
            elif prev_win > 0 and ratio_percent == 0.0 and curr_win == 0:
                print("\n⚠️ DETEKSI LAMPU MERAH - Enable Line Direset (Ratio 0%)")
                reset_cycle_state("Lampu merah / window kosong")

        # FPS
        model_fps_history.append(1.0 / model_time if model_time > 0 else 0.0)
        avg_fps = sum(model_fps_history[-30:]) / min(30, len(model_fps_history))

        if enable_active:
            cycle_fps_history.append(avg_fps)

        # Overlay
        overlay = vis_frame.copy()
        cv2.rectangle(overlay, (10, 10), (320, 420), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, vis_frame, 0.4, 0, vis_frame)

        durasi_display = 0.0
        if first_vehicle_time is not None:
            durasi_display = current_time - first_vehicle_time

        stats = [
            "North Lane",
            f"Model FPS: {avg_fps:.1f}",
            f"Objects: {len(tracks)}",
            f"Frame: {frame_count}",
            f"Enable: {'ON' if enable_active else 'OFF'}",
            f"Count: {vehicle_counter}",
            f"Mobil: {vehicle_types_count['mobil']}",
            f"Motor: {vehicle_types_count['motor']}",
            f"Bis: {vehicle_types_count['bis']}",
            f"Truk: {vehicle_types_count['truk']}",
            f"Durasi: {durasi_display:.1f}s",
            f"Threshold: {int(THRESHOLD_PERCENTAGE * 100)}%",
            f"Ratio: {ratio_percent:.1f}%",
            f"Win Now: {curr_win}",
            f"Win Prev: {prev_win}",
            f"EnableIdle: {MIN_CYCLE_DURATION_SEC * ENABLE_IDLE_TIMEOUT_FACTOR:.1f}s"
        ]

        for i, s in enumerate(stats):
            cv2.putText(
                vis_frame, s, (20, 30 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 255, 255) if i == 0 else (0, 255, 0), 2
            )

        for i, line in enumerate(counting_lines):
            if line["start"] != (0, 0) and line["end"] != (0, 0):
                if i < 2:
                    line_color = (0, 255, 0) if enable_active else (255, 0, 0)
                else:
                    line_color = (0, 0, 255)
                cv2.line(vis_frame, line["start"], line["end"], line_color, 3)

        try:
            result_image_queue.put_nowait(vis_frame)
        except queue.Full:
            pass

    cap.release()
    status_queue.put("STOPPED")


class App:
    def __init__(self, root, args):
        self.root = root
        self.root.title("Vehicle Counter - North Lane (Configurable)")
        self.root.configure(bg='#1e1e1e')
        self.args = args
        self.thread = None

        self.orig_width = 640
        self.orig_height = 480
        self.display_width = 640
        self.display_height = 480

        header = tk.Frame(root, bg='#2c3e50', height=50)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        tk.Label(
            header, text="North Lane",
            font=("Segoe UI", 14, "bold"),
            fg="white", bg='#2c3e50'
        ).pack(side=tk.LEFT, padx=15, pady=10)

        self.status_dot = tk.Canvas(header, width=16, height=16,
                                    bg='#2c3e50', highlightthickness=0)
        self.status_dot.pack(side=tk.LEFT, padx=10, pady=15)
        self.status_dot.create_oval(4, 4, 12, 12, fill="red", tags="dot")

        btn_frame = tk.Frame(header, bg='#2c3e50')
        btn_frame.pack(side=tk.RIGHT, padx=15, pady=8)

        self.start_btn = tk.Button(
            btn_frame, text="Start", bg='#27ae60', fg="white",
            font=("Segoe UI", 10, "bold"),
            relief=tk.FLAT, padx=20, command=self.start
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = tk.Button(
            btn_frame, text="Stop", bg='#c0392b', fg="white",
            font=("Segoe UI", 10, "bold"),
            relief=tk.FLAT, padx=20, command=self.stop, state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        settings_frame = tk.Frame(root, bg='#34495e', height=80)
        settings_frame.pack(fill=tk.X, padx=20, pady=(10, 0))
        settings_frame.pack_propagate(False)

        threshold_frame = tk.Frame(settings_frame, bg='#34495e')
        threshold_frame.pack(side=tk.LEFT, padx=20, pady=10)

        tk.Label(
            threshold_frame, text="Threshold Ratio:",
            font=("Segoe UI", 10, "bold"),
            fg="white", bg='#34495e'
        ).pack(side=tk.LEFT, padx=(0, 10))

        self.threshold_var = tk.DoubleVar(value=THRESHOLD_PERCENTAGE * 100)

        self.threshold_slider = ttk.Scale(
            threshold_frame, from_=5, to=30,
            variable=self.threshold_var, orient=tk.HORIZONTAL,
            length=200, command=self.on_threshold_change
        )
        self.threshold_slider.pack(side=tk.LEFT, padx=5)

        self.threshold_label = tk.Label(
            threshold_frame, text=f"{int(self.threshold_var.get())}%",
            font=("Segoe UI", 11, "bold"),
            fg="#3498db", bg='#34495e', width=5
        )
        self.threshold_label.pack(side=tk.LEFT, padx=10)

        preset_frame = tk.Frame(settings_frame, bg='#34495e')
        preset_frame.pack(side=tk.LEFT, padx=20)

        tk.Label(
            preset_frame, text="Presets:",
            font=("Segoe UI", 9),
            fg="white", bg='#34495e'
        ).pack(side=tk.LEFT, padx=(0, 5))

        presets = [
            ("Very Sensitive", 5, "#e74c3c"),
            ("Sensitive", 8, "#e67e22"),
            ("Normal", 10, "#27ae60"),
            ("Tolerant", 15, "#3498db"),
            ("Very Tolerant", 20, "#9b59b6")
        ]

        for name, value, color in presets:
            btn = tk.Button(
                preset_frame, text=f"{value}%",
                bg=color, fg="white",
                font=("Segoe UI", 8),
                relief=tk.FLAT, padx=8, pady=2,
                command=lambda v=value: self.set_preset(v)
            )
            btn.pack(side=tk.LEFT, padx=2)

        info_frame = tk.Frame(settings_frame, bg='#34495e')
        info_frame.pack(side=tk.RIGHT, padx=20)

        self.sensitivity_label = tk.Label(
            info_frame, text="Sensitivity: Normal",
            font=("Segoe UI", 10, "bold"),
            fg="#27ae60", bg='#34495e'
        )
        self.sensitivity_label.pack()

        self.info_label = tk.Label(
            info_frame, text="Balanced detection",
            font=("Segoe UI", 8),
            fg="#95a5a6", bg='#34495e'
        )
        self.info_label.pack()

        self.video_frame = tk.Frame(root, bg='black')
        self.video_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=(10, 20))

        self.video_label = tk.Label(self.video_frame, bg='black', bd=0)
        self.video_label.place(relx=0.5, rely=0.5, anchor="center")

        self.video_frame.bind('<Configure>', self.on_resize)
        self.root.bind('<F11>', self.toggle_fullscreen)
        self.root.bind('<Escape>', self.exit_fullscreen)
        self.root.minsize(800, 600)

        self.is_fullscreen = False
        self.update_gui()

    def on_threshold_change(self, value):
        global THRESHOLD_PERCENTAGE
        threshold_value = float(value)
        THRESHOLD_PERCENTAGE = threshold_value / 100.0
        self.threshold_label.config(text=f"{int(threshold_value)}%")
        self.update_sensitivity_info(threshold_value)

    def set_preset(self, value):
        self.threshold_var.set(value)
        self.on_threshold_change(value)

    def update_sensitivity_info(self, value):
        if value <= 5:
            sensitivity = "Very Sensitive"
            color = "#e74c3c"
            info = "Stops quickly with small traffic decrease"
        elif value <= 8:
            sensitivity = "Sensitive"
            color = "#e67e22"
            info = "Good for variable traffic patterns"
        elif value <= 12:
            sensitivity = "Normal"
            color = "#27ae60"
            info = "Balanced detection (recommended)"
        elif value <= 18:
            sensitivity = "Tolerant"
            color = "#3498db"
            info = "Allows more traffic fluctuation"
        else:
            sensitivity = "Very Tolerant"
            color = "#9b59b6"
            info = "Only stops on significant decrease"

        self.sensitivity_label.config(text=f"Sensitivity: {sensitivity}", fg=color)
        self.info_label.config(text=info)

    def toggle_fullscreen(self, event=None):
        self.is_fullscreen = not self.is_fullscreen
        self.root.attributes("-fullscreen", self.is_fullscreen)
        return "break"

    def exit_fullscreen(self, event=None):
        self.is_fullscreen = False
        self.root.attributes("-fullscreen", False)
        return "break"

    def on_resize(self, event=None):
        if not hasattr(self, 'orig_width') or self.orig_width <= 0:
            return

        width = self.video_frame.winfo_width()
        height = self.video_frame.winfo_height()
        if width <= 1 or height <= 1:
            return

        ratio = self.orig_width / self.orig_height
        new_width = int(height * ratio)
        new_height = height

        if new_width > width:
            new_width = width
            new_height = int(width / ratio)

        new_width = max(100, new_width)
        new_height = max(75, new_height)

        self.display_width = new_width
        self.display_height = new_height
        self.video_label.place(relx=0.5, rely=0.5, anchor="center")

    def start(self):
        global running
        if not running:
            running = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.status_dot.itemconfig("dot", fill="yellow")
            self.thread = threading.Thread(
                target=processing_thread,
                args=(self.args,),
                daemon=True
            )
            self.thread.start()

    def stop(self):
        global running
        if running:
            running = False
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)

    def on_close(self):
        global running
        running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        self.root.destroy()

    def update_gui(self):
        while not status_queue.empty():
            msg = status_queue.get_nowait()
            if isinstance(msg, tuple) and msg[0] == "RESOLUTION":
                self.orig_width, self.orig_height = msg[1], msg[2]
            elif msg == "RUNNING":
                self.status_dot.itemconfig("dot", fill="green")
            elif msg == "STOPPED":
                self.status_dot.itemconfig("dot", fill="red")

        if not result_image_queue.empty():
            frame = result_image_queue.get_nowait()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.display_width, self.display_height))
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.root.after(20, self.update_gui)


if __name__ == "__main__":
    args = parse_args()
    root = tk.Tk()
    app = App(root, args)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
