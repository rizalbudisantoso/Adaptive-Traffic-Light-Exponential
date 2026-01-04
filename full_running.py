import subprocess

files_to_run = [
    '3Barat_OBS.py',
    '3Selatan_OBS.py', 
    '3Timur_ManyCam.py',
    '3Utara_XSplit.py',
    'hitung_durasi.py'
]

processes = []

for file in files_to_run:
    process = subprocess.Popen(['python', file])
    processes.append(process)
    print(f"Menjalankan {file}...")

for process in processes:
    process.wait()

print("Semua proses selesai")
