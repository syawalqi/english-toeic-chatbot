"""
Database initialization and seed data for chatbot akademik.
"""
import sqlite3
import os
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(__file__), 'data', 'akademik.db')

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_db()
    c = conn.cursor()

    c.executescript('''
        CREATE TABLE IF NOT EXISTS jadwal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hari TEXT NOT NULL,
            mata_kuliah TEXT NOT NULL,
            jam_mulai TEXT NOT NULL,
            jam_selesai TEXT NOT NULL,
            ruang TEXT,
            dosen TEXT
        );

        CREATE TABLE IF NOT EXISTS tugas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mata_kuliah TEXT NOT NULL,
            deskripsi TEXT NOT NULL,
            deadline TEXT NOT NULL,
            status TEXT DEFAULT 'menunggu',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS info_akademik (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kategori TEXT,
            judul TEXT NOT NULL,
            isi TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT,
            tool_calls TEXT,
            reasoning_content TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS test_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scenario TEXT NOT NULL,
            category TEXT NOT NULL,
            success TEXT NOT NULL,
            response_time REAL,
            error_type TEXT,
            notes TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    ''')

    conn.commit()

def seed_data():
    conn = get_db()
    c = conn.cursor()

    # Check if already seeded
    c.execute("SELECT COUNT(*) FROM jadwal")
    if c.fetchone()[0] > 0:
        conn.close()
        return

    # Jadwal kuliah
    jadwal = [
        ('Senin', 'Metodologi Penelitian', '08:00', '09:40', 'Ruang 201', 'Dr. Ahmad'),
        ('Senin', 'Pemrograman Web', '10:00', '12:00', 'Lab Komputer 2', 'Bu Sari'),
        ('Selasa', 'AI Terapan', '10:00', '12:00', 'Lab Komputer 1', 'Pak Ghifari'),
        ('Rabu', 'Basis Data', '08:00', '09:40', 'Ruang 301', 'Pak Budi'),
        ('Rabu', 'Jaringan Komputer', '13:00', '14:40', 'Lab Jaringan', 'Bu Dewi'),
        ('Kamis', 'AI Terapan (Praktikum)', '10:00', '12:00', 'Lab Komputer 1', 'Pak Ghifari'),
        ('Jumat', 'Pendidikan Agama', '09:00', '10:40', 'Ruang 101', 'Dr. Hadi'),
    ]
    for j in jadwal:
        c.execute("INSERT INTO jadwal (hari, mata_kuliah, jam_mulai, jam_selesai, ruang, dosen) VALUES (?,?,?,?,?,?)", j)

    # Tugas
    today = datetime.now()
    tugas = [
        ('AI Terapan', 'Draft paper bab 1-3', (today + timedelta(days=14)).strftime('%Y-%m-%d')),
        ('AI Terapan', 'Implementasi chatbot asisten akademik', (today + timedelta(days=7)).strftime('%Y-%m-%d')),
        ('Metodologi Penelitian', 'Revisi proposal bab 2', (today + timedelta(days=10)).strftime('%Y-%m-%d')),
        ('Pemrograman Web', 'Project akhir: sistem informasi', (today + timedelta(days=21)).strftime('%Y-%m-%d')),
        ('Basis Data', 'Normalisasi database', (today + timedelta(days=5)).strftime('%Y-%m-%d')),
    ]
    for t in tugas:
        c.execute("INSERT INTO tugas (mata_kuliah, deskripsi, deadline) VALUES (?,?,?)", t)

    # Info akademik
    info = [
        ('akademik', 'Jadwal UTS', 'UTS Semester Gasal 2025/2026 dilaksanakan pada minggu ke-9, sekitar pertengahan Oktober 2025.'),
        ('akademik', 'Jadwal UAS', 'UAS Semester Gasal 2025/2026 dilaksanakan pada minggu ke-16, sekitar pertengahan Desember 2025.'),
        ('akademik', 'Aturan Revisi Nilai', 'Revisi nilai dapat diajukan maksimal 14 hari setelah nilai diumumkan dengan mengisi formulir revisi yang ditandatangani dosen pengampu.'),
        ('akademik', 'Syarat Sidang Skripsi', 'Syarat sidang skripsi: minimal 138 SKS, IPK ≥ 2.0, lulus semua mata kuliah wajib, telah menempuh seminar proposal, dan ACC dosen pembimbing.'),
        ('beasiswa', 'Beasiswa Prestasi', 'Beasiswa prestasi akademik dibuka setiap semester. Syarat: IPK ≥ 3.5, aktif organisasi, tidak menerima beasiswa lain. Pendaftaran melalui portal mahasiswa.'),
        ('akademik', 'Kalender Akademik 2025/2026', 'Semester Gasal: Agustus 2025 - Januari 2026. Semester Genap: Februari - Juli 2026. Liburan: Juli-Agustus 2026.'),
    ]
    for i in info:
        c.execute("INSERT INTO info_akademik (kategori, judul, isi) VALUES (?,?,?)", i)

    conn.commit()
    conn.close()

if __name__ == '__main__':
    init_db()
    seed_data()
    print("✅ Database initialized and seeded.")
