#!/usr/bin/env python3
"""Run 100 test scenarios against chatbot API"""
import json, time, uuid, subprocess, sys, os

BASE = "http://localhost:9120/api/chat"
OUT = "/root/.hermes/workspace/test-results-100.json"

scenarios = []
sid = 0
def add(kat, prompt, exp=None, diff="medium"):
    global sid; sid += 1
    scenarios.append({"id":sid,"kategori":kat,"prompt":prompt,"exp":exp,"diff":diff})

# A. Jadwal (25)
add("A_Jadwal","Jadwal hari ini apa?","get_jadwal","easy")
add("A_Jadwal","Jam berapa AI Terapan?","get_jadwal","easy")
add("A_Jadwal","Kuliah apa hari Senin?","get_jadwal","easy")
add("A_Jadwal","Ruang kelas MetPen dimana?","get_jadwal","easy")
add("A_Jadwal","Siapa dosen AI Terapan?","get_jadwal","easy")
add("A_Jadwal","Jadwal besok apa?","get_jadwal","medium")
add("A_Jadwal","Ada kuliah jam 7 pagi?","get_jadwal","medium")
add("A_Jadwal","Kuliah apa setelah jam 12?","get_jadwal","medium")
add("A_Jadwal","Ada kuliah di Lab Komputer?","get_jadwal","medium")
add("A_Jadwal","Tampilkan semua jadwal","get_jadwal","easy")
add("A_Jadwal","Tambah jadwal: Jumat, WebDev, 13:00","add_jadwal","medium")
add("A_Jadwal","Tambah: Sabtu, AI Workshop, 09:00","add_jadwal","medium")
add("A_Jadwal","Tambah: Senin, Statistika, 14:00","add_jadwal","medium")
add("A_Jadwal","Hapus jadwal hari Sabtu",None,"medium")
add("A_Jadwal","Tambah: Kamis, KKN, 07:00-17:00","add_jadwal","hard")
add("A_Jadwal","Apa jadwal terlengkap?","get_jadwal","easy")
add("A_Jadwal","Berapa kuliah di hari Rabu?","get_jadwal","medium")
add("A_Jadwal","Apakah ada jadwal bentrok?","get_jadwal","hard")
add("A_Jadwal","Hari paling sibuk?","get_jadwal","hard")
add("A_Jadwal","Kuliah Pak Ghifari?","get_jadwal","medium")
add("A_Jadwal","Apakah Jumat libur?","get_jadwal","medium")
add("A_Jadwal","Rata jam kuliah per hari?","get_jadwal","hard")
add("A_Jadwal","Cari jadwal MetPen","get_jadwal","easy")
add("A_Jadwal","Cek jadwal Selasa","get_jadwal","easy")
add("A_Jadwal","Format list jadwal","get_jadwal","medium")

# B. Tugas (25)
add("B_Tugas","Tugas apa yang belum?","get_tugas","easy")
add("B_Tugas","Deadline MetPen?","get_tugas","easy")
add("B_Tugas","Tampilkan semua tugas","get_tugas","easy")
add("B_Tugas","Tugas deadline minggu ini?","get_tugas","medium")
add("B_Tugas","Tugas paling mendesak?","get_tugas","medium")
add("B_Tugas","Berapa tugas menunggu?","get_tugas","easy")
add("B_Tugas","Tugas yang sudah selesai?","get_tugas","medium")
add("B_Tugas","Deadline AI Terapan?","get_tugas","easy")
add("B_Tugas","Deadline Basis Data?","get_tugas","easy")
add("B_Tugas","Tugas deadline paling lama?","get_tugas","medium")
add("B_Tugas","Tambah: Paper AI bab1-3 deadline 25 Juni","add_tugas","medium")
add("B_Tugas","Tambah: Review MetPen deadline 15 Juni","add_tugas","medium")
add("B_Tugas","Tambah: Project Web deadline 30 Juni","add_tugas","medium")
add("B_Tugas","Tambah: Laporan BD deadline 20 Juni","add_tugas","medium")
add("B_Tugas","Tambah: Makalah Jarkom deadline 10 Juli","add_tugas","medium")
add("B_Tugas","Tandai Basis Data selesai","update_tugas_status","medium")
add("B_Tugas","Tandai AI Terapan selesai","update_tugas_status","medium")
add("B_Tugas","Update deadline MetPen jadi 20 Juni",None,"hard")
add("B_Tugas","Hapus tugas selesai","delete_tugas","medium")
add("B_Tugas","Hapus tugas Review","delete_tugas","hard")
add("B_Tugas","Urut deadline terdekat","get_tugas","medium")
add("B_Tugas","Tugas hari ini?","get_tugas","hard")
add("B_Tugas","Ringkasan semua tugas","get_tugas","hard")
add("B_Tugas","Tugas deadline lewat?","get_tugas","hard")
add("B_Tugas","Tambah: AI Paper bab4-6 deadline 5 Juli","add_tugas","medium")

# C. Info (20)
add("C_Info","Kapan UTS?","search_akademik","easy")
add("C_Info","Kapan UAS?","search_akademik","easy")
add("C_Info","Syarat sidang skripsi?","search_akademik","medium")
add("C_Info","Aturan revisi nilai?","search_akademik","medium")
add("C_Info","Cari info beasiswa","search_akademik","easy")
add("C_Info","Info kalender akademik?","search_akademik","medium")
add("C_Info","SKS minimal sidang?","search_akademik","hard")
add("C_Info","IPK min beasiswa?","search_akademik","hard")
add("C_Info","Pendaftaran beasiswa kapan?","search_akademik","medium")
add("C_Info","Apa itu UTS dan UAS?",None,"easy")
add("C_Info","Cara daftar ulang?",None,"medium")
add("C_Info","Info cuti akademik",None,"medium")
add("C_Info","Aturan pindah jurusan?",None,"hard")
add("C_Info","Syarat cumlaude?",None,"hard")
add("C_Info","Berapa UKT?",None,"hard")
add("C_Info","Info PKL",None,"medium")
add("C_Info","Bantuan biaya kuliah?",None,"hard")
add("C_Info","Kapan pengisian KRS?",None,"hard")
add("C_Info","Syarat cuti?",None,"medium")
add("C_Info","Cara cek IPK?",None,"hard")

# D. Bebas (20)
add("D_Bebas","Rangkum jadwal dan tugasku","multi","hard")
add("D_Bebas","Apa yang harus aku kerjakan?","multi","hard")
add("D_Bebas","Buat jadwal belajar UTS","get_jadwal","hard")
add("D_Bebas","Prioritas tugas","get_tugas","medium")
add("D_Bebas","Apa saja fiturmu?",None,"easy")
add("D_Bebas","Halo selamat pagi",None,"easy")
add("D_Bebas","Siapa kamu?",None,"easy")
add("D_Bebas","Kamu bisa apa?",None,"easy")
add("D_Bebas","Ceritakan UMY",None,"medium")
add("D_Bebas","Rencana minggu ini","multi","hard")
add("D_Bebas","Perbedaan UTS dan UAS?","search_akademik","medium")
add("D_Bebas","Tips sukses kuliah?",None,"medium")
add("D_Bebas","Banding jadwal Senin Selasa","get_jadwal","hard")
add("D_Bebas","Hitung total tugas","get_tugas","medium")
add("D_Bebas","Jadwal berhubungan AI?","get_jadwal","hard")
add("D_Bebas","Puisi mahasiswa",None,"medium")
add("D_Bebas","Mock interview sidang",None,"hard")
add("D_Bebas","Apa kabar?",None,"easy")
add("D_Bebas","Terima kasih",None,"easy")
add("D_Bebas","Sampai jumpa",None,"easy")

# E. Edge (10)
add("E_Edge"," ",None,"edge")
add("E_Edge","a"*500,None,"edge")
add("E_Edge","!@#$%^&*()_+",None,"edge")
add("E_Edge","SELECT * FROM jadwal; DROP",None,"edge")
add("E_Edge","<script>alert('xss')</script>",None,"edge")
add("E_Edge","Tugas apa? Jadwal hari ini?","multi","edge")
add("E_Edge","😀🎉🌟🔥💯",None,"edge")
add("E_Edge","Info semua jadwal tugas akademik","multi","edge")
add("E_Edge","Bahas UTS AI detail",None,"edge")
add("E_Edge","Halo null",None,"edge")

print(f"Total: {len(scenarios)} scenarios")
sys.stdout.flush()

results = []
t_start = time.time()

for sc in scenarios:
    prompt = sc['prompt']
    if not prompt.strip():
        results.append({"id":sc['id'],"kategori":sc['kategori'],"prompt":repr(prompt),
            "success":False,"tool_calls":[],"latency":0,"error":"empty","response":""})
        print(f"[{sc['id']:3d}] ⏭️  EMPTY")
        sys.stdout.flush()
        continue
    
    sid_t = f"t{sc['id']}_{uuid.uuid4().hex[:4]}"
    payload = json.dumps({"message":prompt,"session_id":sid_t})
    t0 = time.time()
    try:
        r = subprocess.run(["curl","-s","-m","30","-X","POST",BASE,
            "-H","Content-Type: application/json","-d",payload],
            capture_output=True,text=True,timeout=35)
        data = json.loads(r.stdout)
        elapsed = time.time()-t0
        resp = data.get('response','')
        err = data.get('error','')
        tools = data.get('tool_calls',[])
        api_time = data.get('time',round(elapsed,2))
        success = bool(resp) and not bool(err)
        tn = [t.get('function',{}).get('name','?') for t in tools] if tools else []
        results.append({"id":sc['id'],"kategori":sc['kategori'],"prompt":prompt,
            "success":success,"tool_calls":tn,"latency":api_time,
            "error":err or "none","response":resp[:300]})
        status = '✅' if success else '❌'
        print(f"[{sc['id']:3d}] {sc['kategori'][:6]:6s} {status} tools={str(tn):25s} {api_time:5.1f}s")
    except Exception as e:
        elapsed = time.time()-t0
        results.append({"id":sc['id'],"kategori":sc['kategori'],"prompt":prompt,
            "success":False,"tool_calls":[],"latency":elapsed,
            "error":f"err:{str(e)[:80]}","response":""})
        print(f"[{sc['id']:3d}] ❌ {str(e)[:60]}")
    sys.stdout.flush()

total_s = sum(1 for r in results if r['success'])
total_t = time.time()-t_start
print(f"\n{'='*50}")
print(f"DONE: {len(results)} tests, {total_s}/{len(results)} ({total_s/len(results)*100:.1f}%)")
print(f"Time: {total_t:.0f}s ({total_t/len(results):.1f}s avg)")

for kat in ['A_Jadwal','B_Tugas','C_Info','D_Bebas','E_Edge']:
    kr = [r for r in results if r['kategori']==kat]
    if kr:
        ks = sum(1 for r in kr if r['success'])
        ke = [r for r in kr if r['error']!='none' and r['error']!='empty']
        kl = sum(r['latency'] for r in kr)/len(kr)
        print(f"  {kat:12s}: {ks}/{len(kr)} ({ks/len(kr)*100:.1f}%) errors={len(ke)} latency={kl:.2f}s")

# Error detail
errors = [r for r in results if r['error']!='none' and r['error']!='empty']
if errors:
    print(f"\nErrors ({len(errors)}):")
    for e in errors:
        print(f"  [{e['id']}] {e['kategori']}: '{e['prompt'][:40]}' → {e['error'][:80]}")

with open(OUT,'w') as f:
    json.dump({"timestamp":time.time(),"total":len(results),"success":total_s,
        "total_time":total_t,"results":results}, f, indent=2, ensure_ascii=False)
print(f"\nSaved: {OUT}")
