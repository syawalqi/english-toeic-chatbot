"""
Tool definitions for chatbot akademik.
Each tool has a schema (for LLM function calling) and an execute function.
"""
from data import get_db
import json

# ============================================================
# TOOL SCHEMAS (for OpenAI/DeepSeek function calling)
# ============================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_jadwal",
            "description": "Cari jadwal kuliah berdasarkan hari atau mata kuliah. Kosongkan parameter untuk semua jadwal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "hari": {
                        "type": "string",
                        "description": "Nama hari (Senin, Selasa, Rabu, Kamis, Jumat). Kosongkan jika tidak filter.",
                        "default": ""
                    },
                    "mata_kuliah": {
                        "type": "string",
                        "description": "Nama mata kuliah. Kosongkan jika tidak filter.",
                        "default": ""
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_tugas",
            "description": "Cari tugas berdasarkan mata kuliah atau status. Kosongkan parameter untuk semua tugas.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mata_kuliah": {
                        "type": "string",
                        "description": "Nama mata kuliah. Kosongkan jika semua.",
                        "default": ""
                    },
                    "status": {
                        "type": "string",
                        "enum": ["menunggu", "selesai", ""],
                        "description": "Status tugas. Kosongkan untuk semua.",
                        "default": ""
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_tugas",
            "description": "Tambah tugas baru ke dalam daftar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mata_kuliah": {"type": "string", "description": "Nama mata kuliah"},
                    "deskripsi": {"type": "string", "description": "Deskripsi tugas"},
                    "deadline": {"type": "string", "description": "Deadline tugas (format: YYYY-MM-DD)"}
                },
                "required": ["mata_kuliah", "deskripsi", "deadline"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_tugas_status",
            "description": "Update status tugas (selesai / menunggu).",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "description": "ID tugas"},
                    "status": {"type": "string", "enum": ["menunggu", "selesai"], "description": "Status baru"}
                },
                "required": ["id", "status"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_tugas",
            "description": "Hapus tugas dari daftar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "description": "ID tugas yang akan dihapus"}
                },
                "required": ["id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_jadwal",
            "description": "Tambah jadwal kuliah baru.",
            "parameters": {
                "type": "object",
                "properties": {
                    "hari": {"type": "string", "description": "Nama hari"},
                    "mata_kuliah": {"type": "string", "description": "Nama mata kuliah"},
                    "jam_mulai": {"type": "string", "description": "Jam mulai (format: HH:MM)"},
                    "jam_selesai": {"type": "string", "description": "Jam selesai (format: HH:MM)"},
                    "ruang": {"type": "string", "description": "Ruang kuliah"},
                    "dosen": {"type": "string", "description": "Nama dosen"}
                },
                "required": ["hari", "mata_kuliah", "jam_mulai", "jam_selesai"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_jadwal",
            "description": "Hapus jadwal kuliah berdasarkan ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "description": "ID jadwal"}
                },
                "required": ["id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_akademik",
            "description": "Cari informasi akademik seperti jadwal UTS, aturan, syarat, dll.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Kata kunci pencarian"}
                },
                "required": ["query"]
            }
        }
    }
]

# ============================================================
# TOOL EXECUTORS
# ============================================================

def execute_tool(name, args):
    """Execute a tool by name with given args."""
    handler = TOOL_HANDLERS.get(name)
    if not handler:
        return json.dumps({"error": f"Tool '{name}' not found"})
    try:
        result = handler(**args)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

def tool_get_jadwal(hari="", mata_kuliah=""):
    conn = get_db()
    c = conn.cursor()
    query = "SELECT * FROM jadwal WHERE 1=1"
    params = []
    if hari:
        query += " AND hari LIKE ?"
        params.append(f"%{hari}%")
    if mata_kuliah:
        query += " AND mata_kuliah LIKE ?"
        params.append(f"%{mata_kuliah}%")
    query += " ORDER BY hari, jam_mulai"
    c.execute(query, params)
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows

def tool_get_tugas(mata_kuliah="", status=""):
    conn = get_db()
    c = conn.cursor()
    query = "SELECT * FROM tugas WHERE 1=1"
    params = []
    if mata_kuliah:
        query += " AND mata_kuliah LIKE ?"
        params.append(f"%{mata_kuliah}%")
    if status:
        query += " AND status = ?"
        params.append(status)
    query += " ORDER BY deadline ASC"
    c.execute(query, params)
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows

def tool_add_tugas(mata_kuliah, deskripsi, deadline):
    conn = get_db()
    c = conn.cursor()
    c.execute(
        "INSERT INTO tugas (mata_kuliah, deskripsi, deadline) VALUES (?,?,?)",
        (mata_kuliah, deskripsi, deadline)
    )
    conn.commit()
    new_id = c.lastrowid
    conn.close()
    return {"success": True, "id": new_id, "message": f"Tugas '{deskripsi}' berhasil ditambahkan"}

def tool_update_tugas_status(id, status):
    conn = get_db()
    c = conn.cursor()
    c.execute("UPDATE tugas SET status=? WHERE id=?", (status, id))
    conn.commit()
    affected = c.rowcount
    conn.close()
    if affected:
        return {"success": True, "message": f"Tugas ID {id} diupdate ke status '{status}'"}
    return {"success": False, "message": "Tugas tidak ditemukan"}

def tool_delete_tugas(id):
    conn = get_db()
    c = conn.cursor()
    c.execute("DELETE FROM tugas WHERE id=?", (id,))
    conn.commit()
    affected = c.rowcount
    conn.close()
    if affected:
        return {"success": True, "message": f"Tugas ID {id} berhasil dihapus"}
    return {"success": False, "message": "Tugas tidak ditemukan"}

def tool_add_jadwal(hari, mata_kuliah, jam_mulai, jam_selesai, ruang="", dosen=""):
    conn = get_db()
    c = conn.cursor()
    c.execute(
        "INSERT INTO jadwal (hari, mata_kuliah, jam_mulai, jam_selesai, ruang, dosen) VALUES (?,?,?,?,?,?)",
        (hari, mata_kuliah, jam_mulai, jam_selesai, ruang, dosen)
    )
    conn.commit()
    new_id = c.lastrowid
    conn.close()
    return {"success": True, "id": new_id, "message": f"Jadwal {mata_kuliah} berhasil ditambahkan"}

def tool_delete_jadwal(id):
    conn = get_db()
    c = conn.cursor()
    c.execute("DELETE FROM jadwal WHERE id=?", (id,))
    conn.commit()
    affected = c.rowcount
    conn.close()
    if affected:
        return {"success": True, "message": f"Jadwal ID {id} berhasil dihapus"}
    return {"success": False, "message": "Jadwal tidak ditemukan"}

def tool_search_akademik(query):
    conn = get_db()
    c = conn.cursor()
    c.execute(
        "SELECT * FROM info_akademik WHERE judul LIKE ? OR isi LIKE ? LIMIT 5",
        (f"%{query}%", f"%{query}%")
    )
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows

TOOL_HANDLERS = {
    "get_jadwal": tool_get_jadwal,
    "get_tugas": tool_get_tugas,
    "add_tugas": tool_add_tugas,
    "update_tugas_status": tool_update_tugas_status,
    "delete_tugas": tool_delete_tugas,
    "add_jadwal": tool_add_jadwal,
    "delete_jadwal": tool_delete_jadwal,
    "search_akademik": tool_search_akademik,
}
