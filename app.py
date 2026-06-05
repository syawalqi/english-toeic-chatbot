"""
Chatbot Asisten Akademik - Flask Backend with DeepSeek V4 Flash
"""
import os
import json
import time
import uuid
from datetime import datetime

from flask import Flask, request, jsonify, render_template, session as flask_session
from openai import OpenAI
from dotenv import load_dotenv

from data import init_db, seed_data, get_db, DB_PATH
from tools import TOOLS, execute_tool

load_dotenv(dotenv_path=os.path.expanduser('~/.hermes/.env'))

app = Flask(__name__)
app.secret_key = os.urandom(24).hex()

# Ensure reasoning_content column exists in existing databases
def migrate_db():
    conn = get_db()
    c = conn.cursor()
    try:
        c.execute("ALTER TABLE messages ADD COLUMN reasoning_content TEXT")
        conn.commit()
    except:
        pass  # Column already exists
    conn.close()

migrate_db()

# DeepSeek client via OpenCode Go (OpenAI-compatible)
client = OpenAI(
    api_key=os.getenv('OPENCODE_GO_API_KEY'),
    base_url='https://opencode.ai/zen/go/v1'
)

MODEL = 'deepseek-v4-flash'

# Today's context
TODAY_INFO = f"Hari ini: Jumat, 5 Juni 2026. Semester gasal 2025/2026."

SYSTEM_PROMPT = f"""Kamu adalah asisten akademik untuk mahasiswa Universitas Muhammadiyah Yogyakarta (UMY).
{TODAY_INFO}

Informasi hari ini penting untuk merespon query "hari ini" dengan benar.

Tugasmu membantu mahasiswa mengelola:
1. Jadwal kuliah harian
2. Deadline tugas
3. Informasi akademik (UTS, UAS, aturan, dll)

Gunakan tools yang tersedia untuk membaca dan menulis data.
Jika user menyebut hari tanpa konteks, gunakan hari ini sebagai referensi.

Format jawaban:
- Gunakan Bahasa Indonesia yang natural dan informatif
- Gunakan emoji secukupnya untuk membuat chat lebih hidup
- Jika menampilkan daftar, gunakan format bullet point
- Jangan gunakan markdown sintaks kompleks (seperti ```, #, *)"""

# Store active sessions in memory (simplified)
sessions = {}

def get_or_create_session(session_id):
    if session_id not in sessions:
        sessions[session_id] = []
    return sessions[session_id]

def save_message(session_id, role, content, tool_calls=None, reasoning_content=None):
    conn = get_db()
    c = conn.cursor()
    c.execute(
        "INSERT INTO messages (session_id, role, content, tool_calls, reasoning_content) VALUES (?,?,?,?,?)",
        (session_id, role, content, json.dumps(tool_calls) if tool_calls else None, reasoning_content)
    )
    conn.commit()
    conn.close()

def get_history(session_id, limit=20):
    conn = get_db()
    c = conn.cursor()
    c.execute(
        "SELECT role, content, tool_calls, reasoning_content FROM messages WHERE session_id=? ORDER BY id ASC LIMIT ?",
        (session_id, limit)
    )
    rows = c.fetchall()
    conn.close()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for row in rows:
        role = row["role"]
        content = row["content"]
        tc = row["tool_calls"]
        rc = row["reasoning_content"]
        if tc:
            tc_parsed = json.loads(tc)
        else:
            tc_parsed = None

        if role == "tool":
            messages.append({"role": "tool", "tool_call_id": tc_parsed[0]["tool_call_id"] if tc_parsed else "", "content": content})
        elif role == "assistant":
            msg = {"role": "assistant", "content": content or None}
            if tc_parsed:
                msg["tool_calls"] = tc_parsed
            if rc:
                msg["reasoning_content"] = rc
            messages.append(msg)
        else:
            messages.append({"role": role, "content": content})
    return messages

# ============================================================
# API Routes
# ============================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '').strip()
    session_id = data.get('session_id', '')

    if not session_id:
        session_id = str(uuid.uuid4())

    if not user_message:
        return jsonify({'error': 'Pesan tidak boleh kosong', 'session_id': session_id}), 400

    # Save user message
    conn = get_db()
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO conversations (session_id) VALUES (?)", (session_id,))
    conn.commit()
    conn.close()
    save_message(session_id, "user", user_message)

    # Build message history
    messages = get_history(session_id)
    # Add the current user message (not yet saved in history)
    messages.append({"role": "user", "content": user_message})

    start_time = time.time()

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.3,
            max_tokens=2000,
        )

        elapsed = time.time() - start_time
        assistant_message = response.choices[0].message
        raw_content = assistant_message.content or ""
        reasoning_raw = getattr(assistant_message, 'reasoning_content', None)
        # DeepSeek V4 Flash returns content="" and actual text in reasoning_content
        assistant_content = raw_content or reasoning_raw or ""
        tool_calls = assistant_message.tool_calls
        reasoning_content = reasoning_raw

        saved_tool_calls = []
        saved_tool_results = []

        # Handle tool calls
        if tool_calls:
            # Build combined tool_calls array (all in ONE assistant message)
            combined_tool_calls = []
            for tc in tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                result = execute_tool(fn_name, fn_args)

                tc_dict = {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": fn_name, "arguments": tc.function.arguments}
                }
                combined_tool_calls.append(tc_dict)
                saved_tool_calls.append(tc_dict)
                saved_tool_results.append({
                    "tool_call_id": tc.id,
                    "result": result
                })

            # Add ONE assistant message with ALL tool calls
            tool_call_msg = {
                "role": "assistant",
                "content": None,
                "tool_calls": combined_tool_calls
            }
            if reasoning_content:
                tool_call_msg["reasoning_content"] = reasoning_content
            messages.append(tool_call_msg)

            # Add tool result messages (one per tool_call)
            for tr in saved_tool_results:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tr["tool_call_id"],
                    "content": tr["result"]
                })

            # Save to DB for history continuity
            conn = get_db()
            c = conn.cursor()
            c.execute(
                "INSERT INTO messages (session_id, role, content, tool_calls, reasoning_content) VALUES (?,?,?,?,?)",
                (session_id, "assistant", None, json.dumps(saved_tool_calls), reasoning_content)
            )
            for tr in saved_tool_results:
                c.execute(
                    "INSERT INTO messages (session_id, role, content, tool_calls) VALUES (?,?,?,?)",
                    (session_id, "tool", tr["result"], json.dumps([{"tool_call_id": tr["tool_call_id"]}]))
                )
            conn.commit()
            conn.close()
            
            # Second LLM call to synthesize answer
            response2 = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=2000,
            )
            final_content_raw = response2.choices[0].message.content or ""
            final_reasoning = getattr(response2.choices[0].message, 'reasoning_content', None)
            final_content = final_content_raw or final_reasoning or ""

            # Save final assistant response
            save_message(session_id, "assistant", final_content, reasoning_content=final_reasoning)
            elapsed = time.time() - start_time

            return jsonify({
                'response': final_content,
                'session_id': session_id,
                'tool_calls': saved_tool_calls,
                'time': round(elapsed, 2)
            })

        # No tool calls - direct response
        save_message(session_id, "assistant", assistant_content, reasoning_content=reasoning_content)
        elapsed = time.time() - start_time

        return jsonify({
            'response': assistant_content,
            'session_id': session_id,
            'tool_calls': [],
            'time': round(elapsed, 2)
        })

    except Exception as e:
        elapsed = time.time() - start_time
        return jsonify({
            'response': f"⚠️ Maaf, terjadi error: {str(e)}",
            'session_id': session_id,
            'error': str(e),
            'time': round(elapsed, 2)
        }), 500

@app.route('/api/reset', methods=['POST'])
def reset():
    session_id = request.json.get('session_id', '')
    if session_id:
        conn = get_db()
        c = conn.cursor()
        c.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
        conn.commit()
        conn.close()
    else:
        session_id = str(uuid.uuid4())
    return jsonify({'session_id': session_id, 'message': 'Chat direset'})

@app.route('/api/history', methods=['GET'])
def get_chat_history():
    session_id = request.args.get('session_id', '')
    if not session_id:
        return jsonify({'messages': []})
    
    conn = get_db()
    c = conn.cursor()
    c.execute(
        "SELECT role, content FROM messages WHERE session_id=? AND role != 'system' ORDER BY id ASC LIMIT 50",
        (session_id,)
    )
    rows = []
    for row in c.fetchall():
        rows.append({
            'role': row['role'],
            'text': row['content']
        })
    conn.close()
    return jsonify({'messages': rows})

@app.route('/api/test/record', methods=['POST'])
def record_test():
    """Record a test result."""
    data = request.json
    conn = get_db()
    c = conn.cursor()
    c.execute(
        "INSERT INTO test_results (scenario, category, success, response_time, error_type, notes) VALUES (?,?,?,?,?,?)",
        (data['scenario'], data['category'], data['success'],
         data.get('response_time'), data.get('error_type'), data.get('notes'))
    )
    conn.commit()
    conn.close()
    return jsonify({'success': True})

@app.route('/api/test/results', methods=['GET'])
def get_test_results():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM test_results ORDER BY id ASC")
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return jsonify(rows)

# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    init_db()
    seed_data()
    port = int(os.getenv('PORT', 9120))
    print(f"🤖 Chatbot Akademik running on http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
