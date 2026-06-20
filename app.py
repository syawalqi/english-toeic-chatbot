"""
TOEIC Reading Practice — Flask Backend with DeepSeek / OpenAI
"""
import os
import json
import time
import uuid
import random
from datetime import datetime

from flask import Flask, request, jsonify, render_template
from openai import OpenAI
from dotenv import load_dotenv

from data import init_db, get_db

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
load_dotenv(dotenv_path=os.path.expanduser('~/.hermes/.env'), override=False)

app = Flask(__name__)
app.secret_key = os.urandom(24).hex()

client = OpenAI(
    api_key=os.getenv('OPENCODE_GO_API_KEY'),
    base_url='https://opencode.ai/zen/go/v1'
)

MODEL = 'deepseek-v4-flash'
TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', '0.7'))

CHAT_SYSTEM_PROMPT = """You are a TOEIC Reading practice assistant helping Indonesian university students prepare for the TOEIC Reading section. Always respond in English.

You can:
1. Explain how the Reading Test tab works
2. Discuss test results — explain why answers are correct or incorrect
3. Give TOEIC strategies — time management, reading techniques, question types
4. Teach vocabulary and grammar from passages
5. Answer general questions about TOEIC

Be encouraging, clear, and patient. These are English learners. When a student mentions their score, congratulate them first, then offer to explain specific questions."""

TEST_GENERATION_PROMPT = """You are a TOEIC Reading test generator. Create a unique test.

DIFFICULTY: {difficulty}
SEED: {seed}

Generate 2-3 short business-related reading passages and exactly 10 comprehension questions.

DIFFICULTY GUIDELINES:
- EASY: Short passages (50-80 words), simple vocabulary, direct facts tested
- MEDIUM: Moderate passages (80-120 words), intermediate vocabulary, some inference
- HARD: Longer passages (100-150 words), advanced vocabulary, multiple inference types

PASSAGE TOPICS (choose 2-3 randomly):
- Business emails and memos
- Product advertisements and promotions
- Travel notices and policies
- Office announcements and memos
- Meeting schedules and agendas
- Customer service messages
- Company news and updates

Each question must have 4 options (A, B, C, D) with exactly one correct answer.
Include a brief explanation for each question.

Return ONLY valid JSON with no markdown, no code fences, no extra text:

{{"passages":[{{"id":1,"title":"...","text":"..."}}],"questions":[{{"id":1,"passage_id":1,"stem":"...","options":["A. ...","B. ...","C. ...","D. ..."],"correct_answer":"A","explanation":"..."}}]}}"""

# ── Fallback test data ──────────────────────────────────────────

FALLBACK_PASSAGES = [
    {
        "id": 1,
        "title": "Office Relocation Announcement",
        "text": "To: All Employees\nFrom: Sarah Chen, HR Manager\nSubject: Office Relocation Update\n\nDear Team,\n\nI am pleased to announce that the relocation to our new office building at 45 Business Park Drive will take place next month. The move is scheduled for the weekend of September 15-16.\n\nAll employees should pack their personal belongings by September 14. The IT department will handle the packing and transfer of all office equipment. New desk assignments will be emailed to everyone by September 10.\n\nA welcome breakfast will be held on Monday, September 17 at 8:30 AM in the new building's cafeteria.\n\nBest regards,\nSarah"
    },
    {
        "id": 2,
        "title": "FitLife Pro Fitness App",
        "text": "Are you tired of complicated workout plans that do not fit your schedule? FitLife Pro is the solution. Our app creates personalized 15-minute workouts based on your fitness level, available equipment, and time preferences.\n\nKey Features:\n- AI-powered workout customization\n- Video demonstrations for every exercise\n- Progress tracking with detailed analytics\n- Integration with popular health devices\n- Offline mode for workouts anywhere\n\nDownload now and get your first month free! Available on iOS and Android."
    },
    {
        "id": 3,
        "title": "Flight Schedule Changes",
        "text": "Attention Passengers,\n\nDue to scheduled maintenance at Jakarta International Airport, the following flight changes will take effect from November 1:\n\nFlight GA-207 to Singapore: Departure changed from 14:30 to 16:45\nFlight GA-208 to Tokyo: Departure changed from 08:15 to 10:30\nFlight GA-209 to Sydney: Departure changed from 22:00 to 23:30\n\nAll affected passengers will be contacted via email and SMS. Passengers with connecting flights are advised to contact our customer service center at 1-800-555-0199 for rebooking assistance.\n\nWe apologize for any inconvenience caused.\nGaruda Airways Customer Service"
    }
]

FALLBACK_QUESTIONS = [
    {"id": 1, "passage_id": 1, "stem": "What is the main purpose of this email?", "options": ["A. To introduce a new employee", "B. To announce an office relocation", "C. To request vacation approval", "D. To report a maintenance issue"], "correct_answer": "B", "explanation": "The email announces the relocation to a new office building at 45 Business Park Drive."},
    {"id": 2, "passage_id": 1, "stem": "When should employees pack their personal belongings?", "options": ["A. By September 10", "B. By September 17", "C. By September 14", "D. By September 15"], "correct_answer": "C", "explanation": "The email states: 'All employees should pack their personal belongings by September 14.'"},
    {"id": 3, "passage_id": 1, "stem": "Who is responsible for moving office equipment?", "options": ["A. The employees", "B. The IT department", "C. An external moving company", "D. The HR manager"], "correct_answer": "B", "explanation": "The IT department will handle the packing and transfer of all office equipment."},
    {"id": 4, "passage_id": 1, "stem": "What will happen on September 10?", "options": ["A. The office move will begin", "B. A welcome breakfast will be held", "C. New desk assignments will be sent", "D. The IT department will start packing"], "correct_answer": "C", "explanation": "New desk assignments will be emailed to everyone by September 10."},
    {"id": 5, "passage_id": 2, "stem": "What is FitLife Pro?", "options": ["A. A gym membership service", "B. A fitness mobile application", "C. A line of sports equipment", "D. A health food delivery service"], "correct_answer": "B", "explanation": "The passage describes FitLife Pro as an app that creates personalized workouts."},
    {"id": 6, "passage_id": 2, "stem": "Which of the following is mentioned as a key feature?", "options": ["A. Meal planning", "B. Live coaching sessions", "C. Progress tracking", "D. Group fitness classes"], "correct_answer": "C", "explanation": "Progress tracking with detailed analytics is listed as one of the key features."},
    {"id": 7, "passage_id": 2, "stem": "How can new users try the app?", "options": ["A. One month free trial", "B. Free forever plan", "C. Seven-day trial", "D. Pay-per-workout model"], "correct_answer": "A", "explanation": "The ad states: 'Download now and get your first month free!'"},
    {"id": 8, "passage_id": 3, "stem": "Why are the flights being rescheduled?", "options": ["A. Bad weather conditions", "B. Staff shortage", "C. Airport maintenance", "D. Security concerns"], "correct_answer": "C", "explanation": "Due to scheduled maintenance at Jakarta International Airport, the flight changes will take effect."},
    {"id": 9, "passage_id": 3, "stem": "What should passengers with connecting flights do?", "options": ["A. Wait for email notification", "B. Go to the airport immediately", "C. Book a new flight online", "D. Call customer service"], "correct_answer": "D", "explanation": "Passengers with connecting flights should contact customer service at 1-800-555-0199 for rebooking assistance."},
    {"id": 10, "passage_id": 3, "stem": "How will affected passengers be notified?", "options": ["A. By email and SMS", "B. By phone call only", "C. By postal mail", "D. Through the airport website"], "correct_answer": "A", "explanation": "All affected passengers will be contacted via email and SMS."}
]

# ── Helpers ─────────────────────────────────────────────────────

def save_message(session_id, role, content):
    conn = get_db()
    c = conn.cursor()
    c.execute(
        "INSERT INTO messages (session_id, role, content) VALUES (?,?,?)",
        (session_id, role, content)
    )
    conn.commit()
    conn.close()


def get_history(session_id, limit=20):
    conn = get_db()
    c = conn.cursor()
    c.execute(
        "SELECT role, content FROM messages WHERE session_id=? ORDER BY id ASC LIMIT ?",
        (session_id, limit)
    )
    rows = c.fetchall()
    conn.close()
    messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
    for row in rows:
        messages.append({"role": row["role"], "content": row["content"]})
    return messages


def parse_llm_json(content):
    content = content.strip()
    if content.startswith('```'):
        lines = content.split('\n', 1)
        if len(lines) > 1:
            content = lines[1]
        else:
            content = ''
    if content.endswith('```'):
        content = content.rsplit('```', 1)[0]
    content = content.strip()
    if content.startswith('json'):
        content = content[4:].strip()
    return json.loads(content)


def generate_test_via_llm(difficulty, seed):
    prompt = TEST_GENERATION_PROMPT.format(difficulty=difficulty, seed=seed)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        max_tokens=3000,
    )
    raw = response.choices[0].message.content
    data = parse_llm_json(raw)
    passages = data.get('passages', FALLBACK_PASSAGES)
    questions = data.get('questions', FALLBACK_QUESTIONS)
    if len(questions) != 10:
        questions = FALLBACK_QUESTIONS
    if not passages:
        passages = FALLBACK_PASSAGES
    return passages, questions


def strip_answer_key(questions):
    result = []
    for q in questions:
        result.append({
            "id": q["id"],
            "passage_id": q["passage_id"],
            "stem": q["stem"],
            "options": q["options"]
        })
    return result


# ── Routes ──────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/register', methods=['POST'])
def register():
    data = request.json or {}
    name = data.get('name', '').strip()
    nim = data.get('nim', '').strip()
    if not name or not nim:
        return jsonify({'error': 'Name and NIM are required.'}), 400

    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT id, name, nim FROM students WHERE nim=?", (nim,))
    existing = c.fetchone()
    if existing:
        if existing['name'].lower() != name.lower():
            conn.close()
            return jsonify({'error': 'NIM already registered with a different name.'}), 409
        conn.close()
        return jsonify({'student_id': existing['id'], 'name': existing['name'], 'nim': existing['nim']})

    c.execute("INSERT INTO students (name, nim) VALUES (?,?)", (name, nim))
    conn.commit()
    student_id = c.lastrowid
    conn.close()
    return jsonify({'student_id': student_id, 'name': name, 'nim': nim}), 201


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json or {}
    user_message = data.get('message', '').strip()
    session_id = data.get('session_id', '')

    if not session_id:
        session_id = str(uuid.uuid4())
    if not user_message:
        return jsonify({'response': 'Please enter a message.', 'session_id': session_id}), 400

    conn = get_db()
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO conversations (session_id) VALUES (?)", (session_id,))
    conn.commit()
    conn.close()
    save_message(session_id, 'user', user_message)

    messages = get_history(session_id)
    messages.append({"role": "user", "content": user_message})

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=1500,
        )
        assistant_content = response.choices[0].message.content or ''
        save_message(session_id, 'assistant', assistant_content)
        return jsonify({'response': assistant_content, 'session_id': session_id})
    except Exception as e:
        return jsonify({'response': f'Sorry, an error occurred: {str(e)}', 'session_id': session_id, 'error': str(e)}), 500


@app.route('/api/reset', methods=['POST'])
def reset():
    session_id = (request.json or {}).get('session_id', '')
    if session_id:
        conn = get_db()
        c = conn.cursor()
        c.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
        conn.commit()
        conn.close()
    return jsonify({'session_id': session_id, 'message': 'Chat reset.'})


@app.route('/api/test/generate', methods=['POST'])
def generate_test():
    data = request.json or {}
    student_id = data.get('student_id')
    difficulty = data.get('difficulty', 'medium')

    if not student_id:
        return jsonify({'error': 'Student ID is required.'}), 400
    if difficulty not in ('easy', 'medium', 'hard'):
        return jsonify({'error': 'Invalid difficulty. Use easy, medium, or hard.'}), 400

    test_id = None
    passages = FALLBACK_PASSAGES
    questions = FALLBACK_QUESTIONS

    for attempt in range(3):
        seed = f"{student_id}-{int(time.time() * 1000)}-{random.randint(0, 9999)}"
        try:
            passages, questions = generate_test_via_llm(difficulty, seed)
        except Exception:
            seed = 'fallback'

        conn = get_db()
        c = conn.cursor()
        try:
            c.execute(
                "INSERT INTO reading_tests (student_id, seed, difficulty, passages, questions) VALUES (?,?,?,?,?)",
                (student_id, seed, difficulty, json.dumps(passages, ensure_ascii=False),
                 json.dumps(questions, ensure_ascii=False))
            )
            conn.commit()
            test_id = c.lastrowid
            conn.close()
            break
        except Exception:
            conn.close()
            if attempt == 2:
                return jsonify({'error': 'Failed to generate test. Please try again.'}), 500

    return jsonify({
        'test_id': test_id,
        'passages': passages,
        'questions': strip_answer_key(questions)
    })


@app.route('/api/test/submit', methods=['POST'])
def submit_test():
    data = request.json or {}
    student_id = data.get('student_id')
    test_id = data.get('test_id')
    user_answers = data.get('answers', [])

    if not student_id or not test_id:
        return jsonify({'error': 'Student ID and Test ID are required.'}), 400

    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT passages, questions FROM reading_tests WHERE id=?", (test_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return jsonify({'error': 'Test not found.'}), 404

    stored_questions = json.loads(row['questions'])
    answer_key = {q['id']: q['correct_answer'] for q in stored_questions}
    explanations = {q['id']: q.get('explanation', '') for q in stored_questions}
    stems = {q['id']: q['stem'] for q in stored_questions}

    user_map = {a.get('question_id'): a.get('answer', '') for a in user_answers}
    total = len(stored_questions)
    correct = 0
    results = []

    for q in stored_questions:
        qid = q['id']
        ua = user_map.get(qid, '')
        ca = answer_key[qid]
        is_correct = ua.upper() == ca.upper()
        if is_correct:
            correct += 1
        results.append({
            'question_id': qid,
            'stem': stems[qid],
            'user_answer': ua,
            'correct_answer': ca,
            'correct': is_correct,
            'explanation': explanations.get(qid, '')
        })

    c.execute(
        "INSERT INTO test_attempts (student_id, test_id, answers, raw_score, total_questions) VALUES (?,?,?,?,?)",
        (student_id, test_id, json.dumps(user_answers), correct, total)
    )
    conn.commit()
    conn.close()

    return jsonify({'score': correct, 'total': total, 'results': results})


@app.route('/api/history', methods=['GET'])
def get_history_api():
    student_id = request.args.get('student_id')
    if not student_id:
        return jsonify({'error': 'Student ID is required.'}), 400

    conn = get_db()
    c = conn.cursor()
    c.execute("""
        SELECT ta.id, ta.raw_score, ta.total_questions, ta.completed_at, rt.difficulty
        FROM test_attempts ta
        JOIN reading_tests rt ON ta.test_id = rt.id
        WHERE ta.student_id = ?
        ORDER BY ta.completed_at DESC
    """, (student_id,))
    rows = c.fetchall()
    conn.close()

    attempts = [{
        'attempt_id': r['id'],
        'score': r['raw_score'],
        'total': r['total_questions'],
        'difficulty': r['difficulty'],
        'date': r['completed_at']
    } for r in rows]

    return jsonify({'attempts': attempts})


@app.route('/api/history/chat', methods=['GET'])
def get_chat_history():
    session_id = request.args.get('session_id', '')
    if not session_id:
        return jsonify({'messages': []})
    conn = get_db()
    c = conn.cursor()
    c.execute(
        "SELECT role, content FROM messages WHERE session_id=? ORDER BY id ASC LIMIT 50",
        (session_id,)
    )
    rows = [{'role': r['role'], 'text': r['content']} for r in c.fetchall()]
    conn.close()
    return jsonify({'messages': rows})


@app.route('/api/test/result/<int:test_id>', methods=['GET'])
def get_test_result(test_id):
    student_id = request.args.get('student_id')
    if not student_id:
        return jsonify({'error': 'Student ID is required.'}), 400

    conn = get_db()
    c = conn.cursor()

    c.execute("SELECT passages, questions FROM reading_tests WHERE id=? AND student_id=?",
              (test_id, student_id))
    test_row = c.fetchone()
    if not test_row:
        conn.close()
        return jsonify({'error': 'Test not found.'}), 404

    c.execute("""SELECT answers, raw_score, total_questions, completed_at
                 FROM test_attempts WHERE test_id=? AND student_id=?
                 ORDER BY completed_at DESC LIMIT 1""", (test_id, student_id))
    attempt_row = c.fetchone()
    conn.close()

    passages = json.loads(test_row['passages'])
    questions = json.loads(test_row['questions'])

    if attempt_row:
        user_answers = json.loads(attempt_row['answers'])
        user_map = {a.get('question_id'): a.get('answer', '') for a in user_answers}
    else:
        user_map = {}

    answer_key = {q['id']: q['correct_answer'] for q in questions}
    explanations = {q['id']: q.get('explanation', '') for q in questions}

    results = []
    for q in questions:
        qid = q['id']
        ua = user_map.get(qid, '')
        ca = answer_key[qid]
        results.append({
            'question_id': qid,
            'passage_id': q['passage_id'],
            'stem': q['stem'],
            'options': q['options'],
            'user_answer': ua,
            'correct_answer': ca,
            'correct': ua.upper() == ca.upper(),
            'explanation': explanations.get(qid, '')
        })

    return jsonify({
        'test_id': test_id,
        'passages': passages,
        'results': results,
        'score': attempt_row['raw_score'] if attempt_row else None,
        'total': attempt_row['total_questions'] if attempt_row else len(questions),
        'completed_at': attempt_row['completed_at'] if attempt_row else None
    })


# ── Main ────────────────────────────────────────────────────────

if __name__ == '__main__':
    init_db()
    port = int(os.getenv('PORT', 9120))
    print(f"TOEIC Reading Practice running on http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
