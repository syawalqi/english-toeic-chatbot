"""
TOEIC Reading Practice — Flask Backend with DeepSeek / OpenAI
"""
import os
import json
import time
import uuid
import random
from datetime import datetime

from flask import Flask, request, jsonify, render_template, session
from openai import OpenAI
from dotenv import load_dotenv

from data import init_db, get_db

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
load_dotenv(dotenv_path=os.path.expanduser('~/.hermes/.env'), override=False)

app = Flask(__name__)
app.secret_key = os.urandom(24).hex()

client = OpenAI(
    api_key=os.getenv('OPENCODE_GO_API_KEY'),
    base_url='https://opencode.ai/zen/go/v1',
    timeout=300.0,
    max_retries=2,
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

You have access to tools that let you look up a student's test history and detailed results, and provide TOEIC tips. When a student talks about their test performance, use the available tools to fetch their actual data.

Be encouraging, clear, and patient. These are English learners. When a student mentions their score, congratulate them first, then offer to explain specific questions.

IMPORTANT: When asked a question that requires data, use your tools! Do not make up student data — call get_student_test_history to fetch real data from the database. Always respond in markdown format — use **bold**, *italic*, `code`, lists, headings, and other markdown formatting to make your answers clear and well-structured."""

# ── LLM Tools (Function Calling) ─────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_student_test_history",
            "description": "Get the test history and scores for a student. Use this when a student asks about their past test results, performance, or scores.",
            "parameters": {
                "type": "object",
                "properties": {
                    "student_id": {
                        "type": "integer",
                        "description": "The student's database ID number"
                    }
                },
                "required": ["student_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_test_attempt_details",
            "description": "Get detailed results of a specific test attempt, including all passages, questions, the student's answers, correct answers, and explanations. Use this after getting a student's test history to dive into a specific attempt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "attempt_id": {
                        "type": "integer",
                        "description": "The test attempt ID to look up"
                    },
                    "student_id": {
                        "type": "integer",
                        "description": "The student's database ID number"
                    }
                },
                "required": ["attempt_id", "student_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_toeic_tip",
            "description": "Get a TOEIC Reading strategy tip on a specific topic to share with the student.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "enum": ["time_management", "skimming", "scanning", "trick_questions", "vocabulary", "grammar", "reading_comprehension"],
                        "description": "The TOEIC strategy topic"
                    }
                },
                "required": ["topic"]
            }
        }
    }
]


TOEIC_TIPS = {
    "time_management": (
        "**⏱️ Time Management for TOEIC Reading**\n\n"
        "The Reading section has 100 questions in 75 minutes — that's about 45 seconds per question. Here's how to pace yourself:\n\n"
        "- **Part 5 (Incomplete Sentences):** Spend max 15-20 seconds per question. If you don't know, guess and move on.\n"
        "- **Part 6 (Text Completion):** Spend about 8-10 minutes for this section. Read the text first, then answer.\n"
        "- **Part 7 (Reading Comprehension):** Spend the remaining time. Single passages: 5-7 min each. Double passages: 8-10 min.\n"
        "- **Don't dwell on hard questions.** Mark them and come back if time permits.\n"
        "- **Leave 2-3 minutes at the end** to check your answer sheet for stray marks."
    ),
    "skimming": (
        "**👁️ Skimming Technique**\n\n"
        "Skimming is reading quickly to get the main idea. For TOEIC Reading:\n\n"
        "1. **Read the first and last sentence** of each paragraph — they usually contain the main point.\n"
        "2. **Look for signal words** like 'however', 'therefore', 'firstly', 'in conclusion'.\n"
        "3. **Pay attention to headings, titles, and bold text** — they tell you what's important.\n"
        "4. **Skip detailed examples** on the first read. You can come back if a question asks about them.\n"
        "5. **Ask yourself:** 'What is this passage mainly about?' If you can answer that, you've skimmed successfully."
    ),
    "scanning": (
        "**🔍 Scanning Technique**\n\n"
        "Scanning is searching for specific information. Unlike skimming, you know what you're looking for:\n\n"
        "1. **Read the question first** — identify keywords (names, dates, numbers, locations).\n"
        "2. **Let your eyes float across the text** looking for those keywords.\n"
        "3. **When you find a match**, read the surrounding sentences carefully.\n"
        "4. **Use text features** — bold, italics, bullet points, and numbered lists stand out.\n"
        "5. **Practice with the clock:** Try finding 3 specific facts in a passage in under 2 minutes."
    ),
    "trick_questions": (
        "**🎯 Common TOEIC Trick Questions**\n\n"
        "Watch out for these traps:\n\n"
        "1. **'All of the above' / 'None of the above'** — usually wrong unless every option truly works.\n"
        "2. **'Always' / 'Never' / 'Must'** — extreme words are rarely correct. TOEIC prefers moderate language.\n"
        "3. **Similar-sounding words** — the test uses words that *look like* the correct answer but mean something different.\n"
        "4. **'According to the passage'** — the answer MUST be explicitly stated, even if you know it's factually wrong.\n"
        "5. **Partially correct answers** — two options may both seem right, but one has a small error. Read every word.\n"
        "6. **'NOT' / 'EXCEPT' questions** — circle these words! Students pick the true statement instead of the exception."
    ),
    "vocabulary": (
        "**📚 Building TOEIC Vocabulary**\n\n"
        "TOEIC uses business English. Focus on these areas:\n\n"
        "- **Office & HR:** hiring, resign, promote, salary, benefits, policy\n"
        "- **Travel & Transport:** itinerary, departure, delay, connecting flight, accommodation\n"
        "- **Finance & Banking:** invoice, transaction, refund, deposit, budget\n"
        "- **Meetings & Communication:** agenda, proposal, collaborate, deadline\n"
        "- **Contracts & Agreements:** clause, effective, period, renewal, terminate\n\n"
        "**Study tip:** Keep a notebook. Every time you see a new word in a TOEIC passage, write it down with the sentence it appeared in — not just its definition. Context helps memory!"
    ),
    "grammar": (
        "**📝 TOEIC Grammar Hotspots**\n\n"
        "Part 5 & 6 focus heavily on:\n\n"
        "1. **Prepositions** — at/in/on (time vs place), by/until, for/since. These are highly tested.\n"
        "2. **Verb tenses** — present perfect (has/have + past participle) for recent events; past simple for completed actions.\n"
        "3. **Word forms** — 'economy' (noun) vs 'economic' (adj) vs 'economically' (adv). Learn the suffixes.\n"
        "4. **Subject-verb agreement** — 'The manager __' (has) vs 'The managers __' (have). Always check the subject.\n"
        "5. **Relative clauses** — who (people), which (things), where (places), that (both).\n"
        "6. **Comparatives & Superlatives** — more/most, -er/-est, 'as...as' structures.\n\n"
        "**Quick trick:** When stuck, read the sentence without the blank and hear which word sounds natural in context."
    ),
    "reading_comprehension": (
        "**📖 Reading Comprehension Strategy**\n\n"
        "For Part 7 passages:\n\n"
        "1. **Preview the questions first** — read them before the passage. This tells you what to look for.\n"
        "2. **Identify passage type** — email, memo, article, ad, schedule. Each has a different structure.\n"
        "3. **Annotate mentally** — notice the 'who, what, when, where, why' as you read.\n"
        "4. **Double passages** — read Passage A, answer its questions, then read Passage B. The questions usually refer to one passage at a time.\n"
        "5. **Triple passages** — look for the connection between them. Often B expands on A, and C compares or contrasts.\n"
        "6. **Eliminate wrong answers** — cross out options that are too extreme, irrelevant, or contradict the text."
    )
}


def handle_tool_call(tool_call):
    """Execute a tool call and return the result."""
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

    if name == "get_student_test_history":
        student_id = args.get("student_id")
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
            "attempt_id": r["id"],
            "score": r["raw_score"],
            "total": r["total_questions"],
            "difficulty": r["difficulty"],
            "date": r["completed_at"]
        } for r in rows]
        return json.dumps({"attempts": attempts}, ensure_ascii=False)

    elif name == "get_test_attempt_details":
        attempt_id = args.get("attempt_id")
        student_id = args.get("student_id")
        conn = get_db()
        c = conn.cursor()

        # Get the test attempt
        c.execute("""
            SELECT ta.test_id, ta.answers, ta.raw_score, ta.total_questions, ta.completed_at
            FROM test_attempts ta
            WHERE ta.id = ? AND ta.student_id = ?
        """, (attempt_id, student_id))
        attempt = c.fetchone()
        if not attempt:
            conn.close()
            return json.dumps({"error": "Attempt not found"})

        # Get the test data
        c.execute("SELECT passages, questions FROM reading_tests WHERE id=? AND student_id=?",
                  (attempt["test_id"], student_id))
        test = c.fetchone()
        if not test:
            conn.close()
            return json.dumps({"error": "Test not found"})

        conn.close()

        passages = json.loads(test["passages"])
        questions = json.loads(test["questions"])
        user_answers = json.loads(attempt["answers"])
        user_map = {a.get("question_id"): a.get("answer", "") for a in user_answers}

        results = []
        for q in questions:
            qid = q["id"]
            ua = user_map.get(qid, "")
            results.append({
                "question_id": qid,
                "passage_id": q.get("passage_id"),
                "stem": q["stem"],
                "options": q["options"],
                "user_answer": ua,
                "correct_answer": q["correct_answer"],
                "correct": ua.upper() == q["correct_answer"].upper(),
                "explanation": q.get("explanation", "")
            })

        return json.dumps({
            "test_id": attempt["test_id"],
            "passages": passages,
            "results": results,
            "score": attempt["raw_score"],
            "total": attempt["total_questions"],
            "completed_at": attempt["completed_at"]
        }, ensure_ascii=False)

    elif name == "get_toeic_tip":
        topic = args.get("topic", "time_management")
        tip = TOEIC_TIPS.get(topic, TOEIC_TIPS["time_management"])
        return json.dumps({"topic": topic, "tip": tip}, ensure_ascii=False)

    return json.dumps({"error": f"Unknown tool: {name}"})


TOPIC_POOL = [
    "A company announcing flexible working hours policy",
    "A hotel confirming a booking with special requests",
    "A software release announcing new features",
    "A delivery service changing its shipping rates",
    "A university sending admission results",
    "A restaurant launching a weekend brunch menu",
    "An airline updating its carry-on baggage rules",
    "A department store announcing a clearance sale",
    "A clinic sending appointment reminders",
    "A bank notifying customers about new security features",
    "A conference center announcing an upcoming tech summit",
    "A magazine sending a subscription renewal offer",
    "A car rental company introducing loyalty rewards",
    "A pharmacy launching a home delivery service",
    "A gym announcing new fitness class schedules",
    "An insurance company updating its coverage plans",
    "A coffee shop introducing a loyalty card program",
    "An online retailer announcing free shipping week",
    "A museum announcing a new exhibition opening",
    "A courier service updating its tracking system",
    "A company sending a meeting agenda for quarterly review",
    "A travel agency promoting a holiday package deal",
    "A library announcing extended opening hours",
    "A parking garage introducing mobile payment",
    "A cinema announcing a membership program",
    "A language school offering new course levels",
    "A supermarket announcing a loyalty discount event",
    "A train company updating its schedule",
    "A recruitment agency sending job interview tips",
    "A music streaming service launching a family plan",
    "A co-working space announcing new locations",
    "An appliance store offering a trade-in promotion",
    "A moving company sharing packing tips for customers",
    "An airport announcing new direct flight routes",
    "A bookstore hosting a meet-the-author event",
]

TEST_GENERATION_PROMPT = """You are a TOEIC Reading test generator. Every time you generate, you MUST produce completely different content from before.

DIFFICULTY: {difficulty}
SEED: {seed}
TOPIC: {topic}

IMPORTANT — Use the SEED and TOPIC above to generate unique content. Different SEED + TOPIC combinations MUST produce different passages, different company names, different people names, different places, and different questions. Never reuse content.

Generate 2-3 short business-related reading passages and exactly 10 comprehension questions about the TOPIC given above.

DIFFICULTY GUIDELINES:
- EASY: Short passages (50-80 words), simple vocabulary, direct facts tested
- MEDIUM: Moderate passages (80-120 words), intermediate vocabulary, some inference
- HARD: Longer passages (100-150 words), advanced vocabulary, multiple inference types

Each question must have 4 options (A, B, C, D) with exactly one correct answer.
Include a brief explanation for each question.

RULES:
- Invent unique company names, people names, locations, dates, and prices — never use generic placeholders
- Do NOT use "Company A", "Person B", or any placeholder names
- Every passage must be about the specific TOPIC given above
- The 10 questions should cover: main idea, specific details, inference, and vocabulary in context

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
    """Parse JSON from LLM output, handling common formatting issues."""
    if not content:
        raise ValueError("Empty content")

    # Step 1: Find the first ``` block and extract everything inside it
    start_fence = content.find('```')
    if start_fence != -1:
        after_fence = content[start_fence + 3:].strip()
        # Skip optional language label (e.g. "json")
        first_line_end = after_fence.find('\n')
        if first_line_end != -1:
            candidate = after_fence[first_line_end:].strip()
        else:
            candidate = after_fence
        # Remove trailing ```
        end_fence = candidate.rfind('```')
        if end_fence != -1:
            candidate = candidate[:end_fence].strip()
        content = candidate

    # Step 2: Find outermost { … } pair
    brace_start = content.find('{')
    brace_end = content.rfind('}')
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        content = content[brace_start:brace_end + 1]

    # Step 3: Fix trailing commas before ] or }
    import re
    content = re.sub(r',\s*([\]}])', r'\1', content)

    # Step 4: Parse
    try:
        return json.loads(content, strict=False)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}") from e


FALLBACK_SETS = [
    (FALLBACK_PASSAGES, FALLBACK_QUESTIONS),
    ([
        {"id": 1, "title": "New Branch Opening", "text": "To: Marketing Team\nFrom: David Wong, Regional Director\nSubject: New Branch in Surabaya\n\nDear Team,\n\nI am excited to announce that our new branch office in Surabaya will officially open on March 15. The address is Jl. Panglima Sudirman No. 45.\n\nThe opening ceremony will begin at 9:00 AM with a ribbon-cutting event. Local business partners and media have been invited. Marketing materials and promotional packages are ready for distribution.\n\nPlease confirm your attendance by March 10. Transportation from the main office will be provided.\n\nBest regards,\nDavid"},
        {"id": 2, "title": "TechFix Pro Service", "text": "Is your computer running slow? TechFix Pro offers professional维修 services for all devices.\n\nOur Services:\n- Laptop & desktop repair\n- Virus removal and data recovery\n- Hardware upgrades and installation\n- Network setup and troubleshooting\n- Software installation and configuration\n\nAll repairs come with a 30-day warranty. Free diagnostic check for first-time customers. Visit our store at Plaza Senayan, 2nd Floor or call 021-555-1234.\n\nOpen Monday to Saturday, 9:00 AM to 7:00 PM."},
        {"id": 3, "title": "Annual Company Picnic", "text": "Dear Colleagues,\n\nOur annual company picnic will be held on Saturday, June 22 at Taman Wisata Alam, Bogor. This is a family-friendly event, and we encourage everyone to bring their family members.\n\nSchedule:\n- 08:00: Depart from office (bus provided)\n- 10:00: Arrival and welcome activities\n- 12:00: Lunch buffet\n- 13:30: Team games and competitions\n- 16:00: Free time\n- 17:30: Return to office\n\nPlease RSVP by June 10 to the HR department. Indicate the number of family members joining so we can arrange transportation and food accordingly.\n\nLooking forward to a great day together!\nHR Department"},
    ], [
        {"id": 1, "passage_id": 1, "stem": "What is the purpose of this email?", "options": ["A. To announce a new branch opening", "B. To report sales figures", "C. To request budget approval", "D. To introduce a new product"], "correct_answer": "A", "explanation": "The email announces the opening of a new branch office in Surabaya."},
        {"id": 2, "passage_id": 1, "stem": "When will the opening ceremony take place?", "options": ["A. March 10", "B. March 15", "C. June 22", "D. March 1"], "correct_answer": "B", "explanation": "The email states the new branch will officially open on March 15."},
        {"id": 3, "passage_id": 1, "stem": "What should team members do by March 10?", "options": ["A. Prepare marketing materials", "B. Confirm their attendance", "C. Submit a budget report", "D. Visit the new office"], "correct_answer": "B", "explanation": "Team members should confirm their attendance by March 10."},
        {"id": 4, "passage_id": 2, "stem": "What service does TechFix Pro offer?", "options": ["A. Cooking classes", "B. Computer repair services", "C. Fitness training", "D. Language courses"], "correct_answer": "B", "explanation": "TechFix Pro offers professional repair services for computers and devices."},
        {"id": 5, "passage_id": 2, "stem": "What is offered to first-time customers?", "options": ["A. 50% discount", "B. Free diagnostic check", "C. Lifetime warranty", "D. Free delivery"], "correct_answer": "B", "explanation": "Free diagnostic check is offered for first-time customers."},
        {"id": 6, "passage_id": 2, "stem": "Where is TechFix Pro located?", "options": ["A. Taman Wisata Alam", "B. Plaza Senayan", "C. Jl. Panglima Sudirman", "D. Surabaya"], "correct_answer": "B", "explanation": "The store is located at Plaza Senayan, 2nd Floor."},
        {"id": 7, "passage_id": 3, "stem": "What is the main purpose of the picnic?", "options": ["A. Team building and family gathering", "B. Client entertainment", "C. Product launch", "D. Training session"], "correct_answer": "A", "explanation": "The picnic is described as a family-friendly event for team building."},
        {"id": 8, "passage_id": 3, "stem": "What time will the group depart from the office?", "options": ["A. 06:00", "B. 08:00", "C. 10:00", "D. 07:00"], "correct_answer": "B", "explanation": "The bus departs from the office at 08:00."},
        {"id": 9, "passage_id": 3, "stem": "What should employees indicate in their RSVP?", "options": ["A. Food preferences", "B. Number of family members", "C. T-shirt size", "D. Departure time preference"], "correct_answer": "B", "explanation": "Employees should indicate the number of family members joining for transportation and food arrangements."},
        {"id": 10, "passage_id": 3, "stem": "By when should employees RSVP?", "options": ["A. March 15", "B. June 10", "C. June 22", "D. March 10"], "correct_answer": "B", "explanation": "Employees should RSVP by June 10 to the HR department."},
    ]),
    ([
        {"id": 1, "title": "Hotel Booking Confirmation", "text": "Dear Mr. Hartono,\n\nThank you for choosing Grand Horizon Hotel. This email confirms your reservation:\n\nCheck-in: Friday, July 12, 2024\nCheck-out: Sunday, July 14, 2024\nRoom Type: Deluxe Ocean View (2 adults)\nRoom Rate: Rp 850,000 per night\nTotal: Rp 1,700,000\n\nYour request for a quiet room on a high floor has been noted. We will do our best to accommodate.\n\nAmenities included: Complimentary breakfast, airport shuttle, Wi-Fi, and access to the fitness center and pool.\n\nCheck-in time is 2:00 PM. Early check-in is subject to availability. Please present your reservation number (GH-2024-7890) at the front desk.\n\nWe look forward to welcoming you!\nGrand Horizon Hotel Reservations"},
        {"id": 2, "title": "Quarterly Sales Meeting", "text": "To: Sales Department\nFrom: Maya Putri, Head of Sales\nSubject: Q3 Sales Meeting Agenda\n\nDear Team,\n\nOur Q3 sales meeting will be held on Thursday, September 5 at 10:00 AM in Conference Room A.\n\nAgenda:\n1. Q2 performance review (15 min)\n2. New product launch strategy (30 min)\n3. Regional market analysis (20 min)\n4. Q3 target setting (25 min)\n5. Open discussion (15 min)\n\nPlease prepare your individual sales reports and bring them to the meeting. If you cannot attend, notify me by September 3.\n\nRefreshments will be provided.\n\nRegards,\nMaya"},
        {"id": 3, "title": "Weekly Promotion", "text": "GreenMart Supermarket Weekly Deals!\n\nValid from Monday, August 5 to Sunday, August 11.\n\nFresh Produce:\n- Apples: Rp 25,000/kg (regular Rp 35,000)\n- Broccoli: Rp 12,000/piece (regular Rp 18,000)\n- Fresh salmon: Rp 85,000/250g\n\nHousehold:\n- Dish soap: Buy 2 get 1 free\n- Paper towels: Rp 45,000 for 6 rolls\n\nSpecial: Spend Rp 200,000 or more and receive a free shopping bag!\n\n*While stocks last. Prices may vary at different locations."},
    ], [
        {"id": 1, "passage_id": 1, "stem": "What type of room did Mr. Hartono reserve?", "options": ["A. Standard Room", "B. Deluxe Ocean View", "C. Suite", "D. Family Room"], "correct_answer": "B", "explanation": "The reservation is for a Deluxe Ocean View room."},
        {"id": 2, "passage_id": 1, "stem": "How much will Mr. Hartono pay in total?", "options": ["A. Rp 850,000", "B. Rp 1,700,000", "C. Rp 2,000,000", "D. Rp 1,000,000"], "correct_answer": "B", "explanation": "The total is Rp 1,700,000 for two nights at Rp 850,000 per night."},
        {"id": 3, "passage_id": 1, "stem": "Which amenity is included with the room?", "options": ["A. Spa access", "B. Airport shuttle", "C. Room service", "D. Laundry service"], "correct_answer": "B", "explanation": "Complimentary airport shuttle is listed as an included amenity."},
        {"id": 4, "passage_id": 2, "stem": "When is the Q3 sales meeting?", "options": ["A. September 3", "B. September 5", "C. August 5", "D. July 12"], "correct_answer": "B", "explanation": "The meeting is on Thursday, September 5."},
        {"id": 5, "passage_id": 2, "stem": "What should team members bring to the meeting?", "options": ["A. Laptops", "B. Individual sales reports", "C. Client feedback forms", "D. Marketing brochures"], "correct_answer": "B", "explanation": "Team members should prepare their individual sales reports and bring them to the meeting."},
        {"id": 6, "passage_id": 2, "stem": "How much time is allocated for Q2 performance review?", "options": ["A. 15 minutes", "B. 20 minutes", "C. 25 minutes", "D. 30 minutes"], "correct_answer": "A", "explanation": "The Q2 performance review is scheduled for 15 minutes."},
        {"id": 7, "passage_id": 3, "stem": "What is the discounted price for apples?", "options": ["A. Rp 25,000/kg", "B. Rp 35,000/kg", "C. Rp 12,000/kg", "D. Rp 45,000/kg"], "correct_answer": "A", "explanation": "Apples are discounted to Rp 25,000 per kilogram from the regular Rp 35,000."},
        {"id": 8, "passage_id": 3, "stem": "What promotion is offered for dish soap?", "options": ["A. 50% off", "B. Buy 2 get 1 free", "C. Free sample", "D. Rp 10,000 discount"], "correct_answer": "B", "explanation": "Dish soap is on a 'Buy 2 get 1 free' promotion."},
        {"id": 9, "passage_id": 3, "stem": "What do customers get when they spend Rp 200,000?", "options": ["A. A discount voucher", "B. A free shopping bag", "C. A loyalty point bonus", "D. Free delivery"], "correct_answer": "B", "explanation": "Customers who spend Rp 200,000 or more receive a free shopping bag."},
        {"id": 10, "passage_id": 3, "stem": "How long is this promotion valid?", "options": ["A. 3 days", "B. 5 days", "C. 7 days", "D. 10 days"], "correct_answer": "C", "explanation": "The promotion runs from Monday to Sunday, which is 7 days."},
    ]),
]


# ── Dynamic fallback sets from topic templates ──────────────

COMPANY_NAMES = ["Apex Corp", "NexGen", "Blue Horizon", "Mandala Group", "Prima Solutions",
    "GlobalLink", "Sinar Abadi", "Citra Nusantara", "Evergreen", "Mitra Sejahtera"]
PERSON_NAMES = ["Anita Wijaya", "Budi Hartono", "Citra Dewi", "Dimas Prasetyo",
    "Eka Putri", "Fajar Ramadhan", "Gita Permata", "Hendra Gunawan"]
CITIES = ["Jakarta", "Surabaya", "Bandung", "Yogyakarta", "Medan", "Makassar", "Semarang"]
STREETS = ["Jl. Sudirman No. 45", "Jl. Gatot Subroto No. 120", "Jl. Thamrin Kav. 28",
    "Jl. A. Yani No. 67", "Jl. Diponegoro No. 15"]
PRICES = ["Rp 150,000", "Rp 250,000", "Rp 500,000", "Rp 75,000", "Rp 1,200,000"]
MONTHS_DAY = [("January", 15), ("February", 20), ("March", 10), ("April", 5),
    ("May", 18), ("June", 22), ("July", 12), ("August", 8), ("September", 25),
    ("October", 14), ("November", 3), ("December", 19)]


def _p(pid, title, text):
    return {"id": pid, "title": title, "text": text}


def _q(qid, pid, stem, options, correct, explanation):
    return {"id": qid, "passage_id": pid, "stem": stem,
            "options": options, "correct_answer": correct, "explanation": explanation}


def _build_topic_set(topic, idx):
    """Generate a fallback test set from a topic description."""
    mon, day = MONTHS_DAY[idx % len(MONTHS_DAY)]
    co = COMPANY_NAMES[idx % len(COMPANY_NAMES)]
    co2 = COMPANY_NAMES[(idx + 3) % len(COMPANY_NAMES)]
    person = PERSON_NAMES[idx % len(PERSON_NAMES)]
    person2 = PERSON_NAMES[(idx + 4) % len(PERSON_NAMES)]
    city = CITIES[idx % len(CITIES)]
    street = STREETS[idx % len(STREETS)]
    price = PRICES[idx % len(PRICES)]

    p1_title = f"{co} Announcement"
    p1_text = (
        f"To: All Staff\nFrom: {person}, HR Manager\nSubject: {topic}\n\n"
        f"Dear Team,\n\nWe are pleased to announce that effective {mon} {day}, "
        f"{co} will implement new policies regarding {topic.lower()}.\n\n"
        f"All employees are requested to read the updated guidelines posted on the company portal. "
        f"A training session will be held on {mon} {day + 1} at 10:00 AM in the main conference room.\n\n"
        f"For questions, please contact {person2} at extension 4502.\n\nBest regards,\n{person}"
    )

    p2_title = f"{co2} Services"
    p2_text = (
        f"Welcome to {co2}! We are proud to offer our new service package designed to meet your needs.\n\n"
        f"Our Services:\n- Professional consultation\n- Customized solutions\n- 24/7 customer support\n"
        f"- Fast and reliable delivery\n- Satisfaction guaranteed\n\n"
        f"Special introductory offer: {price} for the first month! "
        f"Visit our office at {street}, {city} or call 021-555-{1000 + idx}.\n\n"
        f"Open Monday to Friday, 8:00 AM to 6:00 PM."
    )

    p3_title = f"Event: {city} Workshop"
    p3_text = (
        f"Dear Colleagues,\n\n{co} is hosting a professional development workshop on {mon} {day + 10} "
        f"at the {city} Convention Center.\n\nSchedule:\n"
        f"- 08:30: Registration and coffee\n- 09:00: Opening remarks by {person}\n"
        f"- 10:30: Breakout sessions\n- 12:00: Networking lunch\n- 13:30: Panel discussion\n"
        f"- 15:00: Closing ceremony\n\nPlease RSVP by {mon} {day - 2 if day > 2 else 3} to the HR department. "
        f"Transportation will be provided from the main office.\n\nLooking forward to your participation!"
    )

    passages = [_p(1, p1_title, p1_text), _p(2, p2_title, p2_text), _p(3, p3_title, p3_text)]

    questions = [
        _q(1, 1, f"What is the main purpose of this announcement?",
           ["A. To introduce a new product", "B. To announce new company policies",
            "C. To invite employees to a workshop", "D. To report quarterly results"], "B",
           f"The email announces new policies regarding {topic.lower()}."),
        _q(2, 1, f"When will the training session take place?",
           [f"A. {mon} {day}", f"B. {mon} {day + 1}", f"C. {mon} {day + 10}", f"D. {mon} {day - 1}"], "B",
           f"The training session will be held on {mon} {day + 1} at 10:00 AM."),
        _q(3, 1, "Who should employees contact for questions?",
           ["A. The IT department", f"B. {person2}", "C. The finance team",
            "D. The marketing director"], "B",
           f"Employees can contact {person2} at extension 4502."),
        _q(4, 2, f"What service does {co2} offer?",
           [f"A. {topic} services", "B. Financial consulting", "C. Legal advice",
            "D. Construction services"], "A",
           f"{co2} offers professional {topic.lower()} services."),
        _q(5, 2, f"What is the special introductory offer?",
           [f"A. {price} for the first month", "B. Free consultation",
            "C. 50% discount", "D. Lifetime membership"], "A",
           f"The special offer is {price} for the first month."),
        _q(6, 2, f"Where is the office located?",
           [f"A. {street}, {city}", "B. Jl. Merdeka No. 10", "C. Plaza Indonesia",
            "D. Grand Indonesia"], "A",
           f"The office is at {street}, {city}."),
        _q(7, 3, f"What is the main purpose of the workshop?",
           ["A. Professional development", "B. Product launch",
            "C. Annual celebration", "D. Team building"], "A",
           "The workshop focuses on professional development."),
        _q(8, 3, f"What time does the opening remarks start?",
           ["A. 08:30", "B. 09:00", "C. 10:30", "D. 12:00"], "B",
           "Opening remarks begin at 09:00."),
        _q(9, 3, f"What should participants do by the RSVP deadline?",
           [f"A. Submit a report", "B. Confirm attendance to HR",
            "C. Pay the registration fee", "D. Book their own transport"], "B",
           "Participants should RSVP to the HR department."),
        _q(10, 3, f"What will be provided for transportation?",
           ["A. Reimbursement for taxi fares", "B. A bus from the main office",
            "C. Parking vouchers", "D. Rental car service"], "B",
           "Transportation will be provided from the main office."),
    ]
    return passages, questions


# Build many varied fallback sets from TOPIC_POOL
MORE_FALLBACK_SETS = [_build_topic_set(t, i) for i, t in enumerate(TOPIC_POOL[:15])]
FALLBACK_SETS = FALLBACK_SETS + MORE_FALLBACK_SETS


def generate_test_via_llm(difficulty, seed, topic):
    prompt = TEST_GENERATION_PROMPT.format(difficulty=difficulty, seed=seed, topic=topic)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": "You are a TOEIC test generator. Return ONLY valid JSON."}, {"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        max_tokens=4096,
        tools=TOOLS,
        tool_choice="none",
    )
    raw = response.choices[0].message.content
    data = parse_llm_json(raw)
    passages = data.get('passages', None)
    questions = data.get('questions', None)
    if not passages or not questions or len(questions) != 10:
        raise ValueError("Invalid LLM response")
    return passages, questions


def get_fallback_set(seed):
    return random.choice(FALLBACK_SETS)


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
    student_id = data.get('student_id')

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
        # Include student_id in system prompt context if available
        if student_id:
            messages[0]["content"] += f"\n\nThe current student's database ID is: {student_id}"

        # First call with tools
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=1500,
            tools=TOOLS,
            tool_choice="auto",
        )

        assistant_message = response.choices[0].message
        assistant_content = assistant_message.content or ''

        # Handle tool calls in a loop (max 5 iterations to prevent infinite loops)
        max_tool_rounds = 5
        current_round = 0

        while assistant_message.tool_calls and current_round < max_tool_rounds:
            current_round += 1

            # Append the assistant message with tool calls
            messages.append({
                "role": "assistant",
                "content": assistant_message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in assistant_message.tool_calls
                ]
            })

            # Execute each tool call
            for tc in assistant_message.tool_calls:
                tool_result = handle_tool_call(tc)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result
                })

            # Re-call LLM with tool results
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=1500,
                tools=TOOLS,
                tool_choice="auto",
            )
            assistant_message = response.choices[0].message
            if assistant_message.content:
                assistant_content = assistant_message.content

        final_content = assistant_content or ""
        save_message(session_id, 'assistant', final_content)
        return jsonify({'response': final_content, 'session_id': session_id})

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
    topic = TOPIC_POOL[random.randint(0, len(TOPIC_POOL) - 1)]

    for attempt in range(3):
        seed = f"{student_id}-{int(time.time() * 1000)}-{random.randint(0, 9999)}"
        try:
            passages, questions = generate_test_via_llm(difficulty, seed, topic)
        except Exception as e:
            print(f"[DEBUG] LLM attempt {attempt} failed: {e}", flush=True)
            seed = f"fallback-{int(time.time() * 1000000)}-{random.randint(0, 99999)}"
            passages, questions = get_fallback_set(seed)

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
        except Exception as e:
            print(f"[DEBUG] DB attempt {attempt} failed: {e}", flush=True)
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
    passage_ids = {q['id']: q['passage_id'] for q in stored_questions}

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
            'passage_id': passage_ids.get(qid),
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
        SELECT ta.id, ta.test_id, ta.raw_score, ta.total_questions, ta.completed_at, rt.difficulty
        FROM test_attempts ta
        JOIN reading_tests rt ON ta.test_id = rt.id
        WHERE ta.student_id = ?
        ORDER BY ta.completed_at DESC
    """, (student_id,))
    rows = c.fetchall()
    conn.close()

    attempts = [{
        'attempt_id': r['id'],
        'test_id': r['test_id'],
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


# ── Admin ─────────────────────────────────────────────────────────

ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD', 'admin123')


def require_admin():
    if not session.get('admin'):
        return jsonify({'error': 'Unauthorized'}), 401


@app.route('/admin/login', methods=['POST'])
def admin_login():
    data = request.json or {}
    if data.get('password') == ADMIN_PASSWORD:
        session['admin'] = True
        return jsonify({'success': True})
    return jsonify({'error': 'Wrong password'}), 401


@app.route('/api/admin/stats')
def admin_stats():
    r = require_admin()
    if r:
        return r
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM students")
    students_count = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM test_attempts")
    tests_count = c.fetchone()[0]
    c.execute("SELECT ROUND(AVG(raw_score), 1) FROM test_attempts")
    row = c.fetchone()
    avg_score = row[0] if row[0] else 0
    c.execute("""SELECT s.name, s.nim, ta.raw_score, ta.total_questions, ta.completed_at
                 FROM test_attempts ta JOIN students s ON ta.student_id = s.id
                 ORDER BY ta.completed_at DESC LIMIT 5""")
    recent = [dict(r) for r in c.fetchall()]
    conn.close()
    return jsonify({'students_count': students_count, 'tests_count': tests_count,
                    'avg_score': avg_score, 'recent': recent})


@app.route('/api/admin/students')
def admin_students():
    r = require_admin()
    if r:
        return r
    conn = get_db()
    c = conn.cursor()
    c.execute("""SELECT s.id, s.name, s.nim, s.created_at,
                        COUNT(ta.id) as tests_taken,
                        ROUND(AVG(ta.raw_score), 1) as avg_score,
                        MAX(ta.completed_at) as last_test
                 FROM students s
                 LEFT JOIN test_attempts ta ON ta.student_id = s.id
                 GROUP BY s.id ORDER BY last_test DESC""")
    students = [dict(r) for r in c.fetchall()]
    conn.close()
    return jsonify({'students': students})


@app.route('/api/admin/students/<int:student_id>')
def admin_student_detail(student_id):
    r = require_admin()
    if r:
        return r
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT id, name, nim, created_at FROM students WHERE id=?", (student_id,))
    student = c.fetchone()
    if not student:
        conn.close()
        return jsonify({'error': 'Student not found'}), 404
    c.execute("""SELECT ta.id, ta.raw_score, ta.total_questions, ta.completed_at, rt.difficulty
                 FROM test_attempts ta
                 JOIN reading_tests rt ON rt.id = ta.test_id
                 WHERE ta.student_id = ? ORDER BY ta.completed_at DESC""", (student_id,))
    attempts = [dict(r) for r in c.fetchall()]
    conn.close()
    return jsonify({'student': dict(student), 'attempts': attempts})


@app.route('/api/admin/attempts')
def admin_attempts():
    r = require_admin()
    if r:
        return r
    conn = get_db()
    c = conn.cursor()
    c.execute("""SELECT ta.id, ta.raw_score, ta.total_questions, ta.completed_at,
                        s.name, s.nim, rt.difficulty
                 FROM test_attempts ta
                 JOIN students s ON s.id = ta.student_id
                 JOIN reading_tests rt ON rt.id = ta.test_id
                 ORDER BY ta.completed_at DESC LIMIT 100""")
    attempts = [dict(r) for r in c.fetchall()]
    conn.close()
    return jsonify({'attempts': attempts})


@app.route('/admin')
def admin_page():
    return render_template('admin.html')


# ── Main ────────────────────────────────────────────────────────

if __name__ == '__main__':
    init_db()
    port = int(os.getenv('PORT', 9120))
    print(f"TOEIC Reading Practice running on http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
