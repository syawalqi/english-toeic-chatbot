#!/usr/bin/env python3
"""Test TOEIC Reading Practice API endpoints"""
import json, time, sys, os, uuid
from urllib import request as urlreq

BASE = "http://localhost:9120/api"
OUT = os.getenv('TEST_OUTPUT_PATH', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test-results.json'))

def post(path, body, timeout=15):
    data = json.dumps(body).encode('utf-8')
    req = urlreq.Request(BASE + path, data=data,
        headers={"Content-Type": "application/json"},
        method="POST")
    try:
        with urlreq.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode('utf-8')), resp.status
    except urlreq.HTTPError as e:
        return json.loads(e.read().decode('utf-8')), e.code
    except Exception as e:
        return {"error": str(e)}, 0

def get(path, timeout=10):
    try:
        with urlreq.urlopen(BASE + path, timeout=timeout) as resp:
            return json.loads(resp.read().decode('utf-8')), resp.status
    except Exception as e:
        return {"error": str(e)}, 0

results = []
t_start = time.time()

def check(name, success, detail=""):
    elapsed = time.time() - t_start
    results.append({"name": name, "success": success, "detail": detail, "time": round(elapsed, 2)})
    status = "PASS" if success else "FAIL"
    print(f"  [{status}] {name}")
    sys.stdout.flush()

print("TOEIC Reading Practice API Tests")
print("=" * 40)
print()

# 1. Register student
print("── Registration ──")
data, code = post("/register", {"name": "Test Student", "nim": "20240001"})
check("Register new student", code == 201 and "student_id" in data)

data2, code2 = post("/register", {"name": "Test Student", "nim": "20240001"})
check("Register existing student (same name)", code2 == 200 and data2["student_id"] == data["student_id"])

data3, code3 = post("/register", {"name": "Wrong Name", "nim": "20240001"})
check("Register existing NIM (different name)", code3 == 409)

data4, code4 = post("/register", {"name": "", "nim": ""})
check("Register empty fields", code4 == 400)

student_id = data.get("student_id")

# 2. Generate tests
print("\n── Test Generation ──")
sid = uuid.uuid4().hex[:8]

gen_data, gen_code = post("/test/generate", {"student_id": student_id, "difficulty": "medium"})
check("Generate medium test", gen_code == 200 and "test_id" in gen_data and len(gen_data.get("questions", [])) == 10)

gen_easy, _ = post("/test/generate", {"student_id": student_id, "difficulty": "easy"})
check("Generate easy test", "test_id" in gen_easy)

gen_hard, _ = post("/test/generate", {"student_id": student_id, "difficulty": "hard"})
check("Generate hard test", "test_id" in gen_hard)

gen_bad, bad_code = post("/test/generate", {"student_id": student_id, "difficulty": "extreme"})
check("Generate invalid difficulty", bad_code == 400)

gen_noid, noid_code = post("/test/generate", {"difficulty": "medium"})
check("Generate without student_id", noid_code == 400)

test_id = gen_data.get("test_id")

# 3. Submit answers
print("\n── Answer Submission ──")
questions = gen_data.get("questions", [])
all_correct = [{"question_id": q["id"], "answer": "A"} for q in questions]
all_wrong = [{"question_id": q["id"], "answer": "X"} for q in questions]
mixed = []
for i, q in enumerate(questions):
    mixed.append({"question_id": q["id"], "answer": "A" if i < 5 else "X"})

sub1, sc1 = post("/test/submit", {"student_id": student_id, "test_id": test_id, "answers": all_correct})
check("Submit all correct (should guess)", sc1 == 200 and "score" in sub1)

sub2, _ = post("/test/submit", {"student_id": student_id, "test_id": test_id, "answers": all_wrong})
check("Submit all wrong", sub2.get("score", 0) <= 3)

sub3, _ = post("/test/submit", {"student_id": student_id, "test_id": test_id, "answers": mixed})
check("Submit mixed answers", 1 <= sub3.get("score", 0) <= 9)

sub_bad, bc = post("/test/submit", {"student_id": student_id, "test_id": 99999, "answers": []})
check("Submit invalid test_id", bc == 404)

# 4. History
print("\n── History ──")
hist, hc = get(f"/history?student_id={student_id}")
check("Get history", hc == 200 and "attempts" in hist)
check("History has entries", len(hist.get("attempts", [])) >= 3)

hist_empty, _ = get("/history?student_id=99999")
check("History no student", hc == 200)

# 5. Result detail
print("\n── Result Detail ──")
res, rc = get(f"/test/result/{test_id}?student_id={student_id}")
check("Get result detail", rc == 200 and "passages" in res and "results" in res)

res_noid, nc = get(f"/test/result/{test_id}?student_id=99999")
check("Get result wrong student", nc == 404)

# 6. Chat
print("\n── Chat ──")
chat, cc = post("/chat", {"message": "Give me a TOEIC tip", "session_id": "test-session-1"})
check("Chat message", cc == 200 and "response" in chat)
check("Chat has response text", len(chat.get("response", "")) > 0)

chat_empty, ec = post("/chat", {"message": "", "session_id": "test-session-1"})
check("Chat empty message", ec == 400)

# Summary
total = len(results)
passed = sum(1 for r in results if r["success"])
elapsed = time.time() - t_start

print(f"\n{'=' * 40}")
print(f"Results: {passed}/{total} passed ({passed/total*100:.1f}%)")
print(f"Time: {elapsed:.1f}s")

with open(OUT, 'w') as f:
    json.dump({"timestamp": time.time(), "total": total, "passed": passed,
        "elapsed": elapsed, "results": results}, f, indent=2)
print(f"Saved: {OUT}")
