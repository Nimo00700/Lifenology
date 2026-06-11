#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
라이프놀로지 랩 — GEO 측정 스크립트 v3
Samsung Life x Lifenology Lab 3기

수정 내역:
  v3: questions.csv 경로 os.path 고정 (GitHub Actions 오류 완전 해결)
      google.genai 신버전으로 교체 (FutureWarning 제거)
      Gemma 2 9B → Llama 3.1 8B (지원 종료 대응)
      결과를 results_all.csv 에 누적 저장 (대시보드 자동 연동)

패키지 설치:
  pip install groq google-genai sentence-transformers
"""

import os, csv, json, time, datetime
from groq import Groq
from google import genai as google_genai
from sentence_transformers import SentenceTransformer, util

# ════════════════════════════════════════════════════════════════
#  ① 설정  ←  여기만 수정하면 됩니다
# ════════════════════════════════════════════════════════════════

# GitHub Actions 에서는 Secrets 에서 자동으로 읽어옴
# 로컬에서 직접 실행할 경우 아래 따옴표 안에 키 입력
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY",   "gsk_여기에입력")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY",  "AQ여기에입력")

# ── 경로 (수정 불필요) ──
# 스크립트 파일 위치를 기준으로 찾으므로 어디서 실행해도 동작
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
QUESTIONS_FILE = os.path.join(SCRIPT_DIR, "questions.csv")
ALL_CSV        = os.path.join(SCRIPT_DIR, "results_all.csv")

# ── 브랜드 감지 키워드 ──
BRAND_KEYWORDS = [
    "삼성생명", "라이프놀로지랩", "라이프놀로지 랩",
    "lifenology", "Lifenology",
]

# ── 웹사이트 핵심 소개 텍스트 (시맨틱 점수 기준) ──
WEBSITE_TEXT = """
라이프놀로지 랩은 삼성생명이 대학생들과 함께 '보험을 넘어서는 보험'이라는
브랜드 메시지를 실천하며 더 나은 내일을 위한 혁신 솔루션을 개발하는 산학협력 프로젝트입니다.
1기·2기·3기를 통해 헬스케어, 테크, 라이프스타일 분야의 혁신적인 아이디어를 발굴하고 실현합니다.
삼성생명은 단순한 보험 상품을 넘어 고객의 삶 전반을 지원하는 혁신적 솔루션을 추구합니다.
"""

SEMANTIC_TARGETS = [
    "보험 혁신 솔루션",
    "대학생 산학협력 헬스케어 프로그램",
    "미래 라이프스타일 기술 혁신",
    "혁신적인 보험 브랜드 신뢰",
    "지속가능한 금융 라이프 솔루션",
]

# ════════════════════════════════════════════════════════════════
#  ② 클라이언트 초기화
# ════════════════════════════════════════════════════════════════

groq_client   = Groq(api_key=GROQ_API_KEY)
gemini_client = google_genai.Client(api_key=GEMINI_API_KEY)

print("⟳ 임베딩 모델 로드 중... (첫 실행 시 1~2분 소요)")
embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("✅ 임베딩 모델 로드 완료\n")

# ════════════════════════════════════════════════════════════════
#  ③ 질문 파일 읽기
# ════════════════════════════════════════════════════════════════

def load_questions():
    print(f"⟳ 질문 파일: {QUESTIONS_FILE}")
    if not os.path.exists(QUESTIONS_FILE):
        raise FileNotFoundError(
            f"\n❌ questions.csv 를 찾을 수 없습니다.\n"
            f"   찾는 경로: {QUESTIONS_FILE}\n"
            f"   geo_measure_free.py 와 같은 폴더에 questions.csv 가 있는지 확인하세요."
        )
    questions = []
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            q = row.get("question", "").strip()
            if q:
                questions.append(q)
    print(f"✅ 질문 {len(questions)}개 로드 완료")
    return questions

# ════════════════════════════════════════════════════════════════
#  ④ LLM 쿼리 (3개 모두 무료)
# ════════════════════════════════════════════════════════════════

def query_llama(q):
    try:
        r = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role":"user","content":q}],
            max_tokens=600, temperature=0.7)
        return r.choices[0].message.content or ""
    except Exception as e:
        print(f"  ⚠ Llama 오류: {e}"); return ""

def query_gemini(q):
    try:
        r = gemini_client.models.generate_content(
            model="gemini-1.5-flash", contents=q)
        return r.text or ""
    except Exception as e:
        print(f"  ⚠ Gemini 오류: {e}"); return ""

def query_llama_small(q):
    try:
        r = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role":"user","content":q}],
            max_tokens=600, temperature=0.7)
        return r.choices[0].message.content or ""
    except Exception as e:
        print(f"  ⚠ Llama-small 오류: {e}"); return ""

# ════════════════════════════════════════════════════════════════
#  ⑤ 인용 감지 / 감성 분석
# ════════════════════════════════════════════════════════════════

def detect_citation(text):
    low = text.lower()
    return any(kw.lower() in low for kw in BRAND_KEYWORDS)

def analyze_sentiment(text):
    prompt = (
        "아래 AI 답변에서 삼성생명·라이프놀로지랩이 언급된 맥락을 분석하세요.\n"
        "브랜드가 얼마나 긍정적·혁신적으로 묘사되는지 0.0~1.0 점수만 출력하세요.\n"
        "(0.0=부정/보수적, 0.5=중립, 1.0=긍정/혁신적) 예) 0.82\n\n"
        f"답변:\n{text[:800]}"
    )
    try:
        r = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role":"user","content":prompt}],
            max_tokens=10, temperature=0)
        return round(float(r.choices[0].message.content.strip()), 3)
    except Exception as e:
        print(f"  ⚠ 감성 분석 오류: {e}"); return 0.5

# ════════════════════════════════════════════════════════════════
#  ⑥ 시맨틱 점수 (로컬 — 완전 무료)
# ════════════════════════════════════════════════════════════════

def compute_semantic():
    print("\n⟳ 시맨틱 점수 계산 중 (로컬)...")
    site_vec = embed_model.encode(WEBSITE_TEXT, convert_to_tensor=True)
    scores = {}
    for phrase in SEMANTIC_TARGETS:
        score = round(float(util.cos_sim(site_vec, embed_model.encode(phrase, convert_to_tensor=True))), 4)
        scores[phrase] = score
        print(f"  [{phrase}] → {score:.4f}")
    scores["평균"] = round(sum(scores.values()) / len(scores), 4)
    return scores

# ════════════════════════════════════════════════════════════════
#  ⑦ 메인
# ════════════════════════════════════════════════════════════════

def main():
    today = datetime.date.today()
    print("=" * 62)
    print("  라이프놀로지 랩 GEO 측정 스크립트  v3")
    print(f"  실행 일시 : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  작업 폴더 : {SCRIPT_DIR}")
    print("=" * 62)

    # 오늘 이미 측정했는지 확인 (중복 방지)
    if os.path.exists(ALL_CSV):
        with open(ALL_CSV, encoding="utf-8-sig") as f:
            existing = {r.get("date","") for r in csv.DictReader(f)}
        if today.isoformat() in existing:
            print(f"⚠  오늘({today}) 데이터가 이미 results_all.csv 에 있습니다.")
            print("   재측정하려면 results_all.csv 에서 오늘 날짜 행을 삭제하세요.")
            return

    questions = load_questions()
    models = [
        ("Llama-3.3-70B",   query_llama),
        ("Gemini-1.5-Flash", query_gemini),
        ("Llama-3.1-8B",    query_llama_small),
    ]

    rows, done = [], 0
    total = len(questions) * len(models)

    for qi, question in enumerate(questions, 1):
        print(f"\n[질문 {qi}/{len(questions)}] {question[:55]}...")
        for model_name, fn in models:
            done += 1
            print(f"  ({done}/{total}) {model_name}...", end=" ", flush=True)
            resp     = fn(question)
            cited    = detect_citation(resp)
            sentiment = analyze_sentiment(resp) if cited else 0.0
            print(f"인용={'YES ✓' if cited else 'NO  ✗'}  감성={sentiment:.2f}")
            rows.append({
                "date": today.isoformat(), "model": model_name,
                "question": question, "cited": "TRUE" if cited else "FALSE",
                "sentiment": sentiment, "response": resp[:400].replace("\n"," "),
            })
            time.sleep(1.2)

    # results_all.csv 에 누적 추가
    fieldnames = ["date","model","question","cited","sentiment","response"]
    file_exists = os.path.exists(ALL_CSV)
    with open(ALL_CSV, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)
    print(f"\n✅ results_all.csv 에 추가 완료 ({len(rows)}행) → {ALL_CSV}")

    # 시맨틱 점수
    sem = compute_semantic()
    sem_path = os.path.join(SCRIPT_DIR, f"semantic_{today.strftime('%Y%m%d')}.json")
    with open(sem_path, "w", encoding="utf-8") as f:
        json.dump(sem, f, ensure_ascii=False, indent=2)
    print(f"✅ 시맨틱 점수 저장 → {sem_path}")

    # 요약
    cited_rows = [r for r in rows if r["cited"]=="TRUE"]
    sents = [r["sentiment"] for r in cited_rows]
    print("\n" + "=" * 62)
    print("  오늘의 측정 결과")
    print("=" * 62)
    print(f"  AI 인용률   : {len(cited_rows)/len(rows)*100:.1f}%  ({len(cited_rows)}/{len(rows)})")
    print(f"  평균 감성   : {sum(sents)/len(sents):.3f}" if sents else "  평균 감성   : 0.000")
    print(f"  시맨틱 평균 : {sem.get('평균','N/A')}")
    for m,_ in models:
        mr = [r for r in rows if r["model"]==m]
        mc = sum(1 for r in mr if r["cited"]=="TRUE")
        print(f"  {m:22s}: {mc/len(mr)*100:.1f}%" if mr else f"  {m}: -")
    print("=" * 62)

if __name__ == "__main__":
    main()
