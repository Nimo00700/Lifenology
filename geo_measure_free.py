#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
라이프놀로지 랩 — GEO 측정 스크립트 (완전 무료 버전)
Samsung Life x Lifenology Lab 3기

사용 서비스 (전부 $0):
  Groq API  → Llama 3.3 70B / Gemma 2 9B  (ChatGPT·Claude 유료 대체)
  Google Gemini → Gemini 1.5 Flash          (무료 티어 1,500회/일)
  sentence-transformers → 임베딩 로컬 실행   (OpenAI 임베딩 유료 대체)

패키지 설치 (터미널에 복붙):
  pip install groq google-generativeai sentence-transformers
"""

import csv, json, math, time, datetime
from groq import Groq
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util


# ════════════════════════════════════════════════════════════════
#  ① 설정  ←  여기만 수정하면 됩니다
# ════════════════════════════════════════════════════════════════

GROQ_API_KEY   = "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"    # https://console.groq.com  (무료, 카드 불필요)
GEMINI_API_KEY = "AQ.Ab8RN6LR4NMhxaRc7PI4u_oolpOr0AJb1txdASQeuNJ5hXLWYQ"    # https://aistudio.google.com (구글 계정만 있으면 무료)

# 브랜드 감지 키워드 — 하나라도 포함되면 "인용됨"으로 판단
BRAND_KEYWORDS = [
    "삼성생명",
    "라이프놀로지랩",
    "라이프놀로지 랩",
    "lifenology",
    "Lifenology",
    "보험을 넘어서는 보험",
]

# 웹사이트·프로젝트 핵심 소개 텍스트 (시맨틱 유사도 측정 기준)
# 웹사이트 완성 후 About 페이지 텍스트로 교체하세요
WEBSITE_TEXT = """
라이프놀로지 랩은 삼성생명이 대학생들과 함께 '보험을 넘어서는 보험'이라는
브랜드 메시지를 실천하며 더 나은 내일을 위한 혁신 솔루션을 개발하는 산학협력 프로젝트입니다.
1기·2기·3기를 통해 헬스케어, 테크, 라이프스타일 분야의 혁신적인 아이디어를 발굴하고 실현합니다.
삼성생명은 단순한 보험 상품을 넘어 고객의 삶 전반을 지원하는 종합금융플랫폼으로서 혁신적 솔루션을 추구합니다.
"""

# 시맨틱 유사도를 측정할 타깃 키워드 문장
SEMANTIC_TARGETS = [
    "보험 혁신 솔루션",
    "대학생 산학협력 헬스케어 프로그램",
    "미래 라이프스타일 기술 혁신",
    "혁신적인 보험 브랜드 신뢰",
    "지속가능한 금융 라이프 솔루션",
]

QUESTIONS_FILE = "questions.csv"


# ════════════════════════════════════════════════════════════════
#  ② 클라이언트 초기화 (수정 불필요)
# ════════════════════════════════════════════════════════════════

groq_client = Groq(api_key=GROQ_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

print("⟳ 임베딩 모델 로드 중... (첫 실행 시 1~2분 소요, 이후엔 빠름)")
embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("✅ 임베딩 모델 로드 완료\n")


# ════════════════════════════════════════════════════════════════
#  ③ 질문 파일 읽기
# ════════════════════════════════════════════════════════════════

def load_questions(filepath):
    questions = []
    with open(filepath, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row.get("question", "").strip()
            if q:
                questions.append(q)
    print(f"✅ 질문 {len(questions)}개 로드 완료")
    return questions


# ════════════════════════════════════════════════════════════════
#  ④ LLM 쿼리 (모두 무료)
# ════════════════════════════════════════════════════════════════

def query_llama(question):
    """Llama 3.3 70B via Groq — 무료, GPT-4 수준 성능"""
    try:
        r = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": question}],
            max_tokens=600, temperature=0.7,
        )
        return r.choices[0].message.content or ""
    except Exception as e:
        print(f"  ⚠ Llama 오류: {e}"); return ""

def query_gemini(question):
    """Gemini 1.5 Flash — 무료 티어 1,500회/일"""
    try:
        r = gemini_model.generate_content(question)
        return r.text or ""
    except Exception as e:
        print(f"  ⚠ Gemini 오류: {e}"); return ""

def query_gemma(question):
    """Gemma 2 9B via Groq — 무료"""
    try:
        r = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": question}],
            max_tokens=600, temperature=0.7,
        )
        return r.choices[0].message.content or ""
    except Exception as e:
        print(f"  ⚠ Gemma 오류: {e}"); return ""


# ════════════════════════════════════════════════════════════════
#  ⑤ 인용 여부 감지
# ════════════════════════════════════════════════════════════════

def detect_citation(text):
    low = text.lower()
    return any(kw.lower() in low for kw in BRAND_KEYWORDS)


# ════════════════════════════════════════════════════════════════
#  ⑥ 브랜드 감성 분석 (Groq 재호출 — 무료)
# ════════════════════════════════════════════════════════════════

def analyze_sentiment(text):
    prompt = f"""아래 AI 답변에서 '삼성생명' 또는 '라이프놀로지랩'이 언급된 맥락을 분석하세요.
브랜드가 얼마나 긍정적·혁신적 이미지로 묘사되는지 0.0~1.0 점수로만 답하세요.
(0.0=매우 부정/보수적, 0.5=중립, 1.0=매우 긍정/혁신적)
숫자 하나만 출력하세요. 예) 0.82

답변 텍스트:
{text[:800]}
"""
    try:
        r = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10, temperature=0,
        )
        return round(float(r.choices[0].message.content.strip()), 3)
    except Exception as e:
        print(f"  ⚠ 감성 분석 오류: {e}"); return 0.5


# ════════════════════════════════════════════════════════════════
#  ⑦ 시맨틱 연관성 점수 — 로컬 임베딩 (완전 무료)
# ════════════════════════════════════════════════════════════════

def compute_semantic_scores():
    print("\n⟳ 시맨틱 점수 계산 중 (로컬 실행, 무료)...")
    site_vec = embed_model.encode(WEBSITE_TEXT, convert_to_tensor=True)
    scores = {}
    for phrase in SEMANTIC_TARGETS:
        kw_vec = embed_model.encode(phrase, convert_to_tensor=True)
        score  = round(float(util.cos_sim(site_vec, kw_vec)), 4)
        scores[phrase] = score
        print(f"  [{phrase}] → {score:.4f}")
    scores["평균"] = round(sum(scores.values()) / len(scores), 4)
    return scores


# ════════════════════════════════════════════════════════════════
#  ⑧ 메인 실행
# ════════════════════════════════════════════════════════════════

def main():
    today   = datetime.date.today()
    out_csv = f"results_{today.strftime('%Y%m%d')}.csv"
    out_sem = f"semantic_{today.strftime('%Y%m%d')}.json"

    print("=" * 62)
    print("  라이프놀로지 랩 GEO 측정 스크립트  [완전 무료]")
    print(f"  실행 일시: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 62)

    questions = load_questions(QUESTIONS_FILE)

    models = [
        ("Llama-3.3-70B",    query_llama),
        ("Gemini-1.5-Flash",  query_gemini),
        ("Gemma-2-9B",        query_gemma),
    ]

    rows  = []
    total = len(questions) * len(models)
    done  = 0

    for qi, question in enumerate(questions, 1):
        print(f"\n[질문 {qi}/{len(questions)}] {question[:55]}...")

        for model_name, query_fn in models:
            done += 1
            print(f"  ({done}/{total}) {model_name}...", end=" ", flush=True)

            response  = query_fn(question)
            cited     = detect_citation(response)
            sentiment = analyze_sentiment(response) if cited else 0.0

            print(f"인용={'YES ✓' if cited else 'NO  ✗'}  감성={sentiment:.2f}")

            rows.append({
                "date":      today.isoformat(),
                "model":     model_name,
                "question":  question,
                "cited":     "TRUE" if cited else "FALSE",
                "sentiment": sentiment,
                "response":  response[:400].replace("\n", " "),
            })

            time.sleep(1.2)   # Groq 무료 속도 제한 방지 (30 RPM)

    # CSV 저장
    fieldnames = ["date", "model", "question", "cited", "sentiment", "response"]
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n✅ 결과 저장 → {out_csv}")

    # 시맨틱 점수 JSON
    sem_scores = compute_semantic_scores()
    with open(out_sem, "w", encoding="utf-8") as f:
        json.dump(sem_scores, f, ensure_ascii=False, indent=2)
    print(f"✅ 시맨틱 점수 저장 → {out_sem}")

    # 요약
    cited_rows = [r for r in rows if r["cited"] == "TRUE"]
    sents      = [r["sentiment"] for r in cited_rows]
    cit_rate   = round(len(cited_rows) / len(rows) * 100, 1)
    avg_sent   = round(sum(sents) / len(sents), 3) if sents else 0.0

    print("\n" + "=" * 62)
    print("  오늘의 측정 결과 요약")
    print("=" * 62)
    print(f"  총 쿼리 수      : {len(rows)} 회")
    print(f"  AI 인용률       : {cit_rate}%  ({len(cited_rows)}/{len(rows)})")
    print(f"  평균 감성 지수   : {avg_sent}")
    print(f"  시맨틱 연관성   : {sem_scores.get('평균', 'N/A')}")
    print("=" * 62)
    for m_name, _ in models:
        m_rows  = [r for r in rows if r["model"] == m_name]
        m_cited = sum(1 for r in m_rows if r["cited"] == "TRUE")
        m_rate  = round(m_cited / len(m_rows) * 100, 1) if m_rows else 0
        print(f"    {m_name:22s}: {m_rate}%  ({m_cited}/{len(m_rows)})")
    print("=" * 62)
    print(f"\n완료! {out_csv}  /  {out_sem}")


if __name__ == "__main__":
    main()
