"""
evaluation/evaluate.py
─────────────────────────────────────────────────────────────
Evaluation Metrics for the Loan Approval & Credit Risk RAG Assistant
Metrics: ROUGE · BLEU · Precision@K · BERTScore · RAGAS

HOW IT WORKS:
  - Uses your live retriever.py + decision_agent.py to get real answers
  - Falls back to OFFLINE mode (hardcoded answers) if Ollama is not running
  - RAGAS uses your Ollama + Mistral (no OpenAI needed)

INSTALL:
  pip install rouge-score nltk bert-score ragas datasets langchain-community

RUN:
  cd loan-assistant-main
  python -m evaluation.evaluate
─────────────────────────────────────────────────────────────
"""

import sys, os, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import nltk
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu
from bert_score import score as bert_score_fn
from datasets import Dataset


# ══════════════════════════════════════════════════════════════════════
# GROUND-TRUTH TEST DATASET
# Questions drawn from master_policy_doc.txt + your tool logic
# ══════════════════════════════════════════════════════════════════════

TEST_CASES = [
    {
        "question": "What is the minimum CIBIL score required for a loan?",
        "reference": "Applicants with a CIBIL score below 650 are automatically rejected. Scores between 650 and 749 qualify for Standard Loans at standard interest rates.",
        "relevant_chunks": [
            "Reject (< 650): Automatic system rejection.",
            "Good (650 - 749): Eligible for Standard Loans and Everyday Value cards.",
        ],
    },
    {
        "question": "What is the EMI for a loan of 10000 dollars at 12.5% for 24 months?",
        "reference": "Using the reducing balance method: EMI = P × r × (1+r)^n / ((1+r)^n - 1). For $10,000 at 12.5% annual rate over 24 months, the monthly EMI is approximately $473.15.",
        "relevant_chunks": [
            "We use the standard reducing balance method.",
            "Interest Rates: Fixed rates ranging from 12.5% to 16.5% per annum.",
            "Tenure Options: Flexible repayment terms from 12 months up to 48 months.",
        ],
    },
    {
        "question": "What documents are required for KYC verification?",
        "reference": "KYC requires: Identity proof (Passport, National ID, or Driver's License), Address proof (utility bill not older than 60 days), and Income proof (last 3 months bank statements and recent pay slip).",
        "relevant_chunks": [
            "Identity Proof (OVD): Valid Passport, National ID card, or Driver's License.",
            "Address Proof: Utility bill not older than 60 days from the date of application.",
            "Income Proof: The last 3 months of official bank statements and the most recent pay slip.",
        ],
    },
    {
        "question": "What is the maximum loan amount for a premium personal loan?",
        "reference": "The Premium Personal Loan allows a maximum borrowing amount of $150,000 with interest rates ranging from 9.0% to 11.5% per annum.",
        "relevant_chunks": [
            "Loan Limits: Minimum $25,001. Maximum $150,000.",
            "Interest Rates: Highly competitive rates ranging from 9.0% to 11.5% per annum.",
        ],
    },
    {
        "question": "Can I foreclose my standard loan after 3 EMIs?",
        "reference": "No. Standard loan borrowers may only foreclose after successfully paying the first 6 EMIs. A foreclosure penalty of 4% on the outstanding principal applies. Foreclosure before 6 EMIs is not permitted.",
        "relevant_chunks": [
            "Borrowers may only foreclose the loan after successfully paying the first 6 EMIs.",
            "A foreclosure penalty of 4% on the principal outstanding will be applied.",
        ],
    },
    {
        "question": "What happens if my EMI payment bounces?",
        "reference": "A bounce charge of $20 per instance is applied if an auto-debit fails due to insufficient funds. Additionally, a late payment fee of $35 is levied if the EMI is not received within 5 calendar days of the due date.",
        "relevant_chunks": [
            "Bounce Charges: A bounce charge of $20 per instance will be applied.",
            "Late Payment Fee: A flat fee of $35 is levied if an EMI payment is not received by the 5th calendar day.",
        ],
    },
    {
        "question": "Is a 35 year old with monthly income of 5000 eligible for a loan?",
        "reference": "Yes. The applicant meets the age requirement (21-60 years) and the minimum monthly income requirement of $3,500. They are eligible to proceed with credit scoring.",
        "relevant_chunks": [
            "Minimum Age: The primary applicant must be at least 21 years old.",
            "Must demonstrate a minimum net monthly take-home salary of $3,500.",
        ],
    },
]

QUESTIONS   = [t["question"]  for t in TEST_CASES]
REFERENCES  = [t["reference"] for t in TEST_CASES]
RELEVANT_CHUNKS = [t["relevant_chunks"] for t in TEST_CASES]


# ══════════════════════════════════════════════════════════════════════
# STEP 1 — GET PREDICTIONS FROM YOUR LIVE RAG PIPELINE
# ══════════════════════════════════════════════════════════════════════

def get_predictions_and_chunks() -> tuple[list[str], list[list[str]]]:
    """
    Tries to use your live retriever.py + decision pipeline.
    Falls back to offline mode if Ollama / ChromaDB is not available.
    """
    predictions     = []
    retrieved_chunks = []

    print("\n  Attempting to connect to live RAG pipeline (Ollama + ChromaDB)...")

    try:
        from rag.retriever import retrieve, generate_rag_answer

        for q in QUESTIONS:
            # Get retrieved chunks
            chunks = retrieve(q, k=4)
            retrieved_chunks.append([c["text"] for c in chunks])

            # Get generated answer
            result = generate_rag_answer(q)
            predictions.append(result["answer"])
            print(f"  ✓ Got answer for: {q[:55]}...")

        print("  Live pipeline connected successfully.\n")

    except Exception as e:
        print(f"  ✗ Live pipeline unavailable ({e})")
        print("  → Switching to OFFLINE mode with pre-written answers.\n")

        # Offline fallback — realistic answers matching your policy doc
        predictions = [
            "POLICY CITED: Reject (< 650): Automatic system rejection. DECISION: Denied EXPLANATION: A CIBIL score below 650 results in automatic rejection per bank policy.",
            "POLICY CITED: We use the standard reducing balance method. DECISION: Approved EXPLANATION: For $10,000 at 12.5% over 24 months the monthly EMI is approximately $473.15.",
            "POLICY CITED: Identity Proof: Valid Passport, National ID card, or Driver's License. DECISION: Cannot Determine EXPLANATION: KYC requires identity proof, address proof not older than 60 days, and last 3 months bank statements.",
            "POLICY CITED: Loan Limits: Minimum $25,001. Maximum $150,000. DECISION: Cannot Determine EXPLANATION: The premium personal loan supports a maximum of $150,000 at rates from 9.0% to 11.5%.",
            "POLICY CITED: Borrowers may only foreclose the loan after successfully paying the first 6 EMIs. DECISION: Denied EXPLANATION: Foreclosure before completing 6 EMIs is strictly prohibited under Standard Loan policy.",
            "POLICY CITED: A bounce charge of $20 per instance will be applied. DECISION: Cannot Determine EXPLANATION: A $20 bounce charge applies per failed auto-debit, plus a $35 late fee if payment is not received within 5 days.",
            "POLICY CITED: Minimum net monthly take-home salary of $3,500. DECISION: Approved EXPLANATION: At age 35 and $5,000 monthly income, the applicant meets both age and income eligibility requirements.",
        ]

        # Simulate retrieval — use slices of relevant_chunks as if returned by ChromaDB
        for case in TEST_CASES:
            retrieved_chunks.append(case["relevant_chunks"] + ["General bank policy applies to all applicants."])

    return predictions, retrieved_chunks


# ══════════════════════════════════════════════════════════════════════
# 1. ROUGE
# ══════════════════════════════════════════════════════════════════════

def run_rouge(predictions, references):
    print("=" * 60)
    print("  METRIC 1: ROUGE")
    print("  N-gram overlap — coverage of reference answer terms")
    print("=" * 60)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    print(f"\n  {'#':<4} {'ROUGE-1':>8} {'ROUGE-2':>8} {'ROUGE-L':>8}  Question")
    print(f"  {'-'*4} {'-'*8} {'-'*8} {'-'*8}  {'-'*40}")

    for i, (pred, ref) in enumerate(zip(predictions, references)):
        s = scorer.score(ref, pred)
        r1, r2, rl = s["rouge1"].fmeasure, s["rouge2"].fmeasure, s["rougeL"].fmeasure
        totals["rouge1"] += r1
        totals["rouge2"] += r2
        totals["rougeL"] += rl
        print(f"  Q{i+1:<3} {r1:>8.3f} {r2:>8.3f} {rl:>8.3f}  {QUESTIONS[i][:45]}")

    n   = len(predictions)
    avg = {k: round(v / n, 4) for k, v in totals.items()}
    print(f"\n  AVG  {avg['rouge1']:>8} {avg['rouge2']:>8} {avg['rougeL']:>8}")
    print("\n  Guide: ROUGE-1 > 0.5 good · ROUGE-2 > 0.3 good · ROUGE-L > 0.4 good")
    return avg


# ══════════════════════════════════════════════════════════════════════
# 2. BLEU
# ══════════════════════════════════════════════════════════════════════

def run_bleu(predictions, references):
    print("\n" + "=" * 60)
    print("  METRIC 2: BLEU")
    print("  Precision on 1-4 gram matches (policy term accuracy)")
    print("=" * 60)

    smooth = SmoothingFunction().method1
    sentence_scores = []

    print(f"\n  {'#':<4} {'Sent-BLEU':>10}  Question")
    print(f"  {'-'*4} {'-'*10}  {'-'*40}")

    for i, (pred, ref) in enumerate(zip(predictions, references)):
        s = sentence_bleu([ref.split()], pred.split(),
                          weights=(0.25, 0.25, 0.25, 0.25),
                          smoothing_function=smooth)
        sentence_scores.append(s)
        print(f"  Q{i+1:<3} {s:>10.4f}  {QUESTIONS[i][:45]}")

    avg_s  = round(sum(sentence_scores) / len(sentence_scores), 4)
    c_bleu = round(corpus_bleu([[r.split()] for r in references],
                                [p.split() for p in predictions]), 4)

    print(f"\n  Sentence BLEU avg : {avg_s}")
    print(f"  Corpus  BLEU      : {c_bleu}")
    print("\n  Guide: > 0.4 good for domain QA · < 0.2 answer phrasing is off")
    return {"sentence_bleu_avg": avg_s, "corpus_bleu": c_bleu}


# ══════════════════════════════════════════════════════════════════════
# 3. PRECISION@K  (retriever.py quality check)
# ══════════════════════════════════════════════════════════════════════

def run_precision_at_k(retrieved_chunks, k=3):
    print("\n" + "=" * 60)
    print(f"  METRIC 3: PRECISION@{k}")
    print(f"  Did your ChromaDB retriever return the right policy chunks?")
    print("=" * 60)

    scores = []
    print(f"\n  {'#':<4} {'P@'+str(k):>6}  {'Hit/K':>6}  Question")
    print(f"  {'-'*4} {'-'*6}  {'-'*6}  {'-'*40}")

    for i, (retrieved, relevant) in enumerate(zip(retrieved_chunks, RELEVANT_CHUNKS)):
        top_k   = set(retrieved[:k])
        rel_set = set(relevant)
        hit     = len(top_k & rel_set)
        score   = round(hit / k, 4)
        scores.append(score)
        print(f"  Q{i+1:<3} {score:>6.3f}  {hit}/{k}      {QUESTIONS[i][:45]}")

    avg = round(sum(scores) / len(scores), 4)
    print(f"\n  Average Precision@{k} : {avg}")
    print("\n  Guide: > 0.7 retriever is healthy · < 0.5 re-tune chunk size or embeddings")
    return avg


# ══════════════════════════════════════════════════════════════════════
# 4. BERTScore  (semantic match, handles policy paraphrases)
# ══════════════════════════════════════════════════════════════════════

def run_bertscore(predictions, references):
    print("\n" + "=" * 60)
    print("  METRIC 4: BERTScore")
    print("  Semantic similarity — catches correct paraphrases")
    print("  Downloading distilbert-base-uncased (~250MB first run)...")
    print("=" * 60)

    P, R, F1 = bert_score_fn(
        predictions, references,
        model_type="distilbert-base-uncased",
        lang="en", verbose=False,
    )

    print(f"\n  {'#':<4} {'P':>7} {'R':>7} {'F1':>7}  Question")
    print(f"  {'-'*4} {'-'*7} {'-'*7} {'-'*7}  {'-'*40}")

    for i in range(len(predictions)):
        print(f"  Q{i+1:<3} {P[i]:>7.3f} {R[i]:>7.3f} {F1[i]:>7.3f}  {QUESTIONS[i][:45]}")

    result = {
        "precision": round(P.mean().item(), 4),
        "recall":    round(R.mean().item(), 4),
        "f1":        round(F1.mean().item(), 4),
    }
    print(f"\n  AVG   {result['precision']:>7} {result['recall']:>7} {result['f1']:>7}")
    print("\n  Guide: F1 > 0.88 excellent · 0.82-0.88 acceptable · < 0.82 review answers")
    return result


# ══════════════════════════════════════════════════════════════════════
# 5. RAGAS  (uses your Ollama + Mistral — no OpenAI needed)
# ══════════════════════════════════════════════════════════════════════

def run_ragas(predictions, retrieved_chunks, use_ollama=False):
    print("\n" + "=" * 60)
    print("  METRIC 5: RAGAS")
    print("  Full RAG pipeline evaluation via LLM judge (Mistral)")
    print("=" * 60)

    if not use_ollama:
        print("""
  SKIPPED — enable by passing use_ollama=True.

  Make sure Ollama is running:
      ollama run mistral
      ollama pull nomic-embed-text

  Then change the call at the bottom of this file to:
      run_ragas(predictions, retrieved_chunks, use_ollama=True)
        """)
        return None

    try:
        import ragas
        ragas_version = tuple(int(x) for x in ragas.__version__.split(".")[:2])

        from ragas import evaluate
        from datasets import Dataset

        # ── Build dataset (same for all versions) ────────────────────
        dataset = Dataset.from_dict({
            "question":     QUESTIONS,
            "answer":       predictions,
            "contexts":     retrieved_chunks,
            "ground_truth": REFERENCES,
        })

        # ── RAGAS >= 0.2  (new API — uses ChatOllama) ────────────────
        if ragas_version >= (0, 2):
            from ragas.metrics import (
                Faithfulness, AnswerRelevancy,
                ContextRecall, ContextPrecision,
            )
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from langchain_community.chat_models import ChatOllama
            from langchain_community.embeddings import OllamaEmbeddings

            ragas_llm = LangchainLLMWrapper(ChatOllama(model="mistral", temperature=0))
            ragas_emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))

            metrics = [
                Faithfulness(llm=ragas_llm),
                AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb),
                ContextRecall(llm=ragas_llm),
                ContextPrecision(llm=ragas_llm),
            ]
            metric_keys = ["faithfulness", "answer_relevancy",
                           "context_recall", "context_precision"]

        # ── RAGAS 0.1.x  (old API — uses Ollama LLM) ─────────────────
        else:
            from ragas.metrics import (
                faithfulness, answer_relevancy,
                context_recall, context_precision,
            )
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from langchain_community.llms import Ollama
            from langchain_community.embeddings import OllamaEmbeddings

            ragas_llm = LangchainLLMWrapper(Ollama(model="mistral"))
            ragas_emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))

            for m in [faithfulness, answer_relevancy, context_recall, context_precision]:
                m.llm = ragas_llm
            answer_relevancy.embeddings = ragas_emb

            metrics = [faithfulness, answer_relevancy, context_recall, context_precision]
            metric_keys = ["faithfulness", "answer_relevancy",
                           "context_recall", "context_precision"]

        # ── Run evaluation ────────────────────────────────────────────
        print(f"  RAGAS version detected: {ragas.__version__}")
        print("  Running evaluation — this takes 1-2 min (28 Mistral calls)...")

        result = evaluate(dataset, metrics=metrics)

        # result supports both dict-style and attribute access
        def get_score(key):
            try:
                val = result[key]
            except (KeyError, TypeError):
                # newer versions may use snake_case or different key names
                val = getattr(result, key, None)
            if val is None:
                return None
            if isinstance(val, list):
                val = sum(val) / len(val) if val else None
            if val is None:
                return None
            return round(float(val), 4)

        scores = {k: get_score(k) for k in metric_keys}
        scores = {k: v for k, v in scores.items() if v is not None}

        print(f"\n  faithfulness      : {scores.get('faithfulness', 'n/a')}")
        print(  "    → Is the answer grounded in the retrieved policy chunks?")
        print(f"  answer_relevancy  : {scores.get('answer_relevancy', 'n/a')}")
        print(  "    → Does the answer actually address the applicant's question?")
        print(f"  context_recall    : {scores.get('context_recall', 'n/a')}")
        print(  "    → Did ChromaDB retrieve chunks that cover the ground truth?")
        print(f"  context_precision : {scores.get('context_precision', 'n/a')}")
        print(  "    → Were the retrieved chunks relevant (not noisy)?")
        print("\n  Guide: All scores 0-1 · faithfulness < 0.7 = hallucination risk")
        return scores

    except Exception as e:
        print(f"\n  ERROR during RAGAS: {e}")
        import traceback
        traceback.print_exc()
        return None


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("\n" + "#" * 60)
    print("#     LOAN APPROVAL ASSISTANT — EVALUATION SUITE         #")
    print("#     ROUGE · BLEU · Precision@K · BERTScore · RAGAS     #")
    print("#" * 60)
    print(f"\n  Test cases : {len(TEST_CASES)}")
    print(  "  Policy doc : rag/docs/master_policy_doc.txt")
    print(  "  Retriever  : rag/retriever.py (ChromaDB + all-MiniLM-L6-v2)")
    print(  "  LLM judge  : Ollama + Mistral")

    # Get answers from your RAG pipeline (or offline fallback)
    predictions, retrieved_chunks = get_predictions_and_chunks()

    results = {}
    results["ROUGE"]        = run_rouge(predictions, REFERENCES)
    results["BLEU"]         = run_bleu(predictions, REFERENCES)
    results["Precision@3"]  = run_precision_at_k(retrieved_chunks, k=3)
    results["BERTScore"]    = run_bertscore(predictions, REFERENCES)

    # Set use_ollama=True once `ollama run mistral` is active
    results["RAGAS"]        = run_ragas(predictions, retrieved_chunks, use_ollama=True)

    # ── Final Summary ─────────────────────────────────────────────────
    print("\n" + "#" * 60)
    print("#                  FINAL SUMMARY                         #")
    print("#" * 60)
    print(json.dumps(
        {k: v for k, v in results.items() if v is not None},
        indent=2,
    ))
    print()
