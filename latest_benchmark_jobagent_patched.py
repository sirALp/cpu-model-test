#!/usr/bin/env python
"""
Single-step parameter extraction benchmark for read_sql job_agent with qwen3 via Ollama.

Amaç:
- Router bir soru soruyor
- User cevap veriyor
- LLM, bu soruya verilen cevaptan doğru param(lar)ı çıkarıp JSON döndürüyor

Bu script:
- Birçok test case tanımlar (soru + user cevabı + mevcut paramlar + beklenen güncelleme)
- Seçilen modele (qwen3:1.7b veya qwen3:8b) her test case'i ayrı ayrı gönderir
- Prompt formatı danışmanının job_agent log’undaki gibidir:

  Extract params...
  ...
  Last question: "..."
  User answer: "..."
  Current: {...}
  Missing: ...
  Output JSON only:

- Dönen JSON'u parse eder, param güncellemelerini değerlendirir
- time_to_first_token ve total süreyi ölçer
- Sonuçları detaylı bir log dosyasına yazar ve global özet üretir

Not: action ("ASK"/"TOOL") sadece loglanır, başarı kriterine dahil edilmez.
"""

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# ChatOllama import: önce langchain_ollama, yoksa langchain_community
try:
    from langchain_ollama import ChatOllama
except ImportError:  # pragma: no cover
    from langchain_community.chat_models import ChatOllama

from langchain_core.messages import SystemMessage, HumanMessage


# ---------------------------------------------------------------------------
# SABİTLER & SPEC
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are a STRICT JSON parameter extraction assistant used inside a router.

Your ONLY job on each call:

1. Read the "Last question", "User answer" and "Current" param state.
2. Understand which single parameter the last question is asking about.
3. Extract the value for ONLY THAT parameter from the user answer.
4. Return a single JSON object with:
   {
     "action": "ASK" or "TOOL",      // value is NOT important for this benchmark
     "question": "...",              // optional, ignored in this benchmark
     "params": { ... }               // updated parameters
   }

IMPORTANT RULES (CRITICAL):

- Always start from the Current params:
  - Copy ALL keys and values from Current into "params" unchanged.
  - NEVER delete existing keys.
  - NEVER change an existing value in Current.
  - This benchmark assumes the user is NOT changing previous answers.

- Only add ONE new parameter per call, for the LAST question:
  - If you can confidently extract the value from the user answer:
    - Add or update exactly that one key in "params".
  - If you CANNOT confidently extract the value:
    - DO NOT guess.
    - DO NOT invent values like "string" or "boolean".
    - Just return the same params as Current (no new key).

- NEVER use type names or placeholders as values:
  - INVALID: "string", "varchar", "boolean", "bool", "int", "integer", "float", "double", "text".
  - If the only thing you see is a type name, treat it as "no value found" and do NOT set that param.

- BOOLEAN MAPPING (very important):
  - Some questions ask about boolean parameters: execute_query, write_count, drop_before_create.
  - For such yes/no questions:
    - Map these answers to TRUE:
      "yes", "yeah", "yep", "sure", "please do", "go ahead", "run it", "execute it", "do it"
    - Map these answers to FALSE:
      "no", "nope", "don't", "do not", "not needed", "skip it", "no, just show me the results"
  - For non-boolean questions (e.g. schema name, table name, connection):
    - Ignore filler words like "ok", "okay", "sure", "yes", "no" IF they are not part of the actual value.

- NAME extraction:
  - When the question is asking for the job name (e.g. "What should I name this read_sql job?"):
    - If the user answer is a simple identifier, use it directly:
      - User: "read1172"        ->  "name": "read1172"
    - If the answer is a phrase, extract only the meaningful identifier:
      - User: "Let's call it read1172"         -> "name": "read1172"
      - User: "Call this daily_orders job"     -> "name": "daily_orders"
      - User: 'Let\'s name it "job133"'        -> "name": "job133"
    - Prefer:
      - The last quoted string, if any.
      - Otherwise, the last token that looks like an identifier (letters, digits, underscore).

- SCHEMA / TABLE / CONNECTION extraction:
  - The value is almost always a single identifier:
    - Examples: "ICC_TEST", "DW", "test_table", "RESULT_2025", "ORACLE_10"
  - If the user gives a longer sentence, pick the identifier-like part:
    - User: 'Use schema ICC_TEST for results' -> "result_schema": "ICC_TEST"
    - User: 'let's use "RESULT_2025"'         -> "table_name": "RESULT_2025"

- Output format:
  - Return JSON ONLY.
  - NO Markdown, NO backticks, NO extra text before or after.
  - If you are unsure, it is always better to return only the Current params (no new key) than to guess.

Focus on producing correct "params". The "action" and "question" fields are ignored by the evaluator.
"""

READ_SQL_SPEC = """
Extract params for read_sql job.

You are inside a step-by-step router that collects parameters.
On each call, you see:
- The last question asked to the user ("Last question")
- The user's answer ("User answer")
- The current collected parameters ("Current")
Your job is to update ONE parameter according to the user's latest answer.

Required params:
- name: Job name
- execute_query: Save results to DB? (yes=true, no=false)
- write_count: Track row count? (yes=true, no=false)

IF execute_query=true, ALSO need:
- result_schema (string): Schema to write query results
- table_name (string): Table name to store query results
- drop_before_create (boolean): Drop table before creating? (yes=true, no=false)

IF write_count=true, ALSO need:
- write_count_connection (string): Connection for row count (default: same as query connection)
- write_count_schema (string): Schema for row count table
- write_count_table (string): Table name to store row count

BOOLEAN ANSWERS (IMPORTANT):
- For questions about execute_query, write_count or drop_before_create:
  - Treat "yes"/"yeah"/"yep"/"sure"/"please do"/"go ahead"/"run it"/"execute it"/"do it" as TRUE.
  - Treat "no"/"nope"/"don't"/"do not"/"not needed"/"skip it"/"no, just show me the results" as FALSE.

FILLER WORDS:
- For non-boolean questions (schema, table, connection, name):
  - Ignore filler words like "ok", "okay", "sure", "yes", "no" if they are not part of the actual value.

OUTPUT RULES:
- Only extract actual values from user input.
- Do NOT use type names as values (e.g. "string", "varchar", "boolean", "int", "float", "double", "text").
- Use the "Current" JSON as the base for your "params" and keep all its keys and values unchanged.
- Add or update only the parameter that the last question is asking about.
- If you cannot extract a value reliably, do NOT invent anything. Just return the same params as Current.

Output JSON: {"action": "ASK"|"TOOL", "question": "...", "params": {...}}
""".strip()

# Additional job specs (JobAgent logs)
WRITE_DATA_KEYS = ["name", "table", "schemas", "connection", "drop_or_truncate"]
WRITE_DATA_SPEC = """Extract params for write_data job.

Required params:
- name (string): Job name
- table (string): Target table name
- schemas (string): Target schema name
- connection (string): Target DB connection name
- drop_or_truncate (string): One of: drop, truncate, none
""".strip()

SEND_EMAIL_KEYS = ["name", "to", "subject", "text", "cc"]
SEND_EMAIL_SPEC = """Extract params for send_email job.

Required params:
- name (string): Job name
- to (string): Receiver email address
- subject (string): Email subject
- text (string): Email body text
- cc (string): CC email(s), optional
""".strip()


TYPE_NAME_STRINGS = {
    "string",
    "boolean",
    "bool",
    "integer",
    "int",
    "float",
    "double",
    "varchar",
    "text",
}

# Tüm olası param isimleri (Missing satırını doldurmak için)
ALL_PARAM_KEYS = [
    "name",
    "execute_query",
    "write_count",
    "result_schema",
    "table_name",
    "drop_before_create",
    "write_count_connection",
    "write_count_schema",
    "write_count_table",
]


# ---------------------------------------------------------------------------
# DATA CLASS'LAR
# ---------------------------------------------------------------------------

@dataclass
class ExtractionTestCase:
    tid: str
    description: str
    last_question: str
    user_answer: str
    current: Dict[str, object]
    expected_updates: Dict[str, object]  # bu step sonunda set edilmesini beklediğimiz key'ler

    # Prompt tarafı (farklı job'lar için)
    spec: str = ""  # boşsa default READ_SQL_SPEC kullanılır
    all_param_keys: Optional[List[str]] = None  # boşsa default ALL_PARAM_KEYS kullanılır

    expected_action: Optional[str] = None  # artık metasadece log için, success'te zorunlu değil

@dataclass
class ExtractionTestResult:
    tid: str
    description: str
    success: bool
    parse_error: bool
    error: bool
    type_name_bug: bool
    param_overwrite_bug: bool
    param_drop_bug: bool
    dropped_keys: List[str]

    updates_correct: int
    updates_total: int
    expected_action: Optional[str]
    got_action: Optional[str]
    time_to_first_token: float
    time_total: float
    raw_output: str
    merged_params: Dict[str, object]
    expected_updates: Dict[str, object]
    last_question: str
    user_answer: str
    current: Dict[str, object]
    user_prompt: str
# ---------------------------------------------------------------------------
# LLM YARDIMCI FONKSİYONLARI
# ---------------------------------------------------------------------------

def build_llm(model_size: str, think: bool) -> ChatOllama:
    """
    model_size: '8b' veya '1.7b'
    think: Qwen3'ün 'think' reasoning modu açık mı?
    """
    model_name = f"qwen3:{model_size}"
    print(f"[INFO] Initializing model={model_name}, think={think}")
    return ChatOllama(
        model=model_name,
        temperature=0.1,
        base_url="http://localhost:11434",
        num_predict=512,  # single-step JSON için yeterli, latency'i kısar
        timeout=60.0,
        keep_alive="3600s",
        model_kwargs={
            "think": think,
            "stream": True,
        },
    )


def safe_parse_json(text: str) -> Dict[str, object]:
    """
    - Önce direkt json.loads dener
    - Olmazsa ilk '{' ile son '}' arasını alıp tekrar dener
    - Hala olmazsa ValueError fırlatır
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty response from model")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            return json.loads(candidate)
        raise


def call_llm_with_timing(llm: ChatOllama, messages: List[object]) -> Tuple[str, float, float]:
    """
    (full_text, time_to_first_token, time_total) döndürür.
    Önce stream ile dener, hata alırsa invoke()'a düşer.
    """
    try:
        start = time.perf_counter()
        chunks: List[str] = []
        t_first: Optional[float] = None

        for chunk in llm.stream(messages):
            if t_first is None:
                t_first = time.perf_counter() - start

            text_piece = ""

            if hasattr(chunk, "text") and chunk.text:
                text_piece = chunk.text
            elif hasattr(chunk, "content") and chunk.content:
                if isinstance(chunk.content, str):
                    text_piece = chunk.content
                elif isinstance(chunk.content, list):
                    text_piece = "".join(
                        getattr(m, "content", "") for m in chunk.content
                    )

            if text_piece:
                chunks.append(text_piece)

        full_text = "".join(chunks).strip()
        t_total = time.perf_counter() - start
        if t_first is None:
            t_first = t_total

        return full_text, t_first, t_total

    except Exception as e:
        print(f"[WARN] Streaming failed ({e}), falling back to invoke().")
        start = time.perf_counter()
        resp = llm.invoke(messages)
        t_total = time.perf_counter() - start
        full_text = getattr(resp, "content", str(resp)).strip()
        return full_text, t_total, t_total


def build_user_prompt(case: ExtractionTestCase) -> str:
    """
    LLM'e gidecek USER PROMPT:

    Extract params...
    ...
    Last question: "..."
    User answer: "..."
    Current: {...}
    Missing: ...
    Output JSON only:
    """

    # Missing'i danışmanın örneğine benzetmek için:
    # Basitçe "şu an current'ta olmayan tüm olası paramlar" gibi hesaplıyoruz.
    keys_for_missing = case.all_param_keys or ALL_PARAM_KEYS
    missing_keys = [k for k in keys_for_missing if k not in case.current]
    missing_str = ", ".join(missing_keys) if missing_keys else "none"

    spec_text = (case.spec or READ_SQL_SPEC).strip()

    return f"""{spec_text}

Last question: "{case.last_question}"
User answer: "{case.user_answer}"
Current: {json.dumps(case.current, ensure_ascii=False)}
Missing: {missing_str}

Output JSON only:
""".strip()


# ---------------------------------------------------------------------------
# TEST CASE ÜRETİCİ
# ---------------------------------------------------------------------------


def _infer_job_spec_and_keys(last_question: str, current: Dict[str, object]) -> Tuple[str, List[str]]:
    """
    JobAgent log'larından gelen case'ler mixed olabilir (read_sql / write_data / send_email).
    Basit bir heuristic ile hangi spec/keys kullanılacağını seçiyoruz.
    """
    q = (last_question or "").lower()

    # email
    if "email" in q or "send the email" in q or "send an email" in q:
        return SEND_EMAIL_SPEC, SEND_EMAIL_KEYS

    # write_data
    if "write the data" in q or "drop" in q or "truncate" in q or "append" in q:
        return WRITE_DATA_SPEC, WRITE_DATA_KEYS

    # read_sql default
    return READ_SQL_SPEC, ALL_PARAM_KEYS


def _infer_target_key(last_question: str, keys: List[str]) -> Optional[str]:
    q = (last_question or "").lower()

    if "name" in q:
        return "name" if "name" in keys else None

    if any(kw in q for kw in ["save results", "save to db", "save to database", "execute query", "execute it", "run it"]):
        if "execute_query" in keys:
            return "execute_query"

    if any(kw in q for kw in ["row count", "track the row count", "track row count", "count rows"]):
        if "write_count" in keys:
            return "write_count"

    if "schema" in q:
        # choose schema-like key that exists
        for cand in ["result_schema", "schemas", "write_count_schema"]:
            if cand in keys:
                return cand

    if "table" in q:
        for cand in ["table_name", "table", "write_count_table"]:
            if cand in keys:
                return cand

    if any(kw in q for kw in ["drop", "truncate", "append", "none"]):
        for cand in ["drop_before_create", "drop_or_truncate"]:
            if cand in keys:
                return cand

    if "send" in q and "to" in q and "email" in q:
        if "to" in keys:
            return "to"

    if "subject" in q:
        if "subject" in keys:
            return "subject"

    if "text" in q or "body" in q:
        if "text" in keys:
            return "text"

    if "cc" in q:
        if "cc" in keys:
            return "cc"

    return None


def _extract_expected_value(key: str, user_answer: str, last_question: str) -> Optional[object]:
    ans = (user_answer or "").strip()

    # If blank, no value.
    if ans == "":
        return None

    # Booleans
    if key in ("execute_query", "write_count", "drop_before_create"):
        al = ans.lower()
        true_markers = ["yes", "yeah", "yep", "sure", "please do", "go ahead", "run it", "execute it", "do it"]
        false_markers = [
            "no", "nope", "nah", "don't", "do not", "not needed", "not necessary", "unnecessary",
            "no need", "not required", "skip", "omit", "disable", "turn off", "off", "preview only",
            "just show", "don't save", "do not save"
        ]
        if any(m in al for m in false_markers):
            return False
        if any(m in al for m in true_markers):
            return True
        return None

    # drop_or_truncate string enum
    if key == "drop_or_truncate":
        al = ans.lower()
        for opt in ["drop", "truncate", "none"]:
            if opt in al:
                return opt
        # sometimes only "append" is used -> treat as none
        if "append" in al:
            return "none"
        return None

    # email receiver
    if key == "to":
        m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", ans)
        return m.group(0) if m else None

    # generic string: last quoted or last identifier-ish token
    quoted = re.findall(r'"([^"]+)"', ans)
    if quoted:
        return quoted[-1].strip()

    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", ans)
    if tokens:
        # pick last token (usually the identifier)
        return tokens[-1]

    return None


def load_job_agent_cases_from_txt(folder: Path) -> List[ExtractionTestCase]:
    """
    /mnt/data/0001_job_agent.txt ... 0014_job_agent.txt gibi dosyaların alt kısmından
    Last question / User answer / Current bölümlerini okuyup test case üretir.
    """
    cases: List[ExtractionTestCase] = []
    for i in range(1, 15):
        fp = folder / f"{i:04d}_job_agent.txt"
        if not fp.exists():
            continue
        txt = fp.read_text(encoding="utf-8", errors="replace")

        # Best-effort parse
        q_m = re.search(r'Last question:\s*"([^"]+)"', txt)
        a_m = re.search(r'User answer:\s*"(.*)"\s*\nCurrent:', txt)
        c_m = re.search(r"Current:\s*(\{.*?\})", txt)

        last_q = q_m.group(1) if q_m else ""
        user_a = a_m.group(1) if a_m else ""
        current_raw = c_m.group(1) if c_m else "{}"
        try:
            current = json.loads(current_raw)
        except Exception:
            current = {}

        spec, keys = _infer_job_spec_and_keys(last_q, current)
        target_key = _infer_target_key(last_q, keys)

        expected_updates: Dict[str, object] = {}
        if target_key:
            val = _extract_expected_value(target_key, user_a, last_q)
            if val is not None:
                expected_updates[target_key] = val

        # Special: if last question missing, we expect no updates at all
        if not last_q.strip():
            expected_updates = {}

        cases.append(
            ExtractionTestCase(
                tid=f"J{i:04d}",
                description=f"JobAgent log file {fp.name}",
                last_question=last_q if last_q else "N/A (no last question provided)",
                user_answer=user_a,
                current=current,
                expected_updates=expected_updates,
                spec=spec,
                all_param_keys=keys,
            )
        )

    # JobAgent örneklerinden otomatik case ekle (0001_job_agent.txt ... 0014_job_agent.txt)
    try:
        cases.extend(load_job_agent_cases_from_txt(Path(__file__).resolve().parent))
    except Exception:
        pass

    return cases


def build_test_cases() -> List[ExtractionTestCase]:
    cases: List[ExtractionTestCase] = []

    # 1) NAME extraction testleri (20 adet)
    job_names = [
        "read1172",
        "job133",
        "daily_orders",
        "report_balances",
        "test_job",
        "nightly_etl",
        "job99",
        "rowsync",
        "customer_audit",
        "clicks_agg",
    ]
    q_name = "What should I name this read_sql job?"

    tid_counter = 1
    for name in job_names:
        # a) Düz isim
        cases.append(
            ExtractionTestCase(
                tid=f"T{tid_counter:03d}",
                description=f"name plain: {name}",
                last_question=q_name,
                user_answer=name,
                current={},
                expected_updates={"name": name},
                expected_action=None,  # action önemli değil, log için tutuluyor
            )
        )
        tid_counter += 1

        # b) Cümle içinde isim
        cases.append(
            ExtractionTestCase(
                tid=f"T{tid_counter:03d}",
                description=f"name phrase: Let's call it {name}",
                last_question=q_name,
                user_answer=f"Let's call it {name}",
                current={},
                expected_updates={"name": name},
                expected_action=None,
            )
        )
        tid_counter += 1

    # 2) execute_query boolean testleri (10 adet)
    q_exec = "Should the job execute the query and save results to the database? (yes/no)"
    yes_exec_answers = [
        "yes",
        "yeah, go ahead",
        "sure, execute it",
        "yep, run it and save",
        "please do",
    ]
    no_exec_answers = [
        "no",
        "no, just show me the results",
        "no, don't execute",
        "nope, preview only",
        "let's not run it",
    ]

    base_current = {"name": "job_exec_test"}

    for ans in yes_exec_answers:
        cases.append(
            ExtractionTestCase(
                tid=f"T{tid_counter:03d}",
                description=f"execute_query yes: {ans}",
                last_question=q_exec,
                user_answer=ans,
                current=base_current,
                expected_updates={"execute_query": True},
                expected_action=None,
            )
        )
        tid_counter += 1

    for ans in no_exec_answers:
        cases.append(
            ExtractionTestCase(
                tid=f"T{tid_counter:03d}",
                description=f"execute_query no: {ans}",
                last_question=q_exec,
                user_answer=ans,
                current=base_current,
                expected_updates={"execute_query": False},
                expected_action=None,
            )
        )
        tid_counter += 1

    # 3) write_count boolean testleri (10 adet)
    q_wc = "Should the job track the row count? (yes/no)"
    yes_wc_answers = [
        "yes",
        "yes, track it",
        "yeah, keep the row count",
        "please track rows",
        "sure, count rows",
    ]
    no_wc_answers = [
        "no",
        "no, I don't need row count",
        "nope, skip row count",
        "don't track it",
        "no, it's not necessary",
    ]
    base_current_wc = {"name": "job_wc_test", "execute_query": True}

    for ans in yes_wc_answers:
        cases.append(
            ExtractionTestCase(
                tid=f"T{tid_counter:03d}",
                description=f"write_count yes: {ans}",
                last_question=q_wc,
                user_answer=ans,
                current=base_current_wc,
                expected_updates={"write_count": True},
                expected_action=None,
            )
        )
        tid_counter += 1

    for ans in no_wc_answers:
        cases.append(
            ExtractionTestCase(
                tid=f"T{tid_counter:03d}",
                description=f"write_count no: {ans}",
                last_question=q_wc,
                user_answer=ans,
                current=base_current_wc,
                expected_updates={"write_count": False},
                expected_action=None,
            )
        )
        tid_counter += 1

    # 4) result_schema / table_name / drop_before_create / write_count_* string testleri (~9 adet)
    q_schema = "Which schema should I write the results to?"
    q_table = "What table should I write the results to?"
    q_drop = "Should I drop the table before creating it? (yes/no)"
    q_wc_conn = "What is the connection string for the row count?"
    q_wc_schema = "What is the schema name for the row count?"
    q_wc_table = "What is the table name to store the row count?"

    cases.extend(
        [
            ExtractionTestCase(
                tid=f"T{tid_counter:03d}",
                description="result_schema simple",
                last_question=q_schema,
                user_answer="ICC_TEST",
                current={"name": "job_full_1", "execute_query": True},
                expected_updates={"result_schema": "ICC_TEST"},
                expected_action=None,
            )
        ]
    )
    tid_counter += 1

    cases.extend(
        [
            ExtractionTestCase(
                tid=f"T{tid_counter:03d}",
                description="result_schema lowercased",
                last_question=q_schema,
                user_answer="dw",
                current={"name": "job_full_2", "execute_query": True},
                expected_updates={"result_schema": "dw"},
                expected_action=None,
            )
        ]
    )
    tid_counter += 1

    cases.extend(
        [
            ExtractionTestCase(
                tid=f"T{tid_counter:03d}",
                description="table_name simple",
                last_question=q_table,
                user_answer="test_table",
                current={
                    "name": "job_full_3",
                    "execute_query": True,
                    "result_schema": "ICC_TEST",
                },
                expected_updates={"table_name": "test_table"},
                expected_action=None,
            )
        ]
    )
    tid_counter += 1

    cases.extend(
        [
            ExtractionTestCase(
                tid=f"T{tid_counter:03d}",
                description="table_name quoted",
                last_question=q_table,
                user_answer='let\'s use "RESULT_2025"',
                current={
                    "name": "job_full_4",
                    "execute_query": True,
                    "result_schema": "DW",
                },
                expected_updates={"table_name": "RESULT_2025"},
                expected_action=None,
            )
        ]
    )
    tid_counter += 1

    cases.extend(
        [
            ExtractionTestCase(
                tid=f"T{tid_counter:03d}",
                description="drop_before_create yes",
                last_question=q_drop,
                user_answer="yes, drop it first",
                current={"name": "job_full_5", "execute_query": True},
                expected_updates={"drop_before_create": True},
                expected_action=None,
            )
        ]
    )
    tid_counter += 1

    cases.extend(
        [
            ExtractionTestCase(
                tid=f"T{tid_counter:03d}",
                description="drop_before_create no",
                last_question=q_drop,
                user_answer="no, do not drop it",
                current={"name": "job_full_6", "execute_query": True},
                expected_updates={"drop_before_create": False},
                expected_action=None,
            )
        ]
    )
    tid_counter += 1

    cases.extend(
        [
            ExtractionTestCase(
                tid=f"T{tid_counter:03d}",
                description="write_count_connection",
                last_question=q_wc_conn,
                user_answer="ORACLE_10",
                current={"name": "job_full_7", "write_count": True},
                expected_updates={"write_count_connection": "ORACLE_10"},
                expected_action=None,
            )
        ]
    )
    tid_counter += 1

    cases.extend(
        [
            ExtractionTestCase(
                tid=f"T{tid_counter:03d}",
                description="write_count_schema",
                last_question=q_wc_schema,
                user_answer="ICC_TEST",
                current={
                    "name": "job_full_8",
                    "write_count": True,
                    "write_count_connection": "ORACLE_10",
                },
                expected_updates={"write_count_schema": "ICC_TEST"},
                expected_action=None,
            )
        ]
    )
    tid_counter += 1

    cases.extend(
        [
            ExtractionTestCase(
                tid=f"T{tid_counter:03d}",
                description="write_count_table",
                last_question=q_wc_table,
                user_answer="ROWCOUNT_T",
                current={
                    "name": "job_full_9",
                    "write_count": True,
                    "write_count_connection": "ORACLE_10",
                    "write_count_schema": "ICC_TEST",
                },
                expected_updates={"write_count_table": "ROWCOUNT_T"},
                expected_action=None,
            )
        ]
    )
    tid_counter += 1

    # JobAgent örneklerinden otomatik case ekle (0001_job_agent.txt ... 0014_job_agent.txt)
    try:
        cases.extend(load_job_agent_cases_from_txt(Path(__file__).resolve().parent))
    except Exception:
        pass

    return cases


# ---------------------------------------------------------------------------
# TEK TEST ÇALIŞTIRICI
# ---------------------------------------------------------------------------

def run_single_test(llm: ChatOllama, case: ExtractionTestCase) -> ExtractionTestResult:
    user_prompt = build_user_prompt(case)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    parse_error = False
    error = False
    type_name_bug = False
    param_overwrite_bug = False
    param_drop_bug = False
    dropped_keys: List[str] = []
    got_action: Optional[str] = None
    new_params: Dict[str, object] = {}
    merged_params: Dict[str, object] = dict(case.current)

    try:
        raw_text, ttf, t_total = call_llm_with_timing(llm, messages)
    except Exception as e:
        error = True
        raw_text = f"<<LLM ERROR: {e}>>"
        ttf = 0.0
        t_total = 0.0

    if not error:
        try:
            obj = safe_parse_json(raw_text)
            got_action = obj.get("action")
            new_params = obj.get("params") or {}
        except Exception as e:
            parse_error = True
            raw_text = f"<<PARSE ERROR: {e}>>" + "\nRAW: " + raw_text

    # Param bug kontrolleri + merge
    if not parse_error and not error:
        # type-name bug
        for k, v in new_params.items():
            if isinstance(v, str) and v.strip().lower() in TYPE_NAME_STRINGS:
                type_name_bug = True

        # overwrite bug (önceki değerden farklıysa)
        for k, v in new_params.items():
            if k in merged_params and merged_params[k] != v:
                param_overwrite_bug = True

        # drop bug: model 'params' içinde Current'taki key'leri tamamen vermediyse (sadece tek key döndürdüyse)
        dropped_keys = [k for k in case.current.keys() if k not in new_params]
        param_drop_bug = len(dropped_keys) > 0

        merged_params.update(new_params)

    # expected_updates karşılaştırması
    updates_correct = 0
    updates_total = len(case.expected_updates)
    for k, expected_val in case.expected_updates.items():
        actual_val = merged_params.get(k, None)
        if actual_val == expected_val:
            updates_correct += 1

    # Başarı kriteri:
    # Sadece extraction + bug yokluğu; action'ı zorunlu tutmuyoruz
    success = (
        not error
        and not parse_error
        and not type_name_bug
        and not param_overwrite_bug
        and not param_drop_bug
        and updates_correct == updates_total
    )

    return ExtractionTestResult(
        tid=case.tid,
        description=case.description,
        success=success,
        parse_error=parse_error,
        error=error,
        type_name_bug=type_name_bug,
        param_overwrite_bug=param_overwrite_bug,
        param_drop_bug=param_drop_bug,
        dropped_keys=dropped_keys,
        updates_correct=updates_correct,
        updates_total=updates_total,
        expected_action=case.expected_action,
        got_action=got_action,
        time_to_first_token=ttf,
        time_total=t_total,
        raw_output=raw_text,
        merged_params=merged_params,
        expected_updates=case.expected_updates,
        last_question=case.last_question,
        user_answer=case.user_answer,
        current=case.current,
        user_prompt=user_prompt,
    )
 

# ---------------------------------------------------------------------------
# LOG YAZIMI & MAIN
# ---------------------------------------------------------------------------

def write_log(
    log_path: str,
    model_name: str,
    results: List[ExtractionTestResult],
) -> None:
    total_tests = len(results)
    success_count = sum(1 for r in results if r.success)
    fail_count = total_tests - success_count
    parse_errors = sum(1 for r in results if r.parse_error)
    type_name_bugs = sum(1 for r in results if r.type_name_bug)
    overwrite_bugs = sum(1 for r in results if r.param_overwrite_bug)
    errors = sum(1 for r in results if r.error)

    total_ttf = sum(r.time_to_first_token for r in results)
    total_time = sum(r.time_total for r in results)
    total_updates_correct = sum(r.updates_correct for r in results)
    total_updates = sum(r.updates_total for r in results)

    avg_ttf = total_ttf / total_tests if total_tests > 0 else 0.0
    avg_total_time = total_time / total_tests if total_tests > 0 else 0.0
    update_accuracy = (
        total_updates_correct / total_updates if total_updates > 0 else 0.0
    )

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("JOB_AGENT SINGLE-STEP EXTRACTION BENCHMARK LOG\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Total tests: {total_tests}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write("=" * 80 + "\n\n")

        for r in results:
            f.write(f"=== TEST {r.tid} ===\n")
            f.write(f"Description: {r.description}\n")

            f.write(f"Success: {r.success}\n")
            f.write(
                f"Flags: parse_error={r.parse_error}, "
                f"error={r.error}, "
                f"type_name_bug={r.type_name_bug}, "
                f"overwrite_bug={r.param_overwrite_bug}\n"
            )
            f.write(
                f"Updates: {r.updates_correct}/{r.updates_total} correct "
                f"({(r.updates_correct/r.updates_total*100 if r.updates_total else 0):.1f}%)\n"
            )
            f.write(
                f"Action (ignored for success): expected={r.expected_action!r}, "
                f"got={r.got_action!r}\n"
            )
            f.write(
                f"Timing: TTFT={r.time_to_first_token:.3f}s, "
                f"total={r.time_total:.3f}s\n"
            )
            f.write("Last question:\n")
            f.write(f"{r.last_question}\n")
            f.write("User answer:\n")
            f.write(f"{r.user_answer}\n")
            f.write("Current params (before call):\n")
            f.write(json.dumps(r.current, indent=2, ensure_ascii=False) + "\n")
            f.write("Full USER PROMPT sent to model:\n")
            f.write(r.user_prompt + "\n\n")
            f.write("Expected updates:\n")
            f.write(json.dumps(r.expected_updates, indent=2, ensure_ascii=False) + "\n")
            f.write("Merged params after LLM:\n")
            f.write(json.dumps(r.merged_params, indent=2, ensure_ascii=False) + "\n")
            f.write("Raw output:\n")
            f.write(r.raw_output + "\n")
            f.write("\n" + "-" * 80 + "\n\n")

        # GLOBAL SUMMARY
        f.write("\n" + "=" * 80 + "\n")
        f.write("GLOBAL SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Total tests: {total_tests}\n")
        f.write(f"Success: {success_count}\n")
        f.write(f"Fail: {fail_count}\n")
        f.write(f"Parse errors: {parse_errors}\n")
        f.write(f"Type-name bugs: {type_name_bugs}\n")
        f.write(f"Overwrite bugs: {overwrite_bugs}\n")
        f.write(f"Other errors: {errors}\n")
        f.write(f"Average TTFT: {avg_ttf:.3f}s\n")
        f.write(f"Average total time per test: {avg_total_time:.3f}s\n")
        f.write(
            f"Overall expected-update accuracy: "
            f"{total_updates_correct}/{total_updates} ({update_accuracy*100:.1f}%)\n"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Single-step read_sql job_agent extraction benchmark with qwen3 via Ollama."
    )
    parser.add_argument(
        "--model_size",
        choices=["8b", "1.7b"],
        default="1.7b",
        help="qwen3 model size (default: 1.7b)",
    )
    parser.add_argument(
        "--think",
        action="store_true",
        help="Enable 'think' reasoning mode.",
    )
    parser.add_argument(
        "--max_tests",
        type=int,
        default=None,
        help="Run only the first N tests (default: all).",
    )
    args = parser.parse_args()

    llm = build_llm(args.model_size, args.think)
    model_name = f"qwen3:{args.model_size}"

    cases = build_test_cases()
    if args.max_tests is not None and args.max_tests < len(cases):
        cases = cases[: args.max_tests]

    results: List[ExtractionTestResult] = []
    print(f"[INFO] Running {len(cases)} extraction tests...")

    for i, case in enumerate(cases, start=1):
        print(f"[INFO] Test {i}/{len(cases)}: {case.tid} - {case.description}")
        res = run_single_test(llm, case)
        results.append(res)
        print(
            f"   -> success={res.success}, "
            f"updates={res.updates_correct}/{res.updates_total}, "
            f"action={res.got_action}, "
            f"time_total={res.time_total:.3f}s"
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"job_agent_extraction_{model_name.replace(':', '_')}_{ts}.log"
    write_log(log_path, model_name, results)

    print(f"[INFO] Benchmark finished. Log written to: {log_path}")


if __name__ == "__main__":
    main()
