#!/usr/bin/env python
"""
Unified single-step parameter extraction benchmark (read_sql + write_data + email) for Qwen3 via Ollama.

Goal
- Router asks ONE question at a time
- User answers (may include extra info)
- LLM must return JSON: {"params": {...}} by starting from Current and updating:
  - the PRIMARY parameter requested by the last question
  - plus OPTIONAL extra parameters ONLY if they are currently missing/unknown AND explicitly provided in the answer

Notes
- This file intentionally mixes:
  - "single-answer" cases (classic slot-fill)
  - "multi-answer" cases (user provides more than asked; model should opportunistically fill missing)
- Router question groups are realistic: read_sql questions cluster, then write_data, then email.
"""

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ChatOllama import: prefer langchain_ollama, fallback to langchain_community
try:
    from langchain_ollama import ChatOllama
except ImportError:  # pragma: no cover
    from langchain_community.chat_models import ChatOllama

from langchain_core.messages import SystemMessage, HumanMessage


# ---------------------------------------------------------------------------
# SYSTEM PROMPT (unified)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = r"""
You are a STRICT JSON parameter extraction assistant used inside a step-by-step router.

You will receive for EACH call:
- Last question: the router's most recent question (ONE question).
- User answer: the user's answer to that question (may contain extra info).
- Current: the current collected params (JSON).
- Missing: params that are still unknown (computed outside).

Your output MUST be JSON ONLY:
{"params": {...}}

CRITICAL BASE RULES:
1) ALWAYS START FROM Current:
   - Copy ALL keys+values from Current into params unchanged.
   - NEVER delete keys.
   - NEVER overwrite an existing real value in Current.
     * Real value means: not "" and not null.
     * Overwriting "" or null IS allowed (those mean "unknown").

2) PRIMARY UPDATE:
   - Identify the PRIMARY parameter the Last question is asking about.
   - Extract its value from User answer and set ONLY that parameter (unless extraction fails).
   - If you cannot confidently extract the primary value: return Current unchanged.

3) OPTIONAL EXTRA FILL (important, realistic behavior):
   - In addition to the primary parameter, you MAY fill extra parameters ONLY IF ALL are true:
     a) That parameter is listed in Missing (or is missing/unknown in Current).
     b) The user answer explicitly provides its value (no guessing).
     c) The parameter belongs to the same job type as the last question (read_sql OR write_data OR email).
   - NEVER fill extra params from a different job type.

JOB BOUNDARIES (do not mix keys across jobs):
- read_sql keys: name, execute_query, write_count, result_schema, table_name, drop_before_create,
               write_count_connection, write_count_schema, write_count_table
- write_data keys: name, table, schemas, connection, drop_or_truncate
- email keys: name, to, subject, text, cc

KEY SELECTION HINTS (avoid common mistakes):
- If the question asks for a TABLE (results table or write_data table), NEVER set "name".
- If the question asks "Who should I send the email to?", set "to" (email address), NEVER set "name".
- If the question is about schema+table and the answer contains "SCHEMA.TABLE" (dot):
  - For read_sql: result_schema = SCHEMA, table_name = TABLE
  - For write_data: schemas = SCHEMA, table = TABLE

BOOLEAN MAPPING (only when the question is explicitly yes/no):
TRUE if answer contains: "yes", "yeah", "yep", "sure", "please do", "go ahead", "run it", "execute it", "do it", "save it"
FALSE if answer contains: "no", "nope", "nah", "don't", "do not", "preview only", "just preview", "just show", "don't save",
                          "no saving", "no need", "not required", "skip", "skip it", "unnecessary"

DROP/TRUNCATE/NONE (write_data only):
- If asked drop/truncate/none:
  - "drop" -> "drop"
  - "truncate" -> "truncate"
  - "none" or "append" or "keep appending" -> "none"

STRING EXTRACTION (explicit only):
- Prefer the last quoted string if present.
- Otherwise, extract the strongest identifier token:
  - [A-Za-z_][A-Za-z0-9_]* or emails like something@domain
- Ignore filler words ("ok", "okay", "sure", "please", "let's", "call", "name", etc.)
- NEVER output type names as values ("string", "boolean", "int", ...)

EMAIL EXTRACTION (explicit patterns):
- "to": extract the first email address that is NOT introduced as "cc" (if both exist).
- "cc": if "cc" is present, extract email(s) after it; if user explicitly says "no cc", set cc="" ONLY if cc is missing.
- "subject": if the answer contains the word "subject", extract the quoted text after it (or the nearest phrase).
- "text": if the answer contains the word "text" (or "message" / "body"), extract the quoted text after it.

ABSOLUTE OUTPUT RULE:
- JSON only, exactly one top-level key: "params".
- No markdown, no additional keys, no explanations.
""".strip()


# ---------------------------------------------------------------------------
# JOB SPECS (keep them short; System prompt holds the core rules)
# ---------------------------------------------------------------------------

READ_SQL_SPEC = """
Extract params for read_sql job.

You see: Last question / User answer / Current / Missing.
Update params by starting from Current.

read_sql keys:
- name
- execute_query (yes/no -> true/false)
- write_count (yes/no -> true/false)
If execute_query=true: result_schema, table_name, drop_before_create (yes/no -> true/false)
If write_count=true: write_count_connection, write_count_schema, write_count_table

Output JSON only: {"params": {...}}
""".strip()

WRITE_DATA_SPEC = """
Extract params for write_data job.

Keys: name, table, schemas, connection, drop_or_truncate
If asked drop/truncate/none: set drop_or_truncate exactly.

Output JSON only: {"params": {...}}
""".strip()

EMAIL_SPEC = """
Extract params for email job.

Keys: name, to, subject, text, cc

Output JSON only: {"params": {...}}
""".strip()

TYPE_NAME_STRINGS = {
    "string", "boolean", "bool", "integer", "int", "float", "double", "varchar", "text"
}

JOB_KEYS = {
    "read_sql": [
        "name", "execute_query", "write_count",
        "result_schema", "table_name", "drop_before_create",
        "write_count_connection", "write_count_schema", "write_count_table",
    ],
    "write_data": ["name", "table", "schemas", "connection", "drop_or_truncate"],
    "email": ["name", "to", "subject", "text", "cc"],
}

JOB_SPEC = {"read_sql": READ_SQL_SPEC, "write_data": WRITE_DATA_SPEC, "email": EMAIL_SPEC}


# ---------------------------------------------------------------------------
# DATA CLASSES
# ---------------------------------------------------------------------------

@dataclass
class ExtractionTestCase:
    tid: str
    description: str
    job: str
    last_question: str
    user_answer: str
    current: Dict[str, Any]
    expected_updates: Dict[str, Any]


@dataclass
class ExtractionTestResult:
    tid: str
    description: str
    success: bool
    parse_error: bool
    error: bool
    type_name_bug: bool
    invalid_key_bug: bool
    param_overwrite_bug: bool
    unexpected_update_bug: bool
    updates_correct: int
    updates_total: int
    time_to_first_token: float
    time_total: float
    raw_output: str
    merged_params: Dict[str, Any]
    expected_updates: Dict[str, Any]
    last_question: str
    user_answer: str
    current: Dict[str, Any]
    user_prompt: str


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def build_llm(
    model_size: str,
    think: bool,
    base_url: str,
    temperature: float,
    num_predict: int,
    timeout_s: float,
    num_ctx: Optional[int],
) -> ChatOllama:
    model_name = f"qwen3:{model_size}"
    print(f"[INFO] Initializing model={model_name}, think={think}, base_url={base_url}")
    kwargs = {"think": think, "stream": True}
    if num_ctx is not None:
        kwargs["num_ctx"] = num_ctx
    return ChatOllama(
        model=model_name,
        temperature=temperature,
        base_url=base_url,
        num_predict=num_predict,
        timeout=timeout_s,
        keep_alive="3600s",
        model_kwargs=kwargs,
    )


def safe_parse_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty response from model")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def call_llm_with_timing(llm: ChatOllama, messages: List[Any]) -> Tuple[str, float, float]:
    start = time.perf_counter()
    chunks: List[str] = []
    t_first: Optional[float] = None

    def pick_text(chunk: Any) -> str:
        if isinstance(chunk, dict):
            msg = chunk.get("message") or {}
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return msg["content"]
            if isinstance(chunk.get("response"), str):
                return chunk["response"]
            return ""
        if hasattr(chunk, "content") and chunk.content:
            if isinstance(chunk.content, str):
                return chunk.content
            if isinstance(chunk.content, list):
                return "".join(getattr(m, "content", "") for m in chunk.content)
        if hasattr(chunk, "text") and chunk.text:
            return chunk.text
        if hasattr(chunk, "message") and hasattr(chunk.message, "content") and chunk.message.content:
            return chunk.message.content
        return ""

    try:
        for chunk in llm.stream(messages):
            if t_first is None:
                t_first = time.perf_counter() - start
            piece = pick_text(chunk)
            if piece:
                chunks.append(piece)
    except Exception:
        chunks = []

    full_text = "".join(chunks).strip()
    t_total = time.perf_counter() - start
    if t_first is None:
        t_first = t_total

    if not full_text:
        start2 = time.perf_counter()
        resp = llm.invoke(messages)
        t_total2 = time.perf_counter() - start2
        full_text = (getattr(resp, "content", "") or "").strip()
        return full_text, t_first, t_total + t_total2

    return full_text, t_first, t_total


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def is_unknown(v: Any) -> bool:
    return v is None or (isinstance(v, str) and v.strip() == "")


def build_user_prompt(case: ExtractionTestCase) -> str:
    spec = JOB_SPEC[case.job]
    keys = JOB_KEYS[case.job]
    missing = [k for k in keys if (k not in case.current) or is_unknown(case.current.get(k))]
    missing_str = ", ".join(missing) if missing else "none"

    return f"""{spec}

Last question: "{case.last_question}"
User answer: "{case.user_answer}"
Current: {json.dumps(case.current, ensure_ascii=False)}
Missing: {missing_str}

Output JSON only:
""".strip()


# ---------------------------------------------------------------------------
# Evaluator (robust for multi-fill)
# ---------------------------------------------------------------------------

def is_placeholder(v: Any) -> bool:
    return v is None or (isinstance(v, str) and v.strip() == "")


def run_single_test(llm: ChatOllama, case: ExtractionTestCase) -> ExtractionTestResult:
    user_prompt = build_user_prompt(case)
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_prompt)]

    parse_error = False
    error = False
    type_name_bug = False
    invalid_key_bug = False
    param_overwrite_bug = False
    unexpected_update_bug = False

    new_params: Dict[str, Any] = {}
    merged_params: Dict[str, Any] = dict(case.current)

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
            new_params = obj.get("params") or {}
            if not isinstance(new_params, dict):
                raise ValueError(f'"params" must be an object/dict, got {type(new_params)}')
        except Exception as e:
            parse_error = True
            raw_text = f"<<PARSE ERROR: {e}>>\nRAW: {raw_text}"

    if not parse_error and not error:
        allowed_keys = set(JOB_KEYS[case.job])
        for k in new_params.keys():
            if k not in allowed_keys:
                invalid_key_bug = True
                break

        for _, v in new_params.items():
            if isinstance(v, str) and v.strip().lower() in TYPE_NAME_STRINGS:
                type_name_bug = True

        # overwriting REAL values is forbidden (""/null are considered unknown and can be filled)
        for k, v in new_params.items():
            if k in merged_params:
                prev = merged_params[k]
                if is_placeholder(prev):
                    continue
                if prev != v:
                    param_overwrite_bug = True

        # if it changes a non-placeholder key that wasn't expected, flag as unexpected (helps catch "name" mistakes)
        expected_keys = set(case.expected_updates.keys())
        for k, v in new_params.items():
            prev = case.current.get(k, None)
            if prev == v:
                continue
            if not is_placeholder(prev) and k not in expected_keys:
                unexpected_update_bug = True

        merged_params.update(new_params)

    updates_total = len(case.expected_updates)
    updates_correct = 0
    for k, expected_val in case.expected_updates.items():
        if merged_params.get(k, None) == expected_val:
            updates_correct += 1

    success = (
        not error
        and not parse_error
        and not type_name_bug
        and not invalid_key_bug
        and not param_overwrite_bug
        and not unexpected_update_bug
        and updates_correct == updates_total
    )

    return ExtractionTestResult(
        tid=case.tid,
        description=case.description,
        success=success,
        parse_error=parse_error,
        error=error,
        type_name_bug=type_name_bug,
        invalid_key_bug=invalid_key_bug,
        param_overwrite_bug=param_overwrite_bug,
        unexpected_update_bug=unexpected_update_bug,
        updates_correct=updates_correct,
        updates_total=updates_total,
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
# Test cases (60)
# ---------------------------------------------------------------------------

def build_test_cases() -> List[ExtractionTestCase]:
    cases: List[ExtractionTestCase] = []
    tid = 1

    def add(job: str, desc: str, q: str, a: str, cur: Dict[str, Any], exp: Dict[str, Any]) -> None:
        nonlocal tid
        cases.append(
            ExtractionTestCase(
                tid=f"T{tid:03d}",
                description=desc,
                job=job,
                last_question=q,
                user_answer=a,
                current=cur,
                expected_updates=exp,
            )
        )
        tid += 1

    # -------------------------
    # READ_SQL (single)
    # -------------------------
    q_name = "What should I name this read_sql job?"
    add("read_sql", "read_sql name plain", q_name, "read1172", {"name": "", "execute_query": "", "write_count": ""}, {"name": "read1172"})
    add("read_sql", "read_sql name phrase", q_name, "Let's call it daily_orders", {"name": "", "execute_query": "", "write_count": ""}, {"name": "daily_orders"})
    add("read_sql", "read_sql name quoted", q_name, 'name it "job133"', {"name": "", "execute_query": "", "write_count": ""}, {"name": "job133"})

    q_exec = "Should the job execute the query and save results to the database? (yes/no)"
    add("read_sql", "execute_query yes", q_exec, "yes", {"name": "job_exec", "execute_query": None}, {"execute_query": True})
    add("read_sql", "execute_query no (preview only)", q_exec, "no, preview only", {"name": "job_exec2", "execute_query": None}, {"execute_query": False})
    add("read_sql", "execute_query yes phrase", q_exec, "sure, execute it and save", {"name": "job_exec3", "execute_query": None}, {"execute_query": True})
    add("read_sql", "execute_query no phrase", q_exec, "don't save to db", {"name": "job_exec4", "execute_query": None}, {"execute_query": False})

    q_wc = "Should the job track the row count? (yes/no)"
    add("read_sql", "write_count yes", q_wc, "yes, track it", {"name": "job_wc", "write_count": None}, {"write_count": True})
    add("read_sql", "write_count no", q_wc, "no, not necessary", {"name": "job_wc2", "write_count": None}, {"write_count": False})
    add("read_sql", "write_count yes short", q_wc, "yep", {"name": "job_wc3", "write_count": None}, {"write_count": True})
    add("read_sql", "write_count no short", q_wc, "nope", {"name": "job_wc4", "write_count": None}, {"write_count": False})

    q_schema = "Which schema should I write the results to?"
    q_table = "What table should I write the results to?"
    q_drop = "Should I drop the table before creating it? (yes/no)"
    add("read_sql", "result_schema simple", q_schema, "ICC_TEST", {"name": "job_s", "execute_query": True, "result_schema": ""}, {"result_schema": "ICC_TEST"})
    add("read_sql", "table_name simple", q_table, "test_table", {"name": "job_t", "execute_query": True, "result_schema": "ICC_TEST", "table_name": ""}, {"table_name": "test_table"})
    add("read_sql", "table_name quoted", q_table, 'let\'s use "RESULT_2025"', {"name": "job_t2", "execute_query": True, "result_schema": "ICC_TEST", "table_name": ""}, {"table_name": "RESULT_2025"})
    add("read_sql", "drop_before_create yes", q_drop, "yes, drop it first", {"name": "job_d", "execute_query": True, "drop_before_create": None}, {"drop_before_create": True})
    add("read_sql", "drop_before_create no", q_drop, "no, do not drop", {"name": "job_d2", "execute_query": True, "drop_before_create": None}, {"drop_before_create": False})

    q_wc_conn = "What is the connection string for the row count?"
    q_wc_schema = "What is the schema name for the row count?"
    q_wc_table = "What is the table name to store the row count?"
    add("read_sql", "write_count_connection", q_wc_conn, "ORACLE_10", {"name": "job_rc1", "write_count": True, "write_count_connection": ""}, {"write_count_connection": "ORACLE_10"})
    add("read_sql", "write_count_schema", q_wc_schema, "ICC_TEST", {"name": "job_rc2", "write_count": True, "write_count_schema": ""}, {"write_count_schema": "ICC_TEST"})
    add("read_sql", "write_count_table", q_wc_table, "ROWCOUNT_T", {"name": "job_rc3", "write_count": True, "write_count_table": ""}, {"write_count_table": "ROWCOUNT_T"})

    # -------------------------
    # READ_SQL (multi-answer / opportunistic fill)
    # -------------------------
    add(
        "read_sql",
        "name + execute_query=false in one answer",
        q_name,
        "name it read778900, no saving to db",
        {"name": "", "execute_query": "", "write_count": ""},
        {"name": "read778900", "execute_query": False},
    )
    add(
        "read_sql",
        "name + execute_query=true + write_count=true",
        q_name,
        "call it daily_orders; yes save to db and yes track row count",
        {"name": "", "execute_query": "", "write_count": ""},
        {"name": "daily_orders", "execute_query": True, "write_count": True},
    )
    add(
        "read_sql",
        "execute_query yes + also provides schema+table",
        q_exec,
        'yes, save it. use schema ICC_TEST and table "RESULT_2025"',
        {"name": "job_full_x", "execute_query": None, "result_schema": "", "table_name": ""},
        {"execute_query": True, "result_schema": "ICC_TEST", "table_name": "RESULT_2025"},
    )
    add(
        "read_sql",
        "write_count question answered + also says don't save to db",
        q_wc,
        "no, skip row count. also don't save to db",
        {"name": "job_mix1", "execute_query": None, "write_count": None},
        {"write_count": False, "execute_query": False},
    )
    add(
        "read_sql",
        "schema question but user also gives table",
        q_schema,
        "schema ICC_TEST table RESULT_2025",
        {"name": "job_mix2", "execute_query": True, "result_schema": "", "table_name": ""},
        {"result_schema": "ICC_TEST", "table_name": "RESULT_2025"},
    )
    add(
        "read_sql",
        "table question answered as SCHEMA.TABLE (dot)",
        q_table,
        "ICC_TEST.RESULT_2025",
        {"name": "job_mix3", "execute_query": True, "result_schema": "", "table_name": ""},
        {"result_schema": "ICC_TEST", "table_name": "RESULT_2025"},
    )
    add(
        "read_sql",
        "drop question + user also repeats schema.table",
        q_drop,
        "yes. ICC_TEST.RESULT_2025",
        {"name": "job_mix4", "execute_query": True, "drop_before_create": None, "result_schema": "", "table_name": ""},
        {"drop_before_create": True, "result_schema": "ICC_TEST", "table_name": "RESULT_2025"},
    )
    add(
        "read_sql",
        "name + preview only + no row count",
        q_name,
        "call it job99, preview only, no row count",
        {"name": "", "execute_query": "", "write_count": ""},
        {"name": "job99", "execute_query": False, "write_count": False},
    )
    add(
        "read_sql",
        "execute_query no + also says row count no",
        q_exec,
        "no, just show; and no row count",
        {"name": "job_mix5", "execute_query": None, "write_count": None},
        {"execute_query": False, "write_count": False},
    )
    add(
        "read_sql",
        "ambiguous answer should not update",
        q_exec,
        "maybe later",
        {"name": "job_amb", "execute_query": None},
        {},
    )

    # -------------------------
    # WRITE_DATA (single)
    # -------------------------
    q_wd_name = "What should I name this write_data job?"
    q_wd_table = "What table should I write the data to?"
    q_wd_schema = "Which schema should I use for the write?"
    q_wd_conn = "Which connection should I use to write the data?"
    q_wd_drop = "Should I 'drop' (remove and recreate), 'truncate' (clear data), or 'none' (append)?"

    add("write_data", "write_data name plain", q_wd_name, "write75684", {"name": "", "table": "", "schemas": "", "connection": "", "drop_or_truncate": ""}, {"name": "write75684"})
    add("write_data", "write_data name phrase", q_wd_name, "let's call it nightly_etl", {"name": "", "table": "", "schemas": "", "connection": "", "drop_or_truncate": ""}, {"name": "nightly_etl"})
    add("write_data", "write_data table simple", q_wd_table, "test_table", {"name": "w1", "table": ""}, {"table": "test_table"})
    add("write_data", "write_data schemas simple", q_wd_schema, "ICC_TEST", {"name": "w2", "schemas": ""}, {"schemas": "ICC_TEST"})
    add("write_data", "write_data connection simple", q_wd_conn, "ORACLE_10", {"name": "w3", "connection": ""}, {"connection": "ORACLE_10"})
    add("write_data", "write_data drop=drop", q_wd_drop, "drop", {"name": "w4", "drop_or_truncate": ""}, {"drop_or_truncate": "drop"})
    add("write_data", "write_data drop=truncate", q_wd_drop, "truncate", {"name": "w5", "drop_or_truncate": ""}, {"drop_or_truncate": "truncate"})
    add("write_data", "write_data drop=none", q_wd_drop, "none", {"name": "w6", "drop_or_truncate": ""}, {"drop_or_truncate": "none"})
    add("write_data", "write_data drop=append synonym", q_wd_drop, "keep appending", {"name": "w7", "drop_or_truncate": ""}, {"drop_or_truncate": "none"})

    # -------------------------
    # WRITE_DATA (multi-answer)
    # -------------------------
    add(
        "write_data",
        "table answer also includes schema+connection",
        q_wd_table,
        "write to test_table, schema ICC_TEST, connection ORACLE_10",
        {"name": "writeA", "table": "", "schemas": "", "connection": "", "drop_or_truncate": ""},
        {"table": "test_table", "schemas": "ICC_TEST", "connection": "ORACLE_10"},
    )
    add(
        "write_data",
        "schema answer includes table+connection",
        q_wd_schema,
        "use ICC_TEST, table test_table, connection ORACLE_10",
        {"name": "writeB", "table": "", "schemas": "", "connection": "", "drop_or_truncate": ""},
        {"schemas": "ICC_TEST", "table": "test_table", "connection": "ORACLE_10"},
    )
    add(
        "write_data",
        "connection answer includes schema+table",
        q_wd_conn,
        "ORACLE_10, schema ICC_TEST, table test_table",
        {"name": "writeC", "table": "", "schemas": "", "connection": "", "drop_or_truncate": ""},
        {"connection": "ORACLE_10", "schemas": "ICC_TEST", "table": "test_table"},
    )
    add(
        "write_data",
        "dot schema.table should split",
        q_wd_table,
        "ICC_TEST.test_table",
        {"name": "writeD", "table": "", "schemas": "", "connection": "ORACLE_10", "drop_or_truncate": ""},
        {"schemas": "ICC_TEST", "table": "test_table"},
    )
    add(
        "write_data",
        "drop question but user also gives schema.table (fill if missing)",
        q_wd_drop,
        "truncate ICC_TEST.test_table",
        {"name": "writeE", "table": "", "schemas": "", "connection": "ORACLE_10", "drop_or_truncate": ""},
        {"drop_or_truncate": "truncate", "schemas": "ICC_TEST", "table": "test_table"},
    )
    add(
        "write_data",
        "drop question re-mentions table but should not overwrite",
        q_wd_drop,
        "none, keep appending to test_table",
        {"name": "writeF", "table": "test_table", "schemas": "ICC_TEST", "connection": "ORACLE_10", "drop_or_truncate": ""},
        {"drop_or_truncate": "none"},
    )
    add(
        "write_data",
        "unknown answer should not update",
        q_wd_table,
        "not sure yet",
        {"name": "writeG", "table": ""},
        {},
    )

    # -------------------------
    # EMAIL (single)
    # -------------------------
    q_em_name = "What should I name this email job?"
    q_em_to = "Who should I send the email to?"
    q_em_subject = "What should the email subject be?"
    q_em_text = "What text should the email contain?"
    q_em_cc = "Should I CC anyone? (provide email or say 'no')"

    add("email", "email name plain", q_em_name, "email4857489", {"name": "", "to": "", "subject": "", "text": "", "cc": ""}, {"name": "email4857489"})
    add("email", "email name phrase", q_em_name, "call it customer_audit", {"name": "", "to": "", "subject": "", "text": "", "cc": ""}, {"name": "customer_audit"})
    add("email", "email to simple", q_em_to, "selcan.yukcu@pia-team.com", {"name": "e1", "to": ""}, {"to": "selcan.yukcu@pia-team.com"})
    add("email", "email to phrase", q_em_to, "send to selcan.yukcu@pia-team.com", {"name": "e2", "to": ""}, {"to": "selcan.yukcu@pia-team.com"})
    add("email", "email subject quoted", q_em_subject, 'subject "Data Info"', {"name": "e3", "subject": ""}, {"subject": "Data Info"})
    add("email", "email text quoted", q_em_text, 'text "You can check the data"', {"name": "e4", "text": ""}, {"text": "You can check the data"})
    add("email", "email cc explicit email", q_em_cc, "cc alp@example.com", {"name": "e5", "cc": ""}, {"cc": "alp@example.com"})
    add("email", "email cc none", q_em_cc, "no cc", {"name": "e6", "cc": ""}, {"cc": ""})

    # -------------------------
    # EMAIL (multi-answer)
    # -------------------------
    add(
        "email",
        "to question but user provides subject+text too",
        q_em_to,
        'Send the text "You can check the data" with the subject "Data Info" to selcan.yukcu@pia-team.com',
        {"name": "email4857489", "to": "", "subject": "", "text": "", "cc": ""},
        {"to": "selcan.yukcu@pia-team.com", "subject": "Data Info", "text": "You can check the data"},
    )
    add(
        "email",
        "to question includes CC too",
        q_em_to,
        'to selcan.yukcu@pia-team.com cc alp@example.com subject "Data Info" text "You can check the data"',
        {"name": "email_cc", "to": "", "subject": "", "text": "", "cc": ""},
        {"to": "selcan.yukcu@pia-team.com", "cc": "alp@example.com", "subject": "Data Info", "text": "You can check the data"},
    )
    add(
        "email",
        "subject question but user also includes to+text",
        q_em_subject,
        'to selcan.yukcu@pia-team.com subject "Data Info" text "You can check the data"',
        {"name": "emailS", "to": "", "subject": "", "text": "", "cc": ""},
        {"subject": "Data Info", "to": "selcan.yukcu@pia-team.com", "text": "You can check the data"},
    )
    add(
        "email",
        "text question but user also includes subject",
        q_em_text,
        'text "Monthly report is ready" subject "Report Ready"',
        {"name": "emailT", "to": "selcan.yukcu@pia-team.com", "subject": "", "text": "", "cc": ""},
        {"text": "Monthly report is ready", "subject": "Report Ready"},
    )
    add(
        "email",
        "cc question but user also provides to",
        q_em_cc,
        "to selcan.yukcu@pia-team.com cc alp@example.com",
        {"name": "emailC", "to": "", "subject": "", "text": "", "cc": ""},
        {"cc": "alp@example.com", "to": "selcan.yukcu@pia-team.com"},
    )
    add(
        "email",
        "to question: should NEVER overwrite name even if 'name it' is mentioned",
        q_em_to,
        "send it to selcan.yukcu@pia-team.com, name it something else",
        {"name": "email4857489", "to": "", "subject": "", "text": "", "cc": ""},
        {"to": "selcan.yukcu@pia-team.com"},
    )
    add(
        "email",
        "ambiguous answer should not update",
        q_em_to,
        "not sure",
        {"name": "emailAmb", "to": ""},
        {},
    )

    return cases


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def write_log(path: str, model_name: str, results: List[ExtractionTestResult]) -> None:
    total = len(results)
    ok = sum(1 for r in results if r.success)
    fail = total - ok
    parse_err = sum(1 for r in results if r.parse_error)
    errors = sum(1 for r in results if r.error)
    type_bugs = sum(1 for r in results if r.type_name_bug)
    invalid_key_bugs = sum(1 for r in results if r.invalid_key_bug)
    overwrite_bugs = sum(1 for r in results if r.param_overwrite_bug)
    unexpected_bugs = sum(1 for r in results if r.unexpected_update_bug)

    avg_ttf = sum(r.time_to_first_token for r in results) / total if total else 0.0
    avg_total = sum(r.time_total for r in results) / total if total else 0.0
    upd_correct = sum(r.updates_correct for r in results)
    upd_total = sum(r.updates_total for r in results)
    upd_acc = (upd_correct / upd_total) if upd_total else 0.0

    with open(path, "w", encoding="utf-8") as f:
        f.write("JOB_AGENT UNIFIED SINGLE-STEP EXTRACTION BENCHMARK LOG\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Total tests: {total}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write("=" * 80 + "\n\n")

        for r in results:
            f.write(f"=== {r.tid} | {'OK' if r.success else 'FAIL'} | {r.description} ===\n")
            f.write(f"ttft={r.time_to_first_token:.3f}s total={r.time_total:.3f}s\n")
            f.write(f"Last question: {r.last_question}\n")
            f.write(f"User answer:  {r.user_answer}\n")
            f.write("Current (before): " + json.dumps(r.current, ensure_ascii=False) + "\n")
            f.write("Raw output: " + (r.raw_output or "").strip().replace("\n", "\\n") + "\n")
            f.write("Merged (after):  " + json.dumps(r.merged_params, ensure_ascii=False) + "\n")
            f.write("Expected updates: " + json.dumps(r.expected_updates, ensure_ascii=False) + "\n")
            f.write(
                f"Bugs: parse={r.parse_error} err={r.error} type={r.type_name_bug} "
                f"invalid_key={r.invalid_key_bug} overwrite={r.param_overwrite_bug} unexpected={r.unexpected_update_bug}\n"
            )
            f.write("\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("GLOBAL SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Total tests: {total}\n")
        f.write(f"Success: {ok}\n")
        f.write(f"Fail: {fail}\n")
        f.write(f"Parse errors: {parse_err}\n")
        f.write(f"Errors: {errors}\n")
        f.write(f"Type-name bugs: {type_bugs}\n")
        f.write(f"Invalid-key bugs: {invalid_key_bugs}\n")
        f.write(f"Overwrite bugs: {overwrite_bugs}\n")
        f.write(f"Unexpected-update bugs: {unexpected_bugs}\n")
        f.write(f"Average TTFT: {avg_ttf:.3f}s\n")
        f.write(f"Average total time per test: {avg_total:.3f}s\n")
        f.write(f"Overall expected-update accuracy: {upd_correct}/{upd_total} ({upd_acc*100:.1f}%)\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Unified job_agent single-step extraction benchmark (read_sql/write_data/email).")
    p.add_argument("--model_size", choices=["1.7b", "8b"], default="1.7b")
    p.add_argument("--think", action="store_true")
    p.add_argument("--base_url", default="http://localhost:11434")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--num_predict", type=int, default=512)
    p.add_argument("--num_ctx", type=int, default=None)
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--max_tests", type=int, default=None)
    p.add_argument("--only_job", choices=["read_sql", "write_data", "email"], default=None)
    args = p.parse_args()

    llm = build_llm(
        model_size=args.model_size,
        think=args.think,
        base_url=args.base_url,
        temperature=args.temperature,
        num_predict=args.num_predict,
        timeout_s=args.timeout,
        num_ctx=args.num_ctx,
    )
    model_name = f"qwen3:{args.model_size}"

    cases = build_test_cases()
    if args.only_job:
        cases = [c for c in cases if c.job == args.only_job]
    if args.max_tests is not None:
        cases = cases[: args.max_tests]

    print(f"[INFO] Running {len(cases)} tests...")
    results: List[ExtractionTestResult] = []
    for i, case in enumerate(cases, start=1):
        print(f"[INFO] {i}/{len(cases)} {case.tid} ({case.job}) {case.description}")
        res = run_single_test(llm, case)
        results.append(res)
        print(f"   -> success={res.success} updates={res.updates_correct}/{res.updates_total} total={res.time_total:.3f}s")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"job_agent_unified_{model_name.replace(':','_')}_{ts}.log"
    write_log(out, model_name, results)
    print(f"[INFO] Done. Log written to: {out}")


if __name__ == "__main__":
    main()
