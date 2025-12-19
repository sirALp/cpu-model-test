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

SYSTEM_PROMPT = r"""You are a STRICT JSON parameter extraction assistant used inside a router.

You will receive:
- Last question: the router's most recent question to the user
- User answer: the user's latest reply
- Current: a JSON object with the currently-collected params (some may be empty/unknown)
- Missing: a comma-separated list of params the router still needs right now

Your job:
- Extract values from the User answer and return JSON only:
  {"params": {...}}

Core rules:
1) Only set params that are in Missing (same job group).
2) Never overwrite a param that already has a real value in Current.
   - Empty string "" means unknown string.
   - null means unknown boolean.
   - Overwriting a non-empty string or a non-null boolean is forbidden.
3) Multi-fill is allowed:
   - Always try to fill the parameter asked by Last question.
   - If the user answer ALSO clearly provides values for OTHER params in Missing, you may fill them too.
   - Do not guess. If not explicit, do not fill.

Output size / safety:
- Prefer returning ONLY the keys you are setting now (a delta). The router will merge with Current.
- You may include unchanged Current keys, but do NOT change their values.

Booleans (JSON true/false, NOT strings):
- Map to TRUE if the answer clearly indicates yes/enable/run/save:
  "yes", "yeah", "yep", "sure", "go ahead", "please do", "run it", "execute it", "do it", "enable", "turn on", "save to db", "save it"
- Map to FALSE if the answer clearly indicates no/disable/skip/preview-only:
  "no", "nope", "nah", "don't", "do not", "do not save", "don't save", "no saving", "no saving to db",
  "preview only", "just preview", "only preview", "skip", "skip it", "not needed", "not necessary", "no need"

String extraction:
- Prefer the last quoted string if present.
- Otherwise, extract the strongest identifier token: [A-Za-z_][A-Za-z0-9_]*
- For email fields:
  - "to" / "cc" must be email addresses (contain '@').
  - "subject" and "text" are often quoted strings.

Dot-separated identifiers:
- For write_data table answers like SCHEMA.TABLE:
  - If both "schemas" and "table" are in Missing, set schemas=SCHEMA and table=TABLE.
  - If only "table" is Missing, set table=TABLE.

Never do these:
- Never output type names as values ("string", "boolean", "varchar", "int", ...)
- Never invent values.
- Never return anything except a single JSON object with top-level key "params".

Examples:

Example A (read_sql name answer also indicates no-save):
Last question: What should I name this read_sql job?
Missing: name, execute_query, write_count
User answer: name it read778900, no saving to db
Output:
{"params":{"name":"read778900","execute_query":false}}

Example B (email "to" answer also includes subject/text/cc):
Last question: Who should I send the email to?
Missing: to, subject, text, cc
User answer: to a@b.com cc c@d.com subject "Data Info" text "You can check the data"
Output:
{"params":{"to":"a@b.com","cc":"c@d.com","subject":"Data Info","text":"You can check the data"}}

Example C (write_data dot table):
Last question: What table should I write the data to?
Missing: table, schemas
User answer: ICC_TEST.test_table
Output:
{"params":{"schemas":"ICC_TEST","table":"test_table"}}""".strip()


# ---------------------------------------------------------------------------
# JOB SPECS (keep them short; System prompt holds the core rules)
# ---------------------------------------------------------------------------

READ_SQL_SPEC = r"""Extract params for read_sql job.

You see: Last question / User answer / Current / Missing.

read_sql keys:
- name
- execute_query (yes/no -> true/false)
- write_count (yes/no -> true/false)

Important (latest constraint):
- When execute_query=true, schema/table are collected via dropdown in later steps.
  Do NOT try to extract result_schema/table_name from free-text unless the Last question explicitly asks for them.

Output JSON only: {"params": {...}}""".strip()

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
    model_name = "qwen3:8b"
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


def compute_missing(job: str, current: Dict[str, object]) -> List[str]:
    """
    More realistic "Missing" computation:
    - For read_sql, conditional keys only appear when the controlling boolean is true.
    - For other jobs, it's simply "keys that are missing/unknown".
    """
    if job != "read_sql":
        return [k for k in JOB_KEYS[job] if is_missing(k, current)]

    missing: List[str] = []
    # Always required for read_sql
    base_keys = ["name", "execute_query", "write_count"]
    for k in base_keys:
        if is_missing(k, current):
            missing.append(k)

    exec_val = current.get("execute_query", None)
    if exec_val is True:
        for k in ["result_schema", "table_name", "drop_before_create"]:
            if is_missing(k, current):
                missing.append(k)

    wc_val = current.get("write_count", None)
    if wc_val is True:
        for k in ["write_count_connection", "write_count_schema", "write_count_table"]:
            if is_missing(k, current):
                missing.append(k)

    return missing

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
    """
    Unified benchmark cases (50-60 range) focusing on realistic router behavior.

    Update based on latest info:
    - For read_sql: if execute_query=true, schema/table are asked via dropdown (user can't "stuff" them into free-text).
      Therefore we focus primarily on execute_query=false flows, and we do NOT include "yes + schema/table" stuffing cases.
    - Multi-fill still matters for:
      - read_sql name answers that also include "no saving" / "track row count"
      - write_data answers that include table + schema + connection together
      - email answers that include to + cc + subject + text together
    """
    cases: List[ExtractionTestCase] = []
    tid = 1

    # -------------------------
    # READ_SQL (execute_query false emphasis)
    # -------------------------
    q_name = "What should I name this read_sql job?"
    q_exec = "Should the job execute the query and save results to the database? (yes/no)"
    q_wc = "Should the job track the row count? (yes/no)"

    # A) Name only (6)
    for nm in ["read1172", "job133", "daily_orders", "report_balances", "rowsync", "customer_audit"]:
        cases.append(
            ExtractionTestCase(
                tid=f"T{tid:03d}",
                job="read_sql",
                description=f"read_sql: name plain ({nm})",
                last_question=q_name,
                user_answer=nm,
                current={"name": "", "execute_query": None, "write_count": None},
                expected_updates={"name": nm},
            )
        )
        tid += 1

    # B) Name + "no save" (6) => expect execute_query=false opportunistically
    for nm in ["read778900", "read9001", "read_daily", "read_clicks", "read_orders", "read_user_audit"]:
        cases.append(
            ExtractionTestCase(
                tid=f"T{tid:03d}",
                job="read_sql",
                description="read_sql: name + no saving => execute_query=false",
                last_question=q_name,
                user_answer=f"name it {nm}, no saving to db",
                current={"name": "", "execute_query": None, "write_count": None},
                expected_updates={"name": nm, "execute_query": False},
            )
        )
        tid += 1

    # C) execute_query question answered NO (6)
    for ans in [
        "no",
        "nope, preview only",
        "no, don't save",
        "do not save to db",
        "no saving please",
        "nah, just preview",
    ]:
        cases.append(
            ExtractionTestCase(
                tid=f"T{tid:03d}",
                job="read_sql",
                description=f"read_sql: execute_query=false ({ans})",
                last_question=q_exec,
                user_answer=ans,
                current={"name": "job_exec_test", "execute_query": None, "write_count": None},
                expected_updates={"execute_query": False},
            )
        )
        tid += 1

    # D) execute_query question answered NO + also mentions row count YES/NO (6)
    mix_answers = [
        ("no saving; yes track row count", False, True),
        ("no, but please track row count", False, True),
        ("no saving and no row count", False, False),
        ("preview only; don't track row count", False, False),
        ("do not save; keep the row count", False, True),
        ("no need to save; row count not necessary", False, False),
    ]
    for ans, ex, wc in mix_answers:
        cases.append(
            ExtractionTestCase(
                tid=f"T{tid:03d}",
                job="read_sql",
                description="read_sql: execute_query + write_count in one answer",
                last_question=q_exec,
                user_answer=ans,
                current={"name": "job_mix", "execute_query": None, "write_count": None},
                expected_updates={"execute_query": ex, "write_count": wc},
            )
        )
        tid += 1

    # E) write_count question answered YES/NO (6)
    for ans, expected in [
        ("yes, track it", True),
        ("sure, count rows", True),
        ("no, don't track it", False),
        ("nope, skip row count", False),
        ("not necessary", False),
        ("please track rows", True),
    ]:
        cases.append(
            ExtractionTestCase(
                tid=f"T{tid:03d}",
                job="read_sql",
                description="read_sql: write_count boolean",
                last_question=q_wc,
                user_answer=ans,
                current={"name": "job_wc", "execute_query": False, "write_count": None},
                expected_updates={"write_count": expected},
            )
        )
        tid += 1

    # -------------------------
    # WRITE_DATA (multi-fill common)
    # -------------------------
    q_w_name = "What should I name this write_data job?"
    q_w_table = "What table should I write the data to?"
    q_w_drop = "Should I 'drop' (remove and recreate), 'truncate' (clear data), or 'none' (append)?"
    q_w_conn = "Which connection should I use for writing?"

    # A) name (3)
    for nm in ["write75684", "write_daily", "write_clicks"]:
        cases.append(
            ExtractionTestCase(
                tid=f"T{tid:03d}",
                job="write_data",
                description="write_data: name",
                last_question=q_w_name,
                user_answer=nm,
                current={"name": "", "table": "", "schemas": "", "connection": "", "drop_or_truncate": ""},
                expected_updates={"name": nm},
            )
        )
        tid += 1

    # B) table question, answer includes table + schema + connection (6)
    combos = [
        ("write to test_table, schema ICC_TEST, connection ORACLE_10", {"table": "test_table", "schemas": "ICC_TEST", "connection": "ORACLE_10"}),
        ("table test_table schema ICC_TEST conn ORACLE_10", {"table": "test_table", "schemas": "ICC_TEST", "connection": "ORACLE_10"}),
        ('use "orders_2025" schema DW connection ORACLE_10', {"table": "orders_2025", "schemas": "DW", "connection": "ORACLE_10"}),
        ("test_table on ORACLE_10 in schema ICC_TEST", {"table": "test_table", "schemas": "ICC_TEST", "connection": "ORACLE_10"}),
        ("write to clicks_agg, schema ICC_TEST", {"table": "clicks_agg", "schemas": "ICC_TEST"}),
        ("table=customer_audit schema=DW", {"table": "customer_audit", "schemas": "DW"}),
    ]
    for ans, exp in combos:
        cases.append(
            ExtractionTestCase(
                tid=f"T{tid:03d}",
                job="write_data",
                description="write_data: multi-fill table/schema/connection",
                last_question=q_w_table,
                user_answer=ans,
                current={"name": "write_job", "table": "", "schemas": "", "connection": "", "drop_or_truncate": ""},
                expected_updates=exp,
            )
        )
        tid += 1

    # C) dot form schema.table (6)
    dot_cases = [
        ("ICC_TEST.test_table", {"schemas": "ICC_TEST", "table": "test_table"}),
        ("DW.orders_2025", {"schemas": "DW", "table": "orders_2025"}),
        ("ICC_TEST.clicks_agg", {"schemas": "ICC_TEST", "table": "clicks_agg"}),
        ("DW.customer_audit", {"schemas": "DW", "table": "customer_audit"}),
        ("ICC_TEST.sales_daily", {"schemas": "ICC_TEST", "table": "sales_daily"}),
        ("DW.report_balances", {"schemas": "DW", "table": "report_balances"}),
    ]
    for ans, exp in dot_cases:
        cases.append(
            ExtractionTestCase(
                tid=f"T{tid:03d}",
                job="write_data",
                description="write_data: schema.table dot split",
                last_question=q_w_table,
                user_answer=ans,
                current={"name": "write_dot", "table": "", "schemas": "", "connection": "ORACLE_10", "drop_or_truncate": ""},
                expected_updates=exp,
            )
        )
        tid += 1

    # D) drop/truncate/none (3)
    for ans, expected in [
        ("none, keep appending", "none"),
        ("truncate it", "truncate"),
        ("drop", "drop"),
    ]:
        cases.append(
            ExtractionTestCase(
                tid=f"T{tid:03d}",
                job="write_data",
                description="write_data: drop_or_truncate",
                last_question=q_w_drop,
                user_answer=ans,
                current={
                    "name": "write75684",
                    "connection": "ORACLE_10",
                    "schemas": "ICC_TEST",
                    "table": "test_table",
                    "drop_or_truncate": "",
                },
                expected_updates={"drop_or_truncate": expected},
            )
        )
        tid += 1

    # E) connection question where answer includes schema too (3)
    conn_cases = [
        ("use ORACLE_10 schema ICC_TEST", {"connection": "ORACLE_10", "schemas": "ICC_TEST"}),
        ("connection ORACLE_10", {"connection": "ORACLE_10"}),
        ("ORACLE_10 in schema DW", {"connection": "ORACLE_10", "schemas": "DW"}),
    ]
    for ans, exp in conn_cases:
        cases.append(
            ExtractionTestCase(
                tid=f"T{tid:03d}",
                job="write_data",
                description="write_data: connection (+ optional schema)",
                last_question=q_w_conn,
                user_answer=ans,
                current={"name": "write_conn", "table": "test_table", "schemas": "", "connection": "", "drop_or_truncate": "none"},
                expected_updates=exp,
            )
        )
        tid += 1

    # -------------------------
    # EMAIL (multi-fill common)
    # -------------------------
    q_e_name = "What should I name this email job?"
    q_e_to = "Who should I send the email to?"

    # A) name (3)
    for nm in ["email4857489", "email_daily", "email_report"]:
        cases.append(
            ExtractionTestCase(
                tid=f"T{tid:03d}",
                job="email",
                description="email: name",
                last_question=q_e_name,
                user_answer=nm,
                current={"name": "", "to": "", "subject": "", "text": "", "cc": ""},
                expected_updates={"name": nm},
            )
        )
        tid += 1

    # B) to only (6)
    for em in [
        "selcan.yukcu@pia-team.com",
        "alp@example.com",
        "ops@company.com",
        "finance@company.com",
        "team.lead@company.com",
        "data@company.com",
    ]:
        cases.append(
            ExtractionTestCase(
                tid=f"T{tid:03d}",
                job="email",
                description="email: to only",
                last_question=q_e_to,
                user_answer=f"send to {em}",
                current={"name": "email_job", "to": "", "subject": "", "text": "", "cc": ""},
                expected_updates={"to": em},
            )
        )
        tid += 1

    # C) to + subject + text (6)
    rich = [
        ('Send the text "You can check the data" with the subject "Data Info" to selcan.yukcu@pia-team.com',
         {"to":"selcan.yukcu@pia-team.com", "subject":"Data Info", "text":"You can check the data"}),
        ('to ops@company.com subject "Alert" text "Pipeline failed"',
         {"to":"ops@company.com", "subject":"Alert", "text":"Pipeline failed"}),
        ('subject "Daily Report" text "Attached" to finance@company.com',
         {"to":"finance@company.com", "subject":"Daily Report", "text":"Attached"}),
        ('send "Hello" with subject "Ping" to team.lead@company.com',
         {"to":"team.lead@company.com", "subject":"Ping", "text":"Hello"}),
        ('to data@company.com subject "ETL" text "Done"',
         {"to":"data@company.com", "subject":"ETL", "text":"Done"}),
        ('send to alp@example.com subject "Test" text "OK"',
         {"to":"alp@example.com", "subject":"Test", "text":"OK"}),
    ]
    for ans, exp in rich:
        cases.append(
            ExtractionTestCase(
                tid=f"T{tid:03d}",
                job="email",
                description="email: to + subject + text",
                last_question=q_e_to,
                user_answer=ans,
                current={"name": "email_job", "to": "", "subject": "", "text": "", "cc": ""},
                expected_updates=exp,
            )
        )
        tid += 1

    # D) to + cc + subject + text (6)
    rich_cc = [
        ('to selcan.yukcu@pia-team.com cc alp@example.com subject "Data Info" text "You can check the data"',
         {"to":"selcan.yukcu@pia-team.com", "cc":"alp@example.com", "subject":"Data Info", "text":"You can check the data"}),
        ('send to ops@company.com cc oncall@company.com subject "Alert" text "Pipeline failed"',
         {"to":"ops@company.com", "cc":"oncall@company.com", "subject":"Alert", "text":"Pipeline failed"}),
        ('to finance@company.com cc ceo@company.com subject "Daily" text "Report is ready"',
         {"to":"finance@company.com", "cc":"ceo@company.com", "subject":"Daily", "text":"Report is ready"}),
        ('cc boss@company.com to team.lead@company.com subject "Ping" text "Hi"',
         {"to":"team.lead@company.com", "cc":"boss@company.com", "subject":"Ping", "text":"Hi"}),
        ('to data@company.com cc qa@company.com subject "ETL" text "Done"',
         {"to":"data@company.com", "cc":"qa@company.com", "subject":"ETL", "text":"Done"}),
        ('to alp@example.com cc selcan.yukcu@pia-team.com subject "Test" text "OK"',
         {"to":"alp@example.com", "cc":"selcan.yukcu@pia-team.com", "subject":"Test", "text":"OK"}),
    ]
    for ans, exp in rich_cc:
        cases.append(
            ExtractionTestCase(
                tid=f"T{tid:03d}",
                job="email",
                description="email: to + cc + subject + text",
                last_question=q_e_to,
                user_answer=ans,
                current={"name": "email_cc", "to": "", "subject": "", "text": "", "cc": ""},
                expected_updates=exp,
            )
        )
        tid += 1

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
    p.add_argument("--model_size", choices={"1.7b", "8b", "7b"}, default="8b")
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
