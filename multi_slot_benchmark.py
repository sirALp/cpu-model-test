#!/usr/bin/env python3
"""
Multi-turn (batch) slot-filling benchmark for Ollama (qwen3:1.7b / qwen3:8b).

Goal:
- Router asks multiple questions in sequence
- User answers each question
- LLM fills the *given slot order* (no need to infer which slot from which question)

Input per scenario:
- slot_order: ordered list of keys (and types)
- current: JSON with those keys ("" for unknown string, null for unknown boolean)
- transcript: Q1/A1, Q2/A2, ...

Output:
{"params": {...}}  (JSON only)

This script:
- runs many scenarios (generated)
- measures TTFT + total latency
- computes per-slot accuracy + scenario accuracy
- writes a compact JSONL log (1 line per scenario) + short console summary

No LangChain dependency: uses Ollama HTTP API directly.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests


# -----------------------------
# Prompt
# -----------------------------

SYSTEM_PROMPT = """
You are a STRICT JSON slot-filling assistant used inside a router.

You will receive:
- Slot order: an ordered list of parameter keys (and their types).
- Current params: a JSON object with those keys. Empty string means unknown string. null means unknown boolean.
- A transcript of multiple question/answer pairs: Q1/A1, Q2/A2, ...

Your job:
- Fill the slots IN THE GIVEN ORDER using ONLY the corresponding user answers.
- Do NOT try to infer which slot a question belongs to. The order is authoritative.
- For each slot i, use Ai to fill slot_order[i].
- If you cannot confidently extract a value for a slot, leave that slot unchanged.

Boolean mapping:
- TRUE: "yes", "yeah", "yep", "sure", "please do", "go ahead", "run it", "execute it", "do it"
- FALSE (map to false if answer contains any of these): "no", "nope", "nah", "don't", "do not", "don't do it", "not needed", "not necessary", "unnecessary", "no need", "no need to", "no requirement", "not required", "skip", "skip it", "skip this", "skip that", "leave it", "leave it out", "omit", "omit it", "disable", "turn off", "off", "preview only", "just preview", "only preview", "just show", "don't save", "do not save"


String extraction:
- Usually a single identifier: letters/digits/underscore (may be quoted).
- If the answer is a sentence, extract the most identifier-like token or the last quoted string.

GENERAL STRING EXTRACTION HEURISTICS (for schema/table/connection/name):
- Prefer the last quoted string if present (e.g. "ICC_TEST").
- Otherwise, extract the strongest IDENTIFIER token:
  - tokens matching [A-Za-z_][A-Za-z0-9_]* (may include digits/underscore)
- Ignore common filler words: ok, okay, sure, yes, no, please, use, let's, call, name, it, the, a, an, to, for, on, in, of.

CONTEXT HINTS (do NOT hardcode values):
- If the answer contains the word "schema", prefer the identifier closest to that word.
- If multiple identifiers exist, prefer the one that is uppercase-like (e.g. ICC_TEST) unless the user clearly provided lowercase (e.g. stage/prod).

Rules:
- Return JSON ONLY, with exactly one top-level key: "params"
- Never output type names as values ("string", "boolean", "varchar", "int", ...)
- Never delete keys.
"""

TYPE_NAME_STRINGS = {
    "string", "varchar", "text", "boolean", "bool", "int", "integer", "float", "double"
}


# -----------------------------
# Scenario model
# -----------------------------

@dataclass
class Scenario:
    sid: str
    description: str
    slot_order: List[str]
    slot_types: Dict[str, str]          # key -> "string"|"boolean"
    current: Dict[str, Any]             # "" for strings, None for booleans
    qa_pairs: List[Tuple[str, str]]     # [(question, answer), ...]
    expected: Dict[str, Any]            # final expected params


@dataclass
class ScenarioResult:
    sid: str
    description: str
    ok: bool
    per_slot_correct: int
    per_slot_total: int
    wrong_keys: List[str]
    type_name_bug_keys: List[str]
    parse_error: bool
    error: bool
    ttft: float
    total: float
    model_params: Dict[str, Any]
    raw_output: str


# -----------------------------
# Ollama client
# -----------------------------

def ollama_chat(
    base_url: str,
    model: str,
    system: str,
    user: str,
    *,
    stream: bool = True,
    timeout_s: float = 90.0,
    num_predict: int = 512,
    temperature: float = 0.1,
    num_ctx: int = 2048,
    seed: Optional[int] = None,
    think: bool = False,
) -> Tuple[str, float, float]:
    """Return (text, ttft_seconds, total_seconds)."""
    url = base_url.rstrip("/") + "/api/chat"
    payload: Dict[str, Any] = {
        "model": model,
        "stream": stream,
        "keep_alive": "3600s",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
            "think": think,  # qwen3 often accepts this under options
        },
    }
    if seed is not None:
        payload["options"]["seed"] = seed

    start = time.perf_counter()
    ttft: Optional[float] = None
    full = ""

    if stream:
        with requests.post(url, json=payload, stream=True, timeout=timeout_s) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                obj = json.loads(line.decode("utf-8"))
                msg = obj.get("message") or {}
                piece = msg.get("content") or ""
                if piece and ttft is None:
                    ttft = time.perf_counter() - start
                full += piece
                if obj.get("done"):
                    break

        total = time.perf_counter() - start
        if ttft is None:
            ttft = total

        # Fallback: retry non-stream if stream produced nothing
        if not full.strip():
            payload["stream"] = False
            start2 = time.perf_counter()
            r2 = requests.post(url, json=payload, timeout=timeout_s)
            r2.raise_for_status()
            obj2 = r2.json()
            msg2 = obj2.get("message") or {}
            full2 = (msg2.get("content") or "").strip()
            total2 = time.perf_counter() - start2
            return full2, ttft, total + total2

        return full.strip(), ttft, total

    r = requests.post(url, json=payload, timeout=timeout_s)
    r.raise_for_status()
    obj = r.json()
    msg = obj.get("message") or {}
    text = (msg.get("content") or "").strip()
    total = time.perf_counter() - start
    return text, total, total


# -----------------------------
# Prompt builder
# -----------------------------

def build_user_prompt(s: Scenario) -> str:
    lines: List[str] = []
    lines.append("Slot order (fill in this exact order using A1..An):")
    for i, k in enumerate(s.slot_order, start=1):
        t = s.slot_types.get(k, "string")
        lines.append(f"{i}) {k} ({t})")

    lines.append("")
    lines.append("Current params JSON (use as base; keep keys; fill blanks if possible):")
    lines.append(json.dumps(s.current, ensure_ascii=False))

    lines.append("")
    lines.append("Transcript:")
    for i, (q, a) in enumerate(s.qa_pairs, start=1):
        lines.append(f"Q{i}: {q}")
        lines.append(f"A{i}: {a}")

    lines.append("")
    lines.append('Return JSON ONLY: {"params": {...}}')
    return "\n".join(lines)


# -----------------------------
# Parsing + normalization
# -----------------------------

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
            return json.loads(text[start:end + 1])
        raise


def normalize_string(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
            s = s[1:-1].strip()
        return s
    return str(v).strip()


def normalize_bool(v: Any) -> Optional[bool]:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "t", "1", "yes", "y"):
            return True
        if s in ("false", "f", "0", "no", "n"):
            return False
    return None


def evaluate(s: Scenario, raw_text: str) -> Tuple[Dict[str, Any], List[str], List[str]]:
    obj = safe_parse_json(raw_text)
    params = obj.get("params")
    if not isinstance(params, dict):
        raise ValueError("Output JSON must be like {'params': {...}}")

    merged = dict(s.current)
    merged.update(params)

    wrong_keys: List[str] = []
    type_name_bug_keys: List[str] = []

    # type-name bugs
    for k, v in params.items():
        if isinstance(v, str) and v.strip().lower() in TYPE_NAME_STRINGS:
            type_name_bug_keys.append(k)

    # compare expected
    for k, exp in s.expected.items():
        t = s.slot_types.get(k, "string")
        got = merged.get(k, None)

        if t == "boolean":
            expn = normalize_bool(exp)
            gotn = normalize_bool(got)
        else:
            expn = normalize_string(exp)
            gotn = normalize_string(got)

        if expn != gotn:
            wrong_keys.append(k)

    return merged, wrong_keys, type_name_bug_keys


# -----------------------------
# Scenario generator (50)
# -----------------------------

def build_scenarios(seed: int = 123) -> List[Scenario]:
    import random
    rnd = random.Random(seed)

    slot_order = ["name", "execute_query", "write_count", "result_schema", "table_name"]
    slot_types = {
        "name": "string",
        "execute_query": "boolean",
        "write_count": "boolean",
        "result_schema": "string",
        "table_name": "string",
    }

    q = {
        "name": "What should I name this read_sql job?",
        "execute_query": "Should the job execute the query and save results to the database? (yes/no)",
        "write_count": "Should the job track the row count? (yes/no)",
        "result_schema": "Which schema should I write the results to?",
        "table_name": "What table should I write the results to?",
    }

    job_names = [
        "read1172", "job133", "daily_orders", "report_balances", "nightly_etl",
        "rowsync", "clicks_agg", "cust_audit", "trx_pull", "recon_01"
    ]
    schemas = ["ICC_TEST", "DW", "FIN", "RISK", "OPS", "stage", "prod"]
    tables = ["test_table", "RESULT_2025", "orders_daily", "balances_snap", "t_rowcount"]

    yes_vars = ["yes", "yeah, go ahead", "sure, execute it", "please do", "yep, run it"]
    no_vars = ["no", "nope", "don't", "not necessary", "skip it"]

    scenarios: List[Scenario] = []
    for i in range(1, 51):
        name = rnd.choice(job_names)
        exec_true = rnd.random() < 0.6
        wc_true = rnd.random() < 0.5
        schema = rnd.choice(schemas)
        table = rnd.choice(tables)

        exec_ans = rnd.choice(yes_vars if exec_true else no_vars)
        wc_ans = rnd.choice(yes_vars if wc_true else no_vars)

        current = {  # blanks
            "name": "",
            "execute_query": None,
            "write_count": None,
            "result_schema": "",
            "table_name": "",
        }

        qa_pairs = [
            (q["name"], f"Let's call it {name}" if rnd.random() < 0.5 else name),
            (q["execute_query"], exec_ans),
            (q["write_count"], wc_ans),
            (q["result_schema"], f"Use schema {schema}" if rnd.random() < 0.5 else schema),
            (q["table_name"], f'let\'s use "{table}"' if rnd.random() < 0.4 else table),
        ]

        expected = {
            "name": name,
            "execute_query": exec_true,
            "write_count": wc_true,
            "result_schema": schema,
            "table_name": table,
        }

        scenarios.append(
            Scenario(
                sid=f"S{i:03d}",
                description=f"5-slot fill (exec={exec_true}, wc={wc_true})",
                slot_order=slot_order,
                slot_types=slot_types,
                current=current,
                qa_pairs=qa_pairs,
                expected=expected,
            )
        )

    return scenarios


# -----------------------------
# Run scenario
# -----------------------------

def run_scenario(
    s: Scenario,
    *,
    base_url: str,
    model: str,
    stream: bool,
    timeout_s: float,
    num_predict: int,
    num_ctx: int,
    temperature: float,
    seed: Optional[int],
    think: bool,
) -> ScenarioResult:
    user_prompt = build_user_prompt(s)
    parse_error = False
    error = False
    raw_text = ""
    merged: Dict[str, Any] = dict(s.current)
    wrong_keys: List[str] = []
    type_bug_keys: List[str] = []

    try:
        raw_text, ttft, total = ollama_chat(
            base_url=base_url,
            model=model,
            system=SYSTEM_PROMPT,
            user=user_prompt,
            stream=stream,
            timeout_s=timeout_s,
            num_predict=num_predict,
            num_ctx=num_ctx,
            temperature=temperature,
            seed=seed,
            think=think,
        )
    except Exception as e:
        error = True
        ttft = 0.0
        total = 0.0
        raw_text = f"<<LLM ERROR: {e}>>"

    if not error:
        try:
            merged, wrong_keys, type_bug_keys = evaluate(s, raw_text)
        except Exception as e:
            parse_error = True
            wrong_keys = list(s.expected.keys())
            raw_text = f"<<PARSE ERROR: {e}>> RAW: {raw_text}"

    per_slot_total = len(s.expected)
    per_slot_correct = per_slot_total - len(set(wrong_keys))
    ok = (not error) and (not parse_error) and (not type_bug_keys) and (per_slot_correct == per_slot_total)

    return ScenarioResult(
        sid=s.sid,
        description=s.description,
        ok=ok,
        per_slot_correct=per_slot_correct,
        per_slot_total=per_slot_total,
        wrong_keys=wrong_keys,
        type_name_bug_keys=type_bug_keys,
        parse_error=parse_error,
        error=error,
        ttft=ttft,
        total=total,
        model_params=merged,
        raw_output=raw_text,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-turn slot-filling benchmark (single LLM call per scenario).")
    ap.add_argument("--model", default="qwen3:1.7b", help="Ollama model name (default: qwen3:1.7b)")
    ap.add_argument("--base_url", default="http://localhost:11434", help="Ollama base URL (default: http://localhost:11434)")
    ap.add_argument("--max_scenarios", type=int, default=50, help="How many scenarios to run (default: 50)")
    ap.add_argument("--seed", type=int, default=123, help="Scenario generator seed (default: 123)")
    ap.add_argument("--think", action="store_true", help="Enable qwen3 think mode (default: off)")
    ap.add_argument("--no_stream", action="store_true", help="Disable streaming (TTFT will approximate)")
    ap.add_argument("--timeout", type=float, default=90.0, help="HTTP timeout seconds (default: 90)")
    ap.add_argument("--num_predict", type=int, default=512, help="Max tokens (default: 512)")
    ap.add_argument("--num_ctx", type=int, default=2048, help="Context window (default: 2048)")
    ap.add_argument("--temperature", type=float, default=0.1, help="Temperature (default: 0.1)")
    ap.add_argument("--out", default=None, help="Write JSONL results to this file (default: auto)")
    args = ap.parse_args()

    scenarios = build_scenarios(seed=args.seed)[: args.max_scenarios]
    out_path = args.out or f"v3_multi_slot_bench_{args.model.replace(':','_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

    print(f"[INFO] Model={args.model} think={args.think} stream={not args.no_stream}")
    print(f"[INFO] Running {len(scenarios)} multi-turn scenarios (single call each)...")
    print(f"[INFO] Writing JSONL log to: {out_path}")

    ok_cnt = 0
    total_slots = 0
    correct_slots = 0
    ttfts: List[float] = []
    totals: List[float] = []
    parse_errs = 0
    errors = 0
    type_bug_scenarios = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for i, s in enumerate(scenarios, start=1):
            res = run_scenario(
                s,
                base_url=args.base_url,
                model=args.model,
                stream=(not args.no_stream),
                timeout_s=args.timeout,
                num_predict=args.num_predict,
                num_ctx=args.num_ctx,
                temperature=args.temperature,
                seed=args.seed + i,
                think=args.think,
            )

            ok_cnt += 1 if res.ok else 0
            total_slots += res.per_slot_total
            correct_slots += res.per_slot_correct
            ttfts.append(res.ttft)
            totals.append(res.total)
            parse_errs += 1 if res.parse_error else 0
            errors += 1 if res.error else 0
            type_bug_scenarios += 1 if bool(res.type_name_bug_keys) else 0

            # Console: compact
            wrong = (f" wrong={','.join(res.wrong_keys)}" if res.wrong_keys else "")
            print(f"[INFO] {i}/{len(scenarios)} {res.sid} ok={res.ok} slots={res.per_slot_correct}/{res.per_slot_total} ttft={res.ttft:.3f}s total={res.total:.3f}s{wrong}")

            # JSONL record (compact)
            rec = {
                "sid": res.sid,
                "desc": res.description,
                "ok": res.ok,
                "slots_correct": res.per_slot_correct,
                "slots_total": res.per_slot_total,
                "wrong_keys": res.wrong_keys,
                "parse_error": res.parse_error,
                "error": res.error,
                "type_name_bug_keys": res.type_name_bug_keys,
                "ttft": round(res.ttft, 3),
                "total": round(res.total, 3),
                # minimal debug payload for failures:
                "current": s.current,
                "model_params": res.model_params if not res.ok else None,
                "raw": res.raw_output if not res.ok else None,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    acc = (correct_slots / total_slots) if total_slots else 0.0
    avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0.0
    avg_total = sum(totals) / len(totals) if totals else 0.0

    print("\n" + "=" * 80)
    print("GLOBAL SUMMARY")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Scenarios: {len(scenarios)}")
    print(f"Scenario OK: {ok_cnt}/{len(scenarios)} ({(ok_cnt/len(scenarios))*100:.1f}%)")
    print(f"Slot accuracy: {correct_slots}/{total_slots} ({acc*100:.1f}%)")
    print(f"Parse errors: {parse_errs}")
    print(f"Errors: {errors}")
    print(f"Type-name bug scenarios: {type_bug_scenarios}")
    print(f"Avg TTFT: {avg_ttft:.3f}s")
    print(f"Avg total: {avg_total:.3f}s")
    print(f"Log: {out_path}")


if __name__ == "__main__":
    main()
