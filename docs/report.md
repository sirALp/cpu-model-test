````markdown
# Router Slot / Parameter Extraction Benchmark  
*(Local LLMs via Ollama)*

**Author:** Alperen Tekin — alperen.tekin@intellica.net  
**Consultant:** Selcan Yükçü — selcan.yukcu@pia-team.com  
**Date:** 17 Dec 2025  
**Environment:** Local machine, Ollama, Python benchmark scripts  

---

## 1. Executive Summary

This report evaluates multiple local LLMs for a router-style parameter extraction task. The goal is to reliably extract one or more structured parameters from user answers and return strict JSON suitable for downstream automation.

Task difficulty was progressively increased across several benchmark groups:

- Single-step extraction (one question / answer)
- Multi-turn transcript (5 Q/A pairs)
- Opportunistic multi-field extraction when the user provides extra information
- Mixed “unified” scenarios combining both behaviors

Models are compared primarily on:

- **Accuracy** (per-test success and per-field update accuracy)
- **Robustness** (JSON parse stability, forbidden behaviors)
- **Latency** (average time per test / TTFT)

---

## Disclaimer

All tests were executed on **CPU only**, assuming no access to GPUs or vLLM-style acceleration.

---

## 2. Problem Definition

### 2.1 Task Overview

We simulate a router that asks the user for missing parameters required to configure jobs such as:

- `read_sql`
- `write_data`
- `email`

The LLM receives:

- The last router question  
- The user answer  
- The current parameter state (**Existing Params**)  
- Optionally a missing-parameters list or a multi-turn transcript  

The LLM **must return JSON only**:

```json
{ "params": { ... } }
````

### 2.2 Core Requirements

* Strict JSON-only output
* Never delete keys from `ExistingParams`
* Do not overwrite existing non-empty values (unless explicitly allowed)
* Extract values using defined rules for:

  * Identifiers (name / table)
  * Booleans (`execute_query`, `write_count`, etc.)
  * Job-specific allowed keys

---

## 3. Benchmark Design

### 3.1 Benchmark Groups (Experiments)

#### Experiment A — Single-step Extraction (Baseline)

**Input:**
Last question + User answer + Missing param + Existing params

**Objective:**
Extract only the parameter implied by the last question (or leave unchanged if uncertain).

**Examples:**

* Ask for job name → extract `name`
* Yes/no question → map to boolean

**Metrics:**
Success rate, parse errors, overwrite bugs, average latency.

---

#### Experiment B — Single-turn Multiple Parameters (5 Q/A Pairs)

**Input:**
Transcript (Q1/A1 … Q5/A5) + slot order + Existing params

**Objective:**
Fill multiple slots in sequence within a single call, respecting order and constraints.

---

#### Experiment C — Opportunistic Multi-field Extraction

**Motivation:**
Users often provide more information than asked.

**Example:**
Router asks for email recipient, user includes subject, text, and CC.

**Objective:**

* Always fill the asked field
* Opportunistically fill other missing fields **only if**:

  * They are explicitly mentioned
  * They are currently empty
  * Confidence is high

**Grouping Logic:**

* `read_sql` parameters first
* Then `write_data`
* Then `email`

---

#### Experiment D — Unified Mixed Scenarios (Final)

A realistic mixed benchmark where each test may be:

* Single-field answer (classic routing)
* Multi-field opportunistic answer

**System Constraints Applied:**

* If `execute_query=true`, schema/table is later selected via UI dropdown
* Opportunistic extraction mainly evaluated for:

  * `execute_query=false`
  * Boolean correctness
  * `write_data` / `email` fills

---

## 4. Models Evaluated

### 4.1 Candidate Models

* **Qwen3 1.7B** (baseline, fast)
* **Mistral 7B** (faster and more accurate than Qwen3 1.7B)
* **Llama2 7B** (tested)
* **Falcon 7B** (attempted, rejected)

### 4.2 Model Selection Rationale

Priorities:

* Local deployment feasibility
* Faster runtime than Qwen3 8B
* Comparable or better accuracy than Qwen3 1.7B

---

## 5. Evaluation Methodology

### 5.1 Metrics

**Primary Metrics**

* Test success rate
* Field-level accuracy (`correct_updates / expected_updates`)
* Parse error rate

**Bug Tracking**

* Overwrite bug
* Invalid key bug
* Unexpected update

**Performance Metrics**

* TTFT (Time to First Token)
* Total time per test
* Aggregate averages and worst-case observations

---

### 5.2 Common Failure Modes

* JSON truncation / unterminated strings
* Boolean confusion
* Wrong field selection
* Dotted identifier mishandling (`schema.table`)
* Over-conservatism (failing to update obvious fields)

---

## 6. Results Summary

### 6.1 Experiment A — Single-step Extraction

**Model:** Qwen3-1.7B

| Metric         | Value            |
| -------------- | ---------------- |
| Total tests    | 136              |
| Success        | 134 / 136 (~98%) |
| Field accuracy | ~98%             |
| Parse errors   | 0                |
| Avg time/test  | 13.65 sec        |

**Observation:**
Highly effective when only one parameter is expected. System prompt quality is critical for boolean extraction.

---

### 6.2 Experiment B — 5-turn Transcript

**Model:** Qwen3-1.7B

| Metric           | Value              |
| ---------------- | ------------------ |
| Total scenarios  | 150                |
| Scenario success | 96 / 150 (64%)     |
| Field accuracy   | 650 / 750 (86.67%) |
| Parse errors     | 10 / 150 (6.67%)   |
| Avg time/test    | 19.17 sec          |

---

### 6.3 Experiment C — Opportunistic Multi-field Extraction

**Model:** Qwen3-1.7B

| Metric         | Value          |
| -------------- | -------------- |
| Total tests    | 12             |
| Success        | 12 / 12 (100%) |
| Field accuracy | 24 / 24        |
| Avg time/test  | 21.11 sec      |

**Key Wins**

* Email: `to`, `subject`, `text`, `cc`
* Write operations: `table`, `schema`, `connection`

**Limitations**

* Dotted identifiers inconsistent
* Schema vs connection confusion

---

### 6.4 Experiment D — Unified Mixed Scenarios

| Model      | Success         | Field Accuracy    | Parse / Total Err | Avg Time (s) |
| ---------- | --------------- | ----------------- | ----------------- | ------------ |
| Qwen3-8B   | 64 / 72 (88.8%) | 115 / 132 (87.1%) | 8 / 8             | 75.865       |
| Qwen3-1.7B | 54 / 72 (75%)   | 109 / 132 (82.6%) | 4 / 18            | 17.132       |
| Mistral 7B | 65 / 72 (90.2%) | 124 / 132 (93.9%) | 0                 | 7.575        |
| Llama2 7B  | 36 / 72 (50%)   | 75 / 132 (56.8%)  | 2*                | 33.702       |

* Llama2 had no parse errors but had one overwrite bug and one unexpected update.

---

## 7. Model Comparison (High Level)

### Qwen3 1.7B

* **Pros:** Stable, accurate
* **Cons:** Slow, requires heavy prompt optimization

### Qwen3 8B

* **Pros:** Fast, stable
* **Cons:** Weak opportunistic fill, requires elaborate prompt engineering

### Mistral 7B

* **Pros:** Fastest, highest accuracy
* **Cons:** Longer TTFT (still faster overall than Qwen3 1.7B)

### Llama2 7B

* **Cons:** Slow, low accuracy, unoptimized prompts

### Falcon

* **Outcome:** Rejected (extreme latency, near-zero accuracy)

---

## 8. Discussion

### 8.1 What Worked Well

* Parameter grouping by job type
* Opportunistic fill for UX improvement
* Strict no-overwrite enforcement

### 8.2 What Still Breaks

* Parse stability issues
* Boolean ambiguity
* Dotted identifier handling

### 8.3 Practical Implications

Parsing reliability is as critical as extraction accuracy. Simple retry or enforced JSON wrappers can significantly reduce failures.

---

## 9. Observations — Prompt Optimization & Debugging

### 9.1 Explicit Role Definition

The model was positioned strictly as a **parameter extraction component**, not a conversational assistant.

### 9.2 Boolean Normalization Strategy

* Natural language mapped to canonical booleans
* Null or ambiguous values disallowed
* Explicit negation mapped to `false`

### 9.3 “Extract What You See” Principle

* No inference
* No guessing
* No context-based filling

### 9.4 Controlled Opportunistic Fill

Allowed only when multiple parameters are **explicitly stated** and clearly mappable.

### 9.5 Output Stability & Parsing Safety

* No trailing text
* No comments
* No partial JSON

After stabilization, parse errors dropped to zero.

---

## 10. Conclusion

The strictness of the router outline is a decisive factor in model success. When parameter fields are predefined, the most effective strategy is explicitly defining potential user responses for each parameter. For further details, refer to the `SYSTEM_PROMPT` sections in the provided code files.

---

*Mentioned test results and files are available in the repository.*
