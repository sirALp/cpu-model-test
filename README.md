# Job-Agent Router Extraction Benchmarks

This folder contains Python benchmarks to evaluate **single-step JSON parameter extraction**
for a router-style “job agent” flow (read_sql / write_data / email).

The scripts call a **local Ollama server** and measure:
- Success/fail vs expected updates
- Parse errors / overwrite bugs (when applicable)
- TTFT and total latency
- Per-test logs + a global summary

---

## Prerequisites

1) **Ollama running**
```bash
ollama serve
```

2) Models pulled (examples)
```bash
ollama pull qwen3:1.7b
ollama pull qwen3:8b
ollama pull mistral:7b   # if you want to test Mistral
ollama pull llama2:7b    # if you want to test Llama2
```

3) Python env
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install langchain-core langchain-community langchain-ollama
```

## Run Example
Example:
```bash
python benchmark_latest_unified_v2_falseonly_mistral.py --base_url http://localhost:11434
```
Limit Tests:
```bash
python benchmark_latest_unified_v2_falseonly_mistral.py --base_url http://localhost:11434 --max_tests 20
```
Run only one job group:
```bash
python benchmark_latest_unified_v2_falseonly_mistral.py --base_url http://localhost:11434 --only_job read_sql
```

---


## Tuning knobs (common)

* Context window:

```bash
--num_ctx 1024
```

* Output budget:

```bash
--num_predict 512
```

* Temperature (usually keep low):

```bash
--temperature 0.1
```

* Request timeout:

```bash
--timeout 120
```

---

## Outputs

Each run writes a `.log` file (and/or `.jsonl` depending on the script) containing:

* Per-test info (prompt input, raw model output, merged params, expected updates)
* Global summary (success rate, parse errors, average TTFT/total time)

Look for files like:

* `job_agent_unified_<model>_<timestamp>.log`
* `..._bench_<model>_<timestamp>.jsonl` (if enabled)

---

## Notes / Troubleshooting

* If you see `model not found`, pull it first:

```bash
ollama pull <model_name>
```

* If outputs are sometimes truncated / parse errors appear:

  * Increase `--num_predict`
  * Increase `--timeout`
  * Consider disabling streaming in the script (if supported)

* Very slow runtimes on some models usually means the model is not a good fit for strict JSON extraction
  under these prompts/settings.


