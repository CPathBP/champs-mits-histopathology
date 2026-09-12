"""Extract structured findings from the reports with a language model.

Sends one chat completion per case to an OpenAI-compatible server (vLLM)
and writes ``<out-dir>/per_case/<case>.json``. The prompt renders the
schema and the case's cleaned report text; the reply is constrained to the
JSON schema derived from the extraction schema. Temperature 0.

``run_meta.json`` in the output directory records the model, the schema,
the prompt and the inference settings; a later run into the same directory
with another schema or prompt is refused, so ``--skip-existing`` cannot mix
versions. A case whose reply cannot be parsed, whose reply names another
case, or that has no report text is written as ``<case>.error.txt``.
"""

import argparse
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from jinja2 import Template

from champs_pipeline.data_prep.extraction_meta import text_sha, write_or_check_run_meta
from champs_pipeline.data_prep.funnel import longest_text_per_case
from champs_pipeline.data_prep.morphology_labels import (
    build_schema_view, load_schema, output_json_schema,
)

log = logging.getLogger("extract_findings")
# A fixed input whose rendered prompt fingerprints the prompt as the model sees it.
FIXED_TEXT = "Liver: unremarkable."


def render_prompt(template, schema, case_id, text):
    return template.render(schema=build_schema_view(schema), case_id=case_id,
                           hetext=text, include_examples=True)


def parse_reply(text):
    """The JSON object in a reply; a code fence around it is tolerated."""
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    return json.loads(text)


def wait_for_server(base_url, model, timeout_s):
    """True once the server lists the model, False after the timeout."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            reply = requests.get(f"{base_url}/models", timeout=10)
            served = [m["id"] for m in reply.json().get("data", [])]
            if reply.ok and model in served:
                return True
        except (requests.RequestException, ValueError):
            pass
        time.sleep(15)
    return False


def server_settings(base_url, model):
    """The engine build and the served model's context length, as the server reports them."""
    root = base_url.rstrip("/").removesuffix("/v1")
    settings = {}
    try:
        settings["engine_version"] = requests.get(f"{root}/version", timeout=10).json()["version"]
    except (requests.RequestException, ValueError, KeyError):
        settings["engine_version"] = None
    try:
        models = requests.get(f"{base_url}/models", timeout=10).json()["data"]
        served = next(m for m in models if m["id"] == model)
        settings["max_model_len"] = served.get("max_model_len")
    except (requests.RequestException, ValueError, KeyError, StopIteration):
        settings["max_model_len"] = None
    return settings


def request_extraction(base_url, model, prompt, json_schema, max_tokens, timeout_s):
    """One constrained chat completion. Returns ``(record, meta, error)``."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "extraction", "schema": json_schema},
        },
    }
    started = time.monotonic()
    try:
        reply = requests.post(f"{base_url}/chat/completions", json=payload, timeout=timeout_s)
    except requests.RequestException as exc:
        return None, None, f"request failed: {exc}"
    if reply.status_code != 200:
        return None, None, f"HTTP {reply.status_code}: {reply.text[:400]}"
    body = reply.json()
    choice = body["choices"][0]
    usage = body.get("usage", {})
    meta = {
        "duration_s": round(time.monotonic() - started, 2),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": choice.get("finish_reason"),
    }
    try:
        record = parse_reply(choice["message"]["content"] or "")
    except (ValueError, TypeError) as exc:
        return None, meta, f"reply is not a JSON object: {exc}"
    return record, meta, None


def extract_case(job):
    case_id, prompt, args, json_schema = job
    record, meta, error = request_extraction(args.base_url, args.model, prompt, json_schema,
                                             args.max_tokens, args.timeout_s)
    if record is not None and record.get("case_id") not in (None, case_id):
        error = f"reply names case {record.get('case_id')!r}"
    return case_id, record, meta, error


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reports-csv", required=True)
    ap.add_argument("--schema", type=Path, required=True)
    ap.add_argument("--prompt", type=Path, required=True)
    ap.add_argument("--case-list", type=Path, required=True, help="One case id per line.")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--base-url", required=True, help="For example http://host:8000/v1")
    ap.add_argument("--model", required=True, help="The served model name.")
    ap.add_argument("--image", default=None, help="Container image of the server, for the record.")
    ap.add_argument("--workers", type=int, default=16, help="Concurrent requests.")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--timeout-s", type=int, default=900)
    ap.add_argument("--wait-server-s", type=int, default=1500)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not wait_for_server(args.base_url, args.model, args.wait_server_s):
        raise SystemExit(f"server at {args.base_url} does not serve {args.model}")
    schema = load_schema(args.schema)
    template = Template(args.prompt.read_text())
    json_schema = output_json_schema(schema)
    settings = {
        **server_settings(args.base_url, args.model),
        "container_image": args.image,
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "decoding": "constrained to the JSON schema of output_json_schema (response_format)",
        "rendered_prompt_sha": text_sha(render_prompt(template, schema, "FIXED", FIXED_TEXT)),
        "rendered_prompt_input": FIXED_TEXT,
    }
    write_or_check_run_meta(args.out_dir, model=args.model, schema=schema,
                            schema_path=args.schema, prompt_path=args.prompt, settings=settings)
    per_case_dir = args.out_dir / "per_case"
    per_case_dir.mkdir(parents=True, exist_ok=True)

    record_keys = [key for key in json_schema["properties"] if key != "case_id"]
    texts = longest_text_per_case(pd.read_csv(args.reports_csv, dtype=str))
    case_ids = [line.strip() for line in args.case_list.read_text().splitlines() if line.strip()]
    if args.skip_existing:
        case_ids = [c for c in case_ids if not (per_case_dir / f"{c}.json").exists()]
    jobs = []
    n_ok = n_failed = 0
    for case_id in case_ids:
        text = texts.get(case_id, "")
        if not text:
            log.warning("%s has no report text", case_id)
            (per_case_dir / f"{case_id}.error.txt").write_text("no report text")
            n_failed += 1
            continue
        jobs.append((case_id, render_prompt(template, schema, case_id, text), args, json_schema))
    log.info("%d cases over %d workers", len(jobs), args.workers)

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed(pool.submit(extract_case, job) for job in jobs):
            case_id, record, meta, error = future.result()
            if error:
                log.warning("%s failed: %s", case_id, error)
                detail = f"{error}\n{json.dumps(meta)}" if meta else error
                (per_case_dir / f"{case_id}.error.txt").write_text(detail)
                n_failed += 1
                continue
            out = {"case_id": case_id}
            out.update({key: record.get(key, []) for key in record_keys})
            out["_meta"] = {"model": args.model, **meta}
            (per_case_dir / f"{case_id}.json").write_text(
                json.dumps(out, indent=2, ensure_ascii=False))
            n_ok += 1
            log.info("%s ok (%.1f s, %s tokens out)", case_id, meta["duration_s"],
                     meta["completion_tokens"])

    summary = {"model": args.model, "case_list": args.case_list.name, "n_ok": n_ok,
               "n_failed": n_failed, "elapsed_s": round(time.monotonic() - started, 1)}
    summary_path = args.out_dir / f"extraction_summary_{args.case_list.stem}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info("done: %s", summary)


if __name__ == "__main__":
    main()
