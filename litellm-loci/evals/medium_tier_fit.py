"""Is the local resident model fit for the jobs medium-tier gets?

Two layers, each run against a local-first proxy (dev) and a cloud-first proxy (prod) of the same config:

  capabilities  tool calling (selection, arguments, result follow-up), streaming, JSON mode, system-prompt
                adherence, multi-turn memory, long-context retrieval, over-context fallback, concurrency
  jobs          the work the MEDIUM class carries: code review, log root-cause, SQL generation, shell
                commands, ordered plans, and Loci-style structured verification

Objective checks only. Requests carry metadata.loci_internal_call so nothing is grounded or written to memory.

    python medium_tier_fit.py --config config.yaml --local-first http://127.0.0.1:4001 --cloud-first http://127.0.0.1:4000 --out r.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

INTERNAL = {"loci_internal_call": True}
UA = "OpenAI/Python 1.40.0"


def _key(config: str) -> str:
    import yaml
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    key = str((cfg.get("general_settings") or {}).get("master_key", ""))
    if key.startswith("os.environ/"):
        name = key.split("/", 1)[1]
        key = os.environ.get(name, "")
        if not key:
            for line in Path(config).with_name(".env").read_text(encoding="utf-8").splitlines():
                k, _, v = line.partition("=")
                if k.strip() == name:
                    key = v.strip().strip("'\"")
    return key


class Arm:
    def __init__(self, name: str, base: str, key: str, group: str = "medium-tier"):
        self.name, self.base, self.key, self.group = name, base, key, group

    def chat(self, messages, timeout=240, stream=False, **extra):
        body = {"model": self.group, "messages": messages, "max_tokens": 600, "metadata": INTERNAL, "stream": stream}
        body.update(extra)
        req = urllib.request.Request(f"{self.base}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}",
                                              "User-Agent": UA})
        t = time.time()
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    served = r.headers.get("x-litellm-model-id", "")
                    if stream:
                        chunks, text, finish = 0, "", None
                        for raw in r:
                            line = raw.decode("utf-8", "replace").strip()
                            if line.startswith("data:") and line != "data: [DONE]":
                                d = json.loads(line[5:])
                                chunks += 1
                                ch = (d.get("choices") or [{}])[0]
                                text += (ch.get("delta") or {}).get("content") or ""
                                finish = ch.get("finish_reason") or finish
                        return {"text": text, "chunks": chunks, "finish": finish, "served": served, "s": time.time() - t}
                    d = json.loads(r.read())
                    msg = d["choices"][0]["message"]
                    return {"text": msg.get("content") or "", "tool_calls": msg.get("tool_calls") or [],
                            "served": served or d.get("model", ""), "s": time.time() - t, "usage": d.get("usage")}
            except urllib.error.HTTPError as e:
                return {"text": "", "tool_calls": [], "err": f"HTTP {e.code} {e.read()[:140].decode('utf-8', 'replace')}",
                        "served": "", "s": time.time() - t}
            except (urllib.error.URLError, ConnectionError) as e:
                if attempt == 2:
                    return {"text": "", "tool_calls": [], "err": repr(e)[:140], "served": "", "s": time.time() - t}
                time.sleep(15)


def _json(text):
    m = re.search(r"\{.*\}|\[.*\]", text or "", re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except ValueError:
        return None


TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
                                      "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "run_sql", "description": "Run a read-only SQL query against the analytics database",
                                      "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "search_docs", "description": "Search the documentation by keywords",
                                      "parameters": {"type": "object", "properties": {"keywords": {"type": "string"}}, "required": ["keywords"]}}},
]


def _tool(res, name):
    for tc in res.get("tool_calls") or []:
        fn = tc.get("function") or {}
        if fn.get("name") == name:
            try:
                return json.loads(fn.get("arguments") or "{}")
            except ValueError:
                return None
    return None


def filler(tokens: int, needle: str, at: float = 0.5) -> str:
    sentence = "The quarterly inventory report lists warehouse codes, shelf counts and routine maintenance notes. "
    n = max(1, int(tokens * 4 / len(sentence)))
    parts = [sentence] * n
    parts.insert(int(n * at), f"IMPORTANT FACT: {needle} ")
    return "".join(parts)


def capabilities(arm: Arm) -> dict:
    r = {}
    x = arm.chat([{"role": "user", "content": "What is the weather in Lisbon right now?"}], tools=TOOLS)
    a = _tool(x, "get_weather")
    r["tool_call_basic"] = bool(a and str(a.get("city", "")).lower().startswith("lisbon"))
    x = arm.chat([{"role": "user", "content": "How many orders were placed in March? Use the database."}], tools=TOOLS)
    a = _tool(x, "run_sql")
    r["tool_choice_and_args"] = bool(a and "select" in str(a.get("query", "")).lower())
    msgs = [{"role": "user", "content": "What's the weather in Lisbon?"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function",
             "function": {"name": "get_weather", "arguments": "{\"city\": \"Lisbon\"}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp_c\": 23, \"sky\": \"clear\"}"}]
    x = arm.chat(msgs, tools=TOOLS)
    r["tool_result_followup"] = "23" in x["text"]
    x = arm.chat([{"role": "user", "content": "Count from 1 to 8 separated by spaces."}], stream=True)
    r["streaming"] = x.get("chunks", 0) > 2 and bool(x["text"].strip()) and x.get("finish") in ("stop", "length")
    x = arm.chat([{"role": "user", "content": 'Return a JSON object with keys "a" (1) and "b" (list of two strings).'}],
                 response_format={"type": "json_object"})
    j = _json(x["text"])
    r["json_mode"] = isinstance(j, dict) and j.get("a") == 1 and isinstance(j.get("b"), list)
    x = arm.chat([{"role": "system", "content": "You must answer only in French, in one short sentence."},
                  {"role": "user", "content": "What is the capital of Spain?"}])
    r["system_prompt_adherence"] = bool(re.search(r"\b(la|est|capitale|madrid)\b", x["text"].lower())) and "capital of" not in x["text"].lower()
    x = arm.chat([{"role": "user", "content": "My dog is called Biscuit."}, {"role": "assistant", "content": "Nice to meet Biscuit!"},
                  {"role": "user", "content": "What is my dog called? One word."}])
    r["multi_turn_memory"] = "biscuit" in x["text"].lower()
    for tokens in (20000, 45000):
        x = arm.chat([{"role": "user", "content": filler(tokens, "the maintenance window code is ORCHID-4471.") +
                       "\n\nWhat is the maintenance window code? Reply with the code only."}], timeout=300)
        r[f"long_context_{tokens // 1000}k"] = "ORCHID-4471" in x["text"]
        r[f"long_context_{tokens // 1000}k_served"] = x.get("served", "")
    x = arm.chat([{"role": "user", "content": filler(78000, "the loading dock bay code is HERON-9.") +
                   "\n\nWhat is the loading dock bay code? Code only."}], timeout=300)
    r["overflow_answered"] = "HERON-9" in x["text"] and not x.get("err")
    r["overflow_served_by"] = x.get("served", "") or x.get("err", "")
    with ThreadPoolExecutor(6) as ex:
        outs = list(ex.map(lambda i: arm.chat([{"role": "user", "content": f"Reply with the number {i} and nothing else."}]), range(6)))
    r["concurrent_6"] = all(str(i) in o["text"] for i, o in enumerate(outs))
    r["concurrent_p95_s"] = round(sorted(o["s"] for o in outs)[-1], 2)
    return r


def _sql_ok(answer: str, expect) -> bool:
    m = re.search(r"```\w*\n(.*?)```", answer, re.S)
    q = (m.group(1) if m else answer).strip().rstrip(";")
    db = sqlite3.connect(":memory:")
    db.executescript("create table orders(id int, customer text, amount real, placed date);"
                     "insert into orders values (1,'ann',120.0,'2026-03-02'),(2,'bob',80.5,'2026-03-15'),"
                     "(3,'ann',42.0,'2026-04-01'),(4,'cy',300.0,'2026-03-20'),(5,'bob',10.0,'2026-04-11');")
    try:
        return [tuple(r) for r in db.execute(q).fetchall()] == expect
    except Exception:
        return False


def _order_ok(answer: str, names: list, before: list) -> bool:
    pos = {}
    for n in names:
        m = re.search(re.escape(n), answer, re.I)
        if not m:
            return False
        pos[n] = m.start()
    return all(pos[a] < pos[b] for a, b in before)


def jobs() -> list:
    J = []
    reviews = [("def avg(xs):\n    return sum(xs) / len(xs)\n", r"empty|zero|ZeroDivision|len\(xs\) == 0|divide by zero"),
               ("def add_item(item, bucket=[]):\n    bucket.append(item)\n    return bucket\n", r"mutable|default argument|shared|same list"),
               ("import sqlite3\ndef find(db, name):\n    return db.execute(f\"select * from users where name = '{name}'\").fetchall()\n", r"inject"),
               ("def read_all(path):\n    f = open(path)\n    data = f.read()\n    return data\n", r"close|context manager|with open|leak")]
    for i, (code, pat) in enumerate(reviews):
        J.append((f"review-{i}", f"Review this Python code and name the main defect in one or two sentences.\n\n```python\n{code}```",
                  lambda t, pat=pat: re.search(pat, t, re.I) is not None))
    logs = [("kubelet: Failed to pull image 'registry.local/app:1.4': 401 Unauthorized\nPod stuck in ImagePullBackOff.", r"credential|auth|secret|unauthori|pull secret|registry login"),
            ("nginx: [emerg] bind() to 0.0.0.0:80 failed (98: Address already in use)", r"already in use|port 80|another process|conflict"),
            ("OOMKilled exit code 137, container memory limit 256Mi, usage peaked 255Mi.", r"memory limit|out of memory|oom|raise.*limit|increase.*memory"),
            ("ssh: connect to host 10.0.0.5 port 22: Connection timed out", r"firewall|unreachable|network|host.*down|routing|security group|timed out")]
    for i, (log, pat) in enumerate(logs):
        J.append((f"rootcause-{i}", f"What is the most likely root cause? One sentence.\n\n{log}", lambda t, pat=pat: re.search(pat, t, re.I) is not None))
    sql = [("Table orders(id, customer, amount, placed). Write one SQLite query returning customer and total amount per customer, highest total first. Query only.",
            [("ann", 162.0), ("cy", 300.0), ("bob", 90.5)]),
           ("Table orders(id, customer, amount, placed). Write one SQLite query for the number of orders placed in March 2026. Query only.", [(3,)]),
           ("Table orders(id, customer, amount, placed). Write one SQLite query returning the customers (distinct, alphabetical) with any order over 100. Query only.", [("ann",), ("cy",)])]
    sql[0] = (sql[0][0], [("cy", 300.0), ("ann", 162.0), ("bob", 90.5)])
    for i, (p, exp) in enumerate(sql):
        J.append((f"sql-{i}", p, lambda t, exp=exp: _sql_ok(t, exp)))
    sh = [("Give one shell command that finds files larger than 100MB under /var/log. Command only.", r"find\s+/var/log.*-size\s*\+100\s*M"),
          ("Give one shell command that shows the 5 largest directories under /srv, human readable. Command only.", r"du\s.*(-h|-sh).*sort.*(-h|-rh|-hr).*(head|tail)"),
          ("Give one shell command to count lines containing ERROR in app.log. Command only.", r"(grep\s+(-c\s+)?['\"]?ERROR['\"]?.*(app\.log).*(wc\s+-l)?|grep\s+-c)")]
    for i, (p, pat) in enumerate(sh):
        J.append((f"shell-{i}", p, lambda t, pat=pat: re.search(pat, t, re.S) is not None))
    plans = [("Order these steps for deploying a service, one per line, numbered: run smoke tests, build image, provision database, push image, apply manifests.",
              ["build", "push", "provision", "apply", "smoke"], [("build", "push"), ("push", "apply"), ("apply", "smoke"), ("provision", "apply")]),
             ("Order these steps for rotating a leaked API key: revoke the old key, create a new key, update the secret store, restart consumers, audit usage logs.",
              ["create", "update", "restart", "revoke", "audit"], [("create", "update"), ("update", "restart"), ("restart", "revoke")])]
    for i, (p, names, before) in enumerate(plans):
        J.append((f"plan-{i}", p, lambda t, n=names, b=before: _order_ok(t, n, b)))
    ver = [("Claim: The service restarts nightly at 02:00.", "Evidence: cron entry '0 2 * * * systemctl restart svc' present in /etc/cron.d/svc.", True),
           ("Claim: The database is encrypted at rest.", "Evidence: config shows encryption_at_rest: true and a key id.", True),
           ("Claim: Backups run hourly.", "Evidence: backup job schedule is '0 3 * * *' (daily).", False),
           ("Claim: The API requires authentication.", "Evidence: curl to /v1/items without a token returned 200 with data.", False),
           ("Claim: TLS 1.0 is disabled.", "Evidence: nginx ssl_protocols TLSv1.2 TLSv1.3;", True),
           ("Claim: Logs are retained 90 days.", "Evidence: logrotate keeps 14 rotated files, rotated daily.", False)]
    for i, (c, e, ok) in enumerate(ver):
        J.append((f"verify-{i}", f"{c}\n{e}\nDecide whether the evidence supports the claim. Reply JSON only: {{\"supported\": true|false, \"reason\": \"short\"}}",
                  lambda t, ok=ok: (lambda j: isinstance(j, dict) and j.get("supported") is ok)(_json(t))))
    return J


def run_jobs(arm: Arm, samples: int) -> dict:
    out = {}
    def one(job):
        name, prompt, check = job
        res = []
        for _ in range(samples):
            x = arm.chat([{"role": "user", "content": prompt}], max_tokens=700)
            res.append({"pass": bool(x["text"]) and bool(check(x["text"])), "s": x["s"], "served": x.get("served", ""),
                        "err": x.get("err"), "text": x["text"][:160]})
        return name, res
    with ThreadPoolExecutor(2) as ex:
        for name, res in ex.map(one, jobs()):
            out[name] = res
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--local-first", required=True)
    ap.add_argument("--cloud-first", required=True)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    key = _key(a.config)
    result = {}
    for name, base in (("local-first", a.local_first), ("cloud-first", a.cloud_first)):
        arm = Arm(name, base, key)
        caps = capabilities(arm)
        jr = run_jobs(arm, a.samples)
        flat = [s for v in jr.values() for s in v]
        by = {}
        for k, v in jr.items():
            fam = k.split("-")[0]
            p, n = by.get(fam, (0, 0))
            by[fam] = (p + sum(s["pass"] for s in v), n + len(v))
        result[name] = {"capabilities": caps, "jobs": jr,
                        "job_summary": {"pass": f"{sum(s['pass'] for s in flat)}/{len(flat)}",
                                        "by_family": {k: f"{p}/{n}" for k, (p, n) in by.items()},
                                        "p50_s": round(statistics.median(s["s"] for s in flat), 2),
                                        "served_by": sorted({s["served"] for s in flat})}}
    Path(a.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    for name, v in result.items():
        print(name, json.dumps({"capabilities": v["capabilities"], "jobs": v["job_summary"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
