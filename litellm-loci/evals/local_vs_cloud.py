"""Graded comparison of the local resident model against the cloud tiers on a fixed prompt suite.

Objective checks wherever possible (exact answers, JSON schema, executed unit tests, format constraints);
a cloud judge only for free-text summaries. Prompts are generic and carry no private data.

    python local_vs_cloud.py --config config.yaml --local http://127.0.0.1:11435 --gateway http://127.0.0.1:4000 \
        --judge-group complex-tier --out results.json

Model names never appear here: the local model is whatever the resident proxy serves, the cloud arms are
gateway model groups. Requests carry metadata.loci_internal_call so nothing is grounded or written to memory.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

INTERNAL = {"loci_internal_call": True}


def _json(text: str):
    m = re.search(r"\{.*\}|\[.*\]", text, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except ValueError:
        return None


def _num(text: str):
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(nums[-1]) if nums else None


def _code(text: str) -> str:
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    return m.group(1) if m else text


def run_code(text: str, tests: str) -> bool:
    with tempfile.TemporaryDirectory() as d:
        Path(d, "t.py").write_text(_code(text) + "\n\n" + tests, encoding="utf-8")
        try:
            r = subprocess.run([sys.executable, "-I", "t.py"], cwd=d, capture_output=True, timeout=10)
        except subprocess.TimeoutExpired:
            return False
        return r.returncode == 0


def tasks() -> list:
    t = []
    ex = [("Extract JSON with keys name, age, city from: 'Maria Lopez, 34, lives in Porto.' Reply with JSON only.",
           lambda j: j == {"name": "Maria Lopez", "age": 34, "city": "Porto"}),
          ("Extract JSON with keys item, quantity, unit_price (numbers) from: 'Order: 12 widgets at $3.50 each.' JSON only.",
           lambda j: j and j.get("item", "").lower().startswith("widget") and j.get("quantity") == 12 and j.get("unit_price") == 3.5),
          ("Return a JSON array of the distinct colors mentioned: 'The red car passed a blue van and another red car, then a green bus.' JSON only.",
           lambda j: isinstance(j, list) and sorted(c.lower() for c in j) == ["blue", "green", "red"]),
          ("Extract JSON {\"date\": \"YYYY-MM-DD\"} from: 'The meeting moved to March 5th, 2027.' JSON only.",
           lambda j: j == {"date": "2027-03-05"}),
          ("From the log line 'ERROR 2026-01-02T10:00:00Z disk=/dev/sda1 used=93%' return JSON with keys level, disk, used_percent (number). JSON only.",
           lambda j: j and j.get("level", "").upper() == "ERROR" and j.get("disk") == "/dev/sda1" and j.get("used_percent") == 93),
          ("Return JSON {\"valid\": true|false} saying whether 'a@b.co' looks like an email address. JSON only.",
           lambda j: j == {"valid": True})]
    for i, (p, c) in enumerate(ex):
        t.append(dict(id=f"extract-{i}", cat="extraction", prompt=p, check=lambda x, c=c: bool(c(_json(x)))))
    cls = [("Is this review positive or negative? 'The battery died within a day and support ignored me.' One word.", "negative"),
           ("Is this review positive or negative? 'Setup took two minutes and it just works.' One word.", "positive"),
           ("Which category fits: 'How do I reset my password?' Options: billing, account, shipping. One word.", "account"),
           ("Which category fits: 'My parcel has not arrived after 10 days.' Options: billing, account, shipping. One word.", "shipping"),
           ("Is this message spam or not_spam? 'You WON a free cruise!!! click now' One word.", "spam"),
           ("Which language is this? 'Où est la bibliothèque ?' One word in English.", "french")]
    for i, (p, a) in enumerate(cls):
        t.append(dict(id=f"classify-{i}", cat="classification", prompt=p,
                      check=lambda x, a=a: a in x.lower() and not (a == "spam" and "not_spam" in x.lower())))
    rs = [("A train leaves at 14:20 and arrives at 17:05. How many minutes is the trip? Answer with the number.", 165),
          ("If 3 pens cost $4.50, how many dollars do 8 pens cost? Answer with the number.", 12),
          ("A list has 5 numbers averaging 12. A sixth number is added and the average becomes 14. What is the sixth number? Number only.", 24),
          ("How many weekdays (Mon-Fri) are there in a 28 day period starting on a Monday? Number only.", 20),
          ("What is 17 * 23? Number only.", 391),
          ("A tank holds 200 L and drains 8 L per minute while 3 L per minute flows in. After how many minutes is it empty? Number only.", 40)]
    for i, (p, a) in enumerate(rs):
        t.append(dict(id=f"reason-{i}", cat="reasoning", prompt=p, check=lambda x, a=a: _num(x) == a))
    code = [("Write a Python function is_palindrome(s) that ignores case and non-alphanumerics. Code only. Pure Python, standard library only, no API calls.",
             "assert is_palindrome('A man, a plan, a canal: Panama') and not is_palindrome('abc') and is_palindrome('')"),
            ("Write a Python function dedupe(xs) returning the list with duplicates removed, keeping first occurrences. Code only. Pure Python, standard library only, no API calls.",
             "assert dedupe([3,1,3,2,1]) == [3,1,2] and dedupe([]) == []"),
            ("Write a Python function fizz(n) returning 'Fizz' if n%3==0, 'Buzz' if n%5==0, 'FizzBuzz' if both, else str(n). Code only. Pure Python, standard library only, no API calls.",
             "assert [fizz(i) for i in (1,3,5,15)] == ['1','Fizz','Buzz','FizzBuzz']"),
            ("Write a Python function merge_intervals(iv) merging overlapping [start,end] pairs, returning sorted merged list of lists. Code only. Pure Python, standard library only, no API calls.",
             "assert merge_intervals([[1,3],[2,6],[8,10]]) == [[1,6],[8,10]] and merge_intervals([]) == []"),
            ("Write a Python function word_count(s) returning a dict of lowercase word -> count, splitting on whitespace. Code only. Pure Python, standard library only, no API calls.",
             "assert word_count('a b A') == {'a':2,'b':1}"),
            ("Write a Python function chunk(xs, n) splitting a list into consecutive lists of length n (last may be shorter). Code only. Pure Python, standard library only, no API calls.",
             "assert chunk([1,2,3,4,5],2) == [[1,2],[3,4],[5]] and chunk([],3) == []")]
    for i, (p, tests) in enumerate(code):
        t.append(dict(id=f"code-{i}", cat="code", prompt=p, check=lambda x, tests=tests: run_code(x, tests)))
    fol = [("Write exactly three bullet points, each starting with '- ', about why backups matter. No other text.",
            lambda x: len([l for l in x.strip().splitlines() if l.strip()]) == 3 and all(l.strip().startswith("- ") for l in x.strip().splitlines() if l.strip())),
           ("Reply with a single sentence of at most 12 words about the sea. No quotes.",
            lambda x: len(x.strip().split()) <= 12 and "\n" not in x.strip()),
           ("List the first five planets from the sun, comma-separated, lowercase, no other text.",
            lambda x: [p.strip() for p in x.strip().lower().rstrip('.').split(",")] == ["mercury", "venus", "earth", "mars", "jupiter"]),
           ("Reply with the word OK in uppercase and nothing else.", lambda x: x.strip().strip('."\'') == "OK"),
           ("Give a title in Title Case of 3 to 5 words about gardening. Title only.",
            lambda x: 3 <= len(x.strip().split()) <= 5 and all(w[0].isupper() for w in x.strip().split())),
           ("Write two sentences about rain; the second must begin with the word 'However'.",
            lambda x: len(re.findall(r"[.!?]", x)) >= 2 and re.search(r"(^|[.!?]\s+)However\b", x) is not None)]
    for i, (p, c) in enumerate(fol):
        t.append(dict(id=f"follow-{i}", cat="instruction", prompt=p, check=lambda x, c=c: bool(c(x))))
    text = ("The city council voted 7-2 on Tuesday to extend the bike lane network by 14 kilometres, funded by a "
            "$3.2 million transport grant. Opponents argued the plan removes 120 parking spaces downtown; supporters "
            "cited a 30 percent rise in cycling commuters over two years. Construction begins in spring and finishes "
            "within eighteen months.")
    for i, q in enumerate(["Summarise in one sentence.", "Summarise in under 25 words.", "List the three key facts as bullets.",
                           "What did opponents and supporters each argue? Two sentences.", "Write a one-line headline.",
                           "Explain the decision to a ten-year-old in two sentences."]):
        t.append(dict(id=f"summ-{i}", cat="summary", prompt=f"{q}\n\nText: {text}", check=None, source=text))
    return t


def _post(url: str, body: dict, key: str = "", timeout: float = 180.0) -> dict:
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, ConnectionError):
            if attempt == 3:        # a proxy restart is a ~1 minute window: ride through it, do not score it as a wrong answer
                raise
            time.sleep(15)


def ask_local(base: str, prompt: str) -> tuple:
    t = time.time()
    try:
        d = _post(f"{base}/api/chat", {"model": "resident", "messages": [{"role": "user", "content": prompt}], "stream": False,
                                       "think": False, "options": {"num_predict": 1500, "temperature": 0.2}})
        return (d.get("message") or {}).get("content", "") or "", time.time() - t, None
    except Exception as exc:
        return "", time.time() - t, repr(exc)[:120]


def ask_gateway(base: str, key: str, group: str, prompt: str, max_tokens: int = 1500) -> tuple:
    t = time.time()
    try:
        d = _post(f"{base}/v1/chat/completions", {"model": group, "max_tokens": max_tokens, "temperature": 0.2,
                                                   "messages": [{"role": "user", "content": prompt}], "metadata": INTERNAL}, key)
        return d["choices"][0]["message"].get("content") or "", time.time() - t, None
    except Exception as exc:
        return "", time.time() - t, repr(exc)[:120]


def judge(base: str, key: str, group: str, task: dict, answers: dict) -> dict:
    """Blind 1-5 grade of each answer against a rubric; answers are shuffled labels, not arm names."""
    labels = {arm: chr(65 + i) for i, arm in enumerate(sorted(answers))}
    body = "\n\n".join(f"Answer {labels[a]}:\n{answers[a][:1200]}" for a in sorted(answers))
    prompt = (f"Task: {task['prompt'].split(chr(10))[0]}\nSource text: {task['source']}\n\n{body}\n\n"
              "Grade each answer 1-5 for correctness against the source, completeness and obeying the instruction "
              "(5 = excellent, 1 = wrong). Reply with JSON like {\"A\": 4, \"B\": 3} only.")
    out, _, err = ask_gateway(base, key, group, prompt, max_tokens=60)
    grades = _json(out) or {}
    return {arm: grades.get(labels[arm]) for arm in answers}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--local", required=True, help="resident proxy base URL (native Ollama API)")
    ap.add_argument("--gateway", required=True, help="LiteLLM base URL for the cloud arms and the judge")
    ap.add_argument("--cloud-groups", default="medium-tier,simple-tier")
    ap.add_argument("--judge-group", default="complex-tier")
    ap.add_argument("--local-samples", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import yaml
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    key = str((cfg.get("general_settings") or {}).get("master_key", ""))
    if key.startswith("os.environ/"):
        name = key.split("/", 1)[1]
        key = os.environ.get(name, "")
        if not key:
            for line in Path(a.config).with_name(".env").read_text(encoding="utf-8").splitlines():
                k, _, v = line.partition("=")
                if k.strip() == name:
                    key = v.strip().strip("'\"")
    groups = [g for g in a.cloud_groups.split(",") if g]
    suite = tasks()
    rows = []

    def run(task):
        arms = {}
        for s in range(a.local_samples):
            arms[f"local#{s}"] = ask_local(a.local, task["prompt"])
        for g in groups:
            arms[f"cloud:{g}"] = ask_gateway(a.gateway, key, g, task["prompt"])
        res = {}
        for arm, (txt, dt, err) in arms.items():
            res[arm] = {"text": txt[:600], "s": round(dt, 2), "err": err,
                        "pass": (bool(task["check"](txt)) if task["check"] and txt else (False if task["check"] else None))}
        if task["check"] is None:
            ans = {arm: r["text"] for arm, r in res.items() if r["text"]}
            grades = judge(a.gateway, key, a.judge_group, task, ans) if len(ans) > 1 else {}
            for arm, g in grades.items():
                res[arm]["grade"] = g
        return {"id": task["id"], "cat": task["cat"], "arms": res}

    with ThreadPoolExecutor(2) as ex:
        rows = list(ex.map(run, suite))
    summary = {}
    for r in rows:
        for arm, v in r["arms"].items():
            s = summary.setdefault(arm.split("#")[0], {"n": 0, "pass": 0, "graded": [], "lat": [], "errors": 0, "by_cat": {}})
            s["lat"].append(v["s"])
            s["errors"] += 1 if v["err"] else 0
            if v["pass"] is not None:
                s["n"] += 1
                s["pass"] += 1 if v["pass"] else 0
                c = s["by_cat"].setdefault(r["cat"], [0, 0])
                c[1] += 1
                c[0] += 1 if v["pass"] else 0
            if isinstance(v.get("grade"), (int, float)):
                s["graded"].append(v["grade"])
    final = {arm: {"objective_pass": f"{s['pass']}/{s['n']}", "objective_rate": round(s["pass"] / s["n"], 3) if s["n"] else None,
                   "summary_mean_grade": round(statistics.mean(s["graded"]), 2) if s["graded"] else None,
                   "p50_s": round(statistics.median(s["lat"]), 2), "max_s": round(max(s["lat"]), 2), "errors": s["errors"],
                   "by_category": {c: f"{p}/{n}" for c, (p, n) in s["by_cat"].items()}} for arm, s in summary.items()}
    Path(a.out).write_text(json.dumps({"summary": final, "rows": rows}, indent=1), encoding="utf-8")
    print(json.dumps(final, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
