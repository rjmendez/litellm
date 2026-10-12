"""Evaluate specialist models on the task they were built for, in the prompt format they expect.

Each model's own GGUF chat template is rendered with jinja2 and sent raw, so no Ollama template can distort it.

  vulnllm   VulnLLM-R: vulnerable vs patched snippets, card prompt + a one-line answer marker
  foundation Foundation-Sec reasoning: ATT&CK / CWE identification and log triage, reasoning left on
  antares   Antares-1B: a real tool loop over small sandboxed toy repos, scored on whether it submits the right file
  rtl       Verilog tasks checked by simulation (iverilog), any model, via /api/chat

    python specialist_fit.py --ollama http://127.0.0.1:11440 --section vulnllm --model vulnllm-r-7b:Q4_K_M --gguf PATH --out r.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import statistics
import struct
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path


# ---- GGUF metadata (chat template + special tokens) -------------------------------------------------------
def gguf_meta(path: str) -> dict:
    want = {"tokenizer.chat_template", "tokenizer.ggml.tokens", "tokenizer.ggml.eos_token_id", "tokenizer.ggml.bos_token_id"}
    f = open(path, "rb")
    assert f.read(4) == b"GGUF"
    f.read(4)
    _, nkv = struct.unpack("<QQ", f.read(16))

    def s():
        n, = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf-8", "replace")

    def val(t):
        if t == 8:
            return s()
        if t == 9:
            at, n = struct.unpack("<IQ", f.read(12))
            return [val(at) for _ in range(n)]
        fmt = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}[t]
        return struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))[0]

    out = {}
    for _ in range(nkv):
        k = s()
        t, = struct.unpack("<I", f.read(4))
        v = val(t)
        if k in want:
            out[k] = v
    toks = out.get("tokenizer.ggml.tokens", [])
    pick = lambda k: toks[out[k]] if k in out and out[k] < len(toks) else ""
    return {"template": out.get("tokenizer.chat_template", ""), "bos": pick("tokenizer.ggml.bos_token_id"),
            "eos": pick("tokenizer.ggml.eos_token_id")}


def render(meta: dict, messages: list, tools: list | None = None, add_generation_prompt: bool = True) -> str:
    import jinja2
    env = jinja2.Environment(loader=jinja2.BaseLoader(), trim_blocks=False, lstrip_blocks=False, extensions=["jinja2.ext.loopcontrols"])

    def raise_exception(msg):
        raise jinja2.TemplateError(msg)

    env.globals["raise_exception"] = raise_exception
    # jinja's tojson returns Markup, and Markup + str HTML-escapes the str: use a plain-string version
    env.filters["tojson"] = lambda v, indent=None, **kw: json.dumps(v, ensure_ascii=False, indent=indent)
    env.globals["strftime_now"] = lambda fmt: time.strftime(fmt)
    return env.from_string(meta["template"]).render(messages=messages, tools=tools, available_tools=tools, documents=None,
                                                    add_generation_prompt=add_generation_prompt, bos_token=meta["bos"],
                                                    eos_token=meta["eos"], enable_thinking=True)


def generate_raw(base: str, model: str, prompt: str, num_predict: int = 1500, temperature: float = 0.1, stop=None, timeout=900) -> tuple:
    body = {"model": model, "prompt": prompt, "raw": True, "stream": False,
            "options": {"num_predict": num_predict, "temperature": temperature, **({"stop": stop} if stop else {})}}
    t = time.time()
    try:
        req = urllib.request.Request(f"{base}/api/generate", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
        return d.get("response", ""), time.time() - t
    except Exception as exc:
        return f"__ERROR__ {exc!r}"[:200], time.time() - t


def chat(base: str, model: str, prompt: str, num_predict: int = 900, timeout=900) -> tuple:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False, "think": False,
            "options": {"num_predict": num_predict, "temperature": 0.1}}
    t = time.time()
    try:
        req = urllib.request.Request(f"{base}/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
        return (d.get("message") or {}).get("content", "") or "", time.time() - t
    except Exception as exc:
        return f"__ERROR__ {exc!r}"[:200], time.time() - t


def after_think(text: str) -> str:
    return text.split("</think>")[-1] if "</think>" in text else text


# ---- VulnLLM-R: vulnerable vs patched pairs ---------------------------------------------------------------
PAIRS = [
    ("CWE-120", "void copy(const char *s) {\n  char buf[16];\n  strcpy(buf, s);\n  puts(buf);\n}\n",
     "void copy(const char *s) {\n  char buf[16];\n  strncpy(buf, s, sizeof(buf) - 1);\n  buf[sizeof(buf) - 1] = 0;\n  puts(buf);\n}\n"),
    ("CWE-190", "int *make(size_t n) {\n  return malloc(n * sizeof(int));\n}\n",
     "int *make(size_t n) {\n  if (n > SIZE_MAX / sizeof(int)) return NULL;\n  return malloc(n * sizeof(int));\n}\n"),
    ("CWE-416", "void f(void) {\n  char *p = malloc(8);\n  strcpy(p, \"hi\");\n  free(p);\n  printf(\"%s\\n\", p);\n}\n",
     "void f(void) {\n  char *p = malloc(8);\n  strcpy(p, \"hi\");\n  printf(\"%s\\n\", p);\n  free(p);\n  p = NULL;\n}\n"),
    ("CWE-78", "import os\ndef ping(host):\n    os.system(\"ping -c1 \" + host)\n",
     "import subprocess, re\ndef ping(host):\n    if not re.fullmatch(r\"[A-Za-z0-9.-]{1,253}\", host):\n        raise ValueError(host)\n    subprocess.run([\"ping\", \"-c1\", host], check=True)\n"),
    ("CWE-89", "def get(db, name):\n    return db.execute(f\"SELECT * FROM users WHERE name = '{name}'\").fetchall()\n",
     "def get(db, name):\n    return db.execute(\"SELECT * FROM users WHERE name = ?\", (name,)).fetchall()\n"),
    ("CWE-22", "import os\ndef read(base, rel):\n    return open(os.path.join(base, rel)).read()\n",
     "import os\ndef read(base, rel):\n    p = os.path.realpath(os.path.join(base, rel))\n    if not p.startswith(os.path.realpath(base) + os.sep):\n        raise PermissionError(rel)\n    return open(p).read()\n"),
    ("CWE-134", "void log_msg(const char *user) {\n  printf(user);\n}\n", "void log_msg(const char *user) {\n  printf(\"%s\", user);\n}\n"),
    ("CWE-502", "import pickle\ndef load(blob):\n    return pickle.loads(blob)\n", "import json\ndef load(blob):\n    return json.loads(blob)\n"),
]


def vulnllm(base, model, gguf, num_predict):
    meta = gguf_meta(gguf)
    items = []
    for cwe, bad, good in PAIRS:
        items.append((cwe, True, bad))
        items.append((cwe, False, good))
    tp = fp = tn = fn = 0
    rows = []
    for cwe, vuln, code in items:
        user = ("You are an advanced vulnerability detection model. Analyze the following code step by step to determine whether it "
                f"contains a security vulnerability.\n\nCode:\n```\n{code}```\n\nPlease provide your reasoning followed by the final answer. "
                "End with exactly one line: FINAL ANSWER: VULNERABLE or FINAL ANSWER: NOT VULNERABLE.")
        prompt = render(meta, [{"role": "user", "content": user}])
        text, dt = generate_raw(base, model, prompt, num_predict=num_predict, stop=[meta["eos"]] if meta["eos"] else None)
        tail = after_think(text)
        m = re.findall(r"final answer\W{0,12}(NOT VULNERABLE|VULNERABLE)", tail, re.I)
        pred = None if not m else (m[-1].upper() == "VULNERABLE")
        rows.append({"cwe": cwe, "truth": vuln, "pred": pred, "s": round(dt, 1), "tail": tail[-120:]})
        if pred is None:
            fn += 1 if vuln else 0
            fp += 0 if vuln else 0
        elif vuln and pred:
            tp += 1
        elif vuln and not pred:
            fn += 1
        elif not vuln and pred:
            fp += 1
        else:
            tn += 1
    acc = (tp + tn) / len(items)
    return {"accuracy": round(acc, 3), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "unparsed": sum(1 for r in rows if r["pred"] is None), "p50_s": round(statistics.median(r["s"] for r in rows), 1), "rows": rows}


# ---- Foundation-Sec: identifiers and triage ---------------------------------------------------------------
FOUND = [
    ("Which MITRE ATT&CK technique ID covers dumping credentials from LSASS memory? Give the ID.", r"T1003\.001"),
    ("Which MITRE ATT&CK sub-technique ID is spearphishing with a malicious attachment?", r"T1566\.001"),
    ("Which MITRE ATT&CK sub-technique ID is execution through PowerShell?", r"T1059\.001"),
    ("Which MITRE ATT&CK sub-technique ID is persistence via a scheduled task on Windows?", r"T1053\.005"),
    ("Which MITRE ATT&CK technique ID is process injection?", r"T1055"),
    ("Which CWE identifier is SQL injection?", r"CWE-89\b"),
    ("Which CWE identifier is use after free?", r"CWE-416\b"),
    ("Which CWE identifier is path traversal (improper limitation of a pathname to a restricted directory)?", r"CWE-22\b"),
    ("Which CWE identifier is out-of-bounds write?", r"CWE-787\b"),
    ("Which CWE identifier is cross-site scripting (improper neutralization of input during web page generation)?", r"CWE-79\b"),
    ("Auth log: 500 failed SSH logins for root from a single IP over 2 minutes, then one success from the same IP. What technique is this and which ATT&CK ID?", r"T1110"),
    ("DNS logs show one host sending thousands of long, high-entropy TXT queries to a single rarely seen domain. Which ATT&CK technique ID best fits?", r"T1071\.004|T1048|T1572"),
]


def foundation(base, model, gguf, num_predict):
    meta = gguf_meta(gguf)
    hits, rows = 0, []
    for q, pat in FOUND:
        prompt = render(meta, [{"role": "user", "content": q}])
        text, dt = generate_raw(base, model, prompt, num_predict=num_predict, temperature=0.1, stop=[meta["eos"]] if meta["eos"] else None)
        tail = after_think(text)
        ok = bool(re.search(pat, tail))
        hits += ok
        rows.append({"q": q[:70], "ok": ok, "s": round(dt, 1), "had_think": "</think>" in text, "tail": tail[-110:]})
    return {"accuracy": round(hits / len(FOUND), 3), "p50_s": round(statistics.median(r["s"] for r in rows), 1), "rows": rows}


# ---- Antares: sandboxed tool loop -------------------------------------------------------------------------
REPOS = {
    "CWE-78": ("CWE-78: Improper Neutralization of Special Elements used in an OS Command ('OS Command Injection')", "tools/net.py",
               {"app/main.py": "from tools.net import check\n\ndef run(host):\n    return check(host)\n",
                "tools/net.py": "import os\n\ndef check(host):\n    return os.system('ping -c1 ' + host)\n",
                "app/util.py": "def clamp(x, lo, hi):\n    return max(lo, min(hi, x))\n", "README.md": "demo service\n"}),
    "CWE-89": ("CWE-89: Improper Neutralization of Special Elements used in an SQL Command ('SQL Injection')", "store/users.py",
               {"store/users.py": "def find(db, name):\n    return db.execute(f\"SELECT * FROM users WHERE name = '{name}'\").fetchall()\n",
                "store/orders.py": "def by_id(db, i):\n    return db.execute('SELECT * FROM orders WHERE id = ?', (i,)).fetchall()\n",
                "web/views.py": "from store import users\n\ndef show(db, n):\n    return users.find(db, n)\n", "LICENSE": "MIT\n"}),
    "CWE-798": ("CWE-798: Use of Hard-coded Credentials", "service/settings.py",
                {"service/settings.py": "DB_HOST = 'db.internal'\nDB_USER = 'admin'\nDB_PASSWORD = 'Winter2024!'\n",
                 "service/db.py": "from service import settings\n\ndef dsn():\n    return f'{settings.DB_USER}@{settings.DB_HOST}'\n",
                 "service/log.py": "import logging\nlog = logging.getLogger('svc')\n", "docs/notes.txt": "rotate keys quarterly\n"}),
    "CWE-22": ("CWE-22: Improper Limitation of a Pathname to a Restricted Directory ('Path Traversal')", "files/download.py",
               {"files/download.py": "import os\nBASE = '/srv/files'\n\ndef get(name):\n    return open(os.path.join(BASE, name), 'rb').read()\n",
                "files/list.py": "import os\n\ndef names():\n    return sorted(os.listdir('/srv/files'))\n",
                "api/routes.py": "from files import download\n\ndef serve(n):\n    return download.get(n)\n", "Makefile": "all:\n\techo ok\n"}),
}
TOOLS = [
    {"type": "function", "function": {"name": "terminal", "description": "Run a shell command in the repository (ls, cat, grep, find, head).",
                                      "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "submit_vulnerable_files", "description": "Submit the files that contain the vulnerability.",
                                      "parameters": {"type": "object", "properties": {"files": {"type": "array", "items": {"type": "string"}}}, "required": ["files"]}}},
    {"type": "function", "function": {"name": "submit_no_vulnerability_found", "description": "Submit that no vulnerability was found.",
                                      "parameters": {"type": "object", "properties": {}}}},
]


def sandbox_exec(root: Path, command: str) -> str:
    """A tiny read-only shell: ls, cat, head, grep, find, wc, with simple pipes, confined to the repo."""
    def safe(p: str) -> Path:
        q = (root / p.lstrip("/")).resolve()
        if root.resolve() not in q.parents and q != root.resolve():
            raise ValueError("outside repository")
        return q

    def run_one(parts: list, stdin: str | None) -> str:
        cmd, args = parts[0], [a for a in parts[1:]]
        if cmd == "ls":
            paths = [a for a in args if not a.startswith("-")] or ["."]
            out = []
            for p in paths:
                q = safe(p)
                out += sorted(x.name + ("/" if x.is_dir() else "") for x in q.iterdir()) if q.is_dir() else [q.name]
            return "\n".join(out)
        if cmd in ("cat", "head", "tail"):
            files = [a for a in args if not a.startswith("-") and not a.isdigit()]
            text = "".join(safe(f).read_text(encoding="utf-8") for f in files) if files else (stdin or "")
            n = 10
            if "-n" in args:
                n = int(args[args.index("-n") + 1])
            lines = text.splitlines()
            return "\n".join(lines if cmd == "cat" else (lines[:n] if cmd == "head" else lines[-n:]))
        if cmd == "grep":
            flags = [a for a in args if a.startswith("-")]
            rest = [a for a in args if not a.startswith("-")]
            pat = rest[0]
            targets = rest[1:] or ["."]
            rx = re.compile(pat, re.I if any("i" in f for f in flags) else 0)
            lines = []
            if stdin is not None and len(rest) == 1:
                return "\n".join(l for l in stdin.splitlines() if rx.search(l))
            for t in targets:
                q = safe(t)
                files = [q] if q.is_file() else sorted(x for x in q.rglob("*") if x.is_file())
                for fp in files:
                    for i, l in enumerate(fp.read_text(encoding="utf-8").splitlines(), 1):
                        if rx.search(l):
                            lines.append(f"{fp.relative_to(root).as_posix()}:{i}:{l}" if any("n" in f for f in flags) or any("r" in f for f in flags) else f"{fp.relative_to(root).as_posix()}:{l}")
            return "\n".join(lines)
        if cmd == "find":
            start = next((a for a in args if not a.startswith("-")), ".")
            return "\n".join(sorted(x.relative_to(root).as_posix() for x in safe(start).rglob("*") if x.is_file()))
        if cmd == "wc":
            return str(len((stdin or "").splitlines()))
        raise ValueError(f"command not available: {cmd}")

    try:
        out = None
        for seg in command.split("|"):
            out = run_one(shlex.split(seg.strip()), out)
        return out or "(no output)"
    except Exception as exc:
        return f"error: {exc}"


def antares(base, model, gguf, max_turns=15):
    meta = gguf_meta(gguf)
    results = []
    for cwe, (desc, target, files) in REPOS.items():
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for rel, content in files.items():
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                (root / rel).write_text(content, encoding="utf-8")
            msgs = [{"role": "system", "content": "You are a security analyst agent with terminal access to a mounted repository. Use shell commands to find "
                                                  "the files that contain the vulnerability, then call submit_vulnerable_files, or "
                                                  "submit_no_vulnerability_found if there is none."},
                    {"role": "user", "content": f"Vulnerability to locate:\n{desc}"}]
            submitted, turns, t0, err = None, 0, time.time(), None
            for turns in range(1, max_turns + 1):
                prompt = render(meta, msgs, tools=TOOLS)
                text, _ = generate_raw(base, model, prompt, num_predict=600, temperature=0.3, stop=[meta["eos"]] if meta["eos"] else None, timeout=600)
                if text.startswith("__ERROR__"):
                    err = text
                    break
                visible = after_think(text)
                calls = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", visible, re.S)
                if not calls:
                    msgs.append({"role": "assistant", "content": visible.strip()})
                    msgs.append({"role": "user", "content": "Continue: call a tool."})
                    continue
                try:
                    call = json.loads(calls[0])
                except ValueError:
                    msgs.append({"role": "assistant", "content": visible.strip()})
                    continue
                name, args = call.get("name"), call.get("arguments") or {}
                msgs.append({"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]})
                if name == "submit_vulnerable_files":
                    submitted = [str(x) for x in (args.get("files") or [])]
                    break
                if name == "submit_no_vulnerability_found":
                    submitted = []
                    break
                out = sandbox_exec(root, str(args.get("command", ""))) if name == "terminal" else f"unknown tool {name}"
                msgs.append({"role": "tool", "content": out[:3000]})
            hit = submitted is not None and any(target == s.lstrip("./") or s.endswith(target) for s in submitted)
            results.append({"cwe": cwe, "target": target, "submitted": submitted, "hit": hit, "turns": turns, "s": round(time.time() - t0, 1), "error": err})
    return {"hit_rate": round(sum(r["hit"] for r in results) / len(results), 3), "rows": results}


# ---- RTL tasks, checked by simulation ---------------------------------------------------------------------
RTL = [
    ("updown", "Write a Verilog module `top` with inputs clk, rst (synchronous, active high), up, and a 4-bit output reg count. "
               "On each rising clock edge: reset to 0, else if up count increments, else count decrements (wrapping).",
     "module tb; reg clk=0, rst=1, up=1; wire [3:0] count; top d(clk,rst,up,count); always #5 clk=~clk;\n"
     "integer i; reg ok=1;\ninitial begin #12 rst=0; repeat(3) @(posedge clk); #1 if (count!==4'd3) ok=0;\n"
     "up=0; repeat(2) @(posedge clk); #1 if (count!==4'd1) ok=0; up=0; repeat(2) @(posedge clk); #1 if (count!==4'd15) ok=0;\n"
     "$display(ok?\"PASS\":\"FAIL\"); $finish; end endmodule\n"),
    ("shift", "Write a Verilog module `top` with inputs clk, rst (synchronous, active high), din (1 bit) and an 8-bit output reg q that shifts left "
              "each clock inserting din at bit 0, and clears on rst.",
     "module tb; reg clk=0, rst=1, din=0; wire [7:0] q; top d(clk,rst,din,q); always #5 clk=~clk; reg ok=1;\n"
     "initial begin #12 rst=0; din=1; @(posedge clk); #1; din=0; @(posedge clk); #1; din=1; @(posedge clk); #1;\n"
     "if (q[2:0]!==3'b101) ok=0; $display(ok?\"PASS\":\"FAIL\"); $finish; end endmodule\n"),
    ("edge", "Write a Verilog module `top` with inputs clk, rst (synchronous, active high), sig, and output reg pulse that is high for exactly one clock "
             "cycle after sig goes from 0 to 1 (a rising-edge detector registered on clk).",
     "module tb; reg clk=0, rst=1, sig=0; wire pulse; top d(clk,rst,sig,pulse); always #5 clk=~clk; integer n=0; reg ok=1;\n"
     "always @(posedge clk) if (pulse) n=n+1;\ninitial begin #12 rst=0; repeat(2) @(posedge clk); sig=1; repeat(5) @(posedge clk); sig=0; repeat(2) @(posedge clk);\n"
     "sig=1; repeat(3) @(posedge clk); #1 if (n!==2) ok=0; $display(ok?\"PASS\":\"FAIL\"); $finish; end endmodule\n"),
    ("seq101", "Write a Verilog module `top` with inputs clk, rst (synchronous), din, and output reg found. found must be high for one cycle when the "
               "last three bits received on din (sampled each clock) were 1,0,1 (overlapping allowed).",
     "module tb; reg clk=0, rst=1, din=0; wire found; top d(clk,rst,din,found); always #5 clk=~clk; integer n=0; reg ok=1;\n"
     "always @(posedge clk) if (found) n=n+1; reg [7:0] pat = 8'b10101100; integer i;\n"
     "initial begin #12 rst=0; for (i=7;i>=0;i=i-1) begin din=pat[i]; @(posedge clk); end repeat(2) @(posedge clk);\n"
     "if (n<2 || n>3) ok=0; $display(ok?\"PASS\":\"FAIL\"); $finish; end endmodule\n"),
    ("mux", "Write a Verilog module `top` with 4 inputs of 8 bits a, b, c, d, a 2-bit input sel, and an 8-bit output y (combinational) selecting a,b,c,d for sel 0..3.",
     "module tb; reg [7:0] a=1,b=2,c=3,d=4; reg [1:0] sel=0; wire [7:0] y; top u(a,b,c,d,sel,y); reg ok=1;\n"
     "initial begin #1 if (y!==1) ok=0; sel=1; #1 if (y!==2) ok=0; sel=2; #1 if (y!==3) ok=0; sel=3; #1 if (y!==4) ok=0;\n"
     "$display(ok?\"PASS\":\"FAIL\"); $finish; end endmodule\n"),
]
IVERILOG = shutil.which("iverilog") or r"C:\ProgramData\chocolatey\bin\iverilog.exe"
VVP = shutil.which("vvp") or r"C:\ProgramData\chocolatey\bin\vvp.exe"


def rtl(base, model, num_predict=900):
    rows = []
    for name, prompt, tb in RTL:
        text, dt = chat(base, model, prompt + "\nReply with only the Verilog code in a ```verilog block.", num_predict=num_predict)
        text = after_think(text)
        m = re.search(r"```(?:verilog|systemverilog|v)?\n(.*?)```", text, re.S)
        code = m.group(1) if m else text
        ok, why = False, ""
        with tempfile.TemporaryDirectory() as td:
            Path(td, "d.v").write_text(code, encoding="utf-8")
            Path(td, "tb.v").write_text(tb, encoding="utf-8")
            try:
                c = subprocess.run([IVERILOG, "-g2012", "-o", "sim", "d.v", "tb.v"], cwd=td, capture_output=True, text=True, timeout=60)
                if c.returncode != 0:
                    why = "compile: " + (c.stderr or c.stdout)[:120]
                else:
                    r = subprocess.run([VVP, "sim"], cwd=td, capture_output=True, text=True, timeout=60)
                    ok = "PASS" in r.stdout
                    why = "" if ok else "sim: " + r.stdout.strip()[-60:]
            except Exception as exc:
                why = repr(exc)[:100]
        rows.append({"task": name, "ok": ok, "s": round(dt, 1), "why": why})
    return {"pass": f"{sum(r['ok'] for r in rows)}/{len(rows)}", "rate": round(sum(r["ok"] for r in rows) / len(rows), 3), "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ollama", default="http://127.0.0.1:11440")
    ap.add_argument("--section", required=True, choices=["vulnllm", "foundation", "antares", "rtl"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--gguf", default="")
    ap.add_argument("--num-predict", type=int, default=1500)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.section == "vulnllm":
        res = vulnllm(a.ollama, a.model, a.gguf, a.num_predict)
    elif a.section == "foundation":
        res = foundation(a.ollama, a.model, a.gguf, a.num_predict)
    elif a.section == "antares":
        res = antares(a.ollama, a.model, a.gguf)
    else:
        res = rtl(a.ollama, a.model)
    out = Path(a.out)
    data = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    data[f"{a.section}:{a.model}"] = res
    out.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(a.section, a.model, {k: v for k, v in res.items() if k != "rows"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
