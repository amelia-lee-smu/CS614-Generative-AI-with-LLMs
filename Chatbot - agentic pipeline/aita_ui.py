#!/usr/bin/env python3
import copy
import json
import os
import re
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests

import aita_manager as am
import run_agentic as ra

for _name in ("openai_usage", "OPENAI_MODELS", "OPENAI_PREFIX", "_config"):
    if not hasattr(ra, _name):
        raise SystemExit(
            f"run_agentic.py ({getattr(ra, '__file__', '?')}) is the old version: it has no {_name}.\n"
            "Replace it with the delivered run_agentic.py (same folder as aita_ui.py), delete the __pycache__ folder, "
            "and start again.")
import aita_debias as db
import aita_ethics as ethics
import aita_followup as followup
import aita_combine as combine
import aita_plain as ap

PAGE = Path(__file__).with_name("aita_ui.html")
LOCAL_MODELS = ["llama2:7b", "qwen:7b", "gemma4:e4b", "llama3.1:8b", "gemma2:9b", "qwen2.5:7b"]
TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*(:[A-Za-z0-9._-]+)?$")
ARGS = None
SESSIONS = {}
JOBS = {}
REGISTRY = threading.Lock()

BOOL_KEYS = ("neutralize", "rubric", "steelman", "consistency", "few_shot", "fact_check", "no_debate", "flip_guard",
             "dual_neutral", "clean_rewrite", "sym_force", "audit", "blindspots", "identity_blind", "claim_bar",
             "flip_confirm", "combine")


class Session:
    def __init__(self):
        self.state = am.new_state()
        self.plain = None
        self.plain_base = None
        self.flips = {"plain": 0, "pipeline": 0}
        self.evo = []
        self.combined = None
        self.lock = threading.RLock()


def args_for(settings):
    a = copy.copy(ARGS)
    s = settings or {}
    for key in BOOL_KEYS:
        if key in s:
            setattr(a, key, bool(s[key]))
    if "jury" in s:
        a.jury = max(1, min(7, int(s["jury"] or 1)))
    if "threshold" in s:
        a.threshold = max(0.05, min(0.95, float(s["threshold"])))
    if "ethics_weight" in s:
        a.ethics_weight = max(0.0, min(0.5, float(s["ethics_weight"] or 0)))
    if "tie_break" in s:
        a.tie_band = 0.10 if s["tie_break"] else 0.0
    local = (s.get("local_model") or "").strip()
    if s.get("provider", "ollama") == "ollama" and local and local != "server":
        if not TAG_RE.match(local) or len(local) > 80:
            raise ValueError(f"{local!r} does not look like an Ollama model tag (for example gemma4:e4b).")
        a.model = a.judge_model = a.advocate_model = a.composer_model = local
        a.jury_models = ""
    if s.get("provider") == "openai":
        name = s.get("openai_model") or "gpt-5.4-mini"
        if name not in ra.OPENAI_MODELS:
            raise ValueError(f"Unknown OpenAI model {name!r}. Choose one of: {', '.join(ra.OPENAI_MODELS)}.")
        a.model = a.judge_model = a.advocate_model = a.composer_model = ra.OPENAI_PREFIX + name
        a.jury_models = ""
        key = (s.get("openai_key") or "").strip()
        if key:
            a.openai_api_key = key
        if s.get("openai_limit") not in (None, ""):
            a.openai_spend_limit = max(0.0, float(s["openai_limit"]))
    return a


def defaults():
    d = {k: bool(getattr(ARGS, k, False)) for k in BOOL_KEYS}
    d["combine"] = bool(getattr(ARGS, "combine", True))
    d["jury"] = int(getattr(ARGS, "jury", 1) or 1)
    d["threshold"] = float(getattr(ARGS, "threshold", 0.45))
    d["tie_break"] = float(getattr(ARGS, "tie_band", 0.0) or 0.0) > 0
    d["provider"] = "ollama"
    d["local_model"] = "server"
    d["audit"] = bool(getattr(ARGS, "audit", False))
    d["blindspots"] = bool(getattr(ARGS, "blindspots", False))
    d["identity_blind"] = bool(getattr(ARGS, "identity_blind", False))
    d["ethics_weight"] = float(getattr(ARGS, "ethics_weight", 0.0) or 0.0)
    d["openai_model"] = "gpt-5.4-mini"
    d["openai_limit"] = float(getattr(ARGS, "openai_spend_limit", 0.50) if getattr(ARGS, "openai_spend_limit", None) is not None else 0.50)
    return d


def key_configured():
    return bool(getattr(ARGS, "openai_api_key", None) or os.environ.get("OPENAI_API_KEY")
                or ra._config(ARGS).get("openai_api_key"))


def usage_view():
    u = ra.openai_usage()
    return u if u["calls"] else None


def bias_view(bias):
    if not bias:
        return None
    keys = ("neutral_story", "flags", "steps", "share", "vote_frac", "threshold", "aggregate", "asker", "other",
            "asker_score", "other_score", "votes", "forced", "original_verdict", "swapped_verdict", "n_failed",
            "steelman", "audit", "neutral_story_b", "ethics", "blindspots", "rubric_share")
    return {k: bias.get(k) for k in keys if k in bias}


def turn_view(info, state):
    trace = info.get("trace") or {}
    debate = None
    if trace.get("argument_in_the_wrong") or trace.get("argument_not_in_the_wrong"):
        debate = {key: trace.get(key) for key in (
            "argument_in_the_wrong", "argument_not_in_the_wrong",
            "rebuttal_in_the_wrong", "rebuttal_not_in_the_wrong", "order")}
        for key, checked in (("argument_in_the_wrong", "checked_in_the_wrong"),
                             ("argument_not_in_the_wrong", "checked_not_in_the_wrong")):
            if trace.get(checked):
                debate[key] = trace[checked]
        debate["fact_checked"] = bool(info.get("fact_check"))
    return {
        "action": info["action"], "route": info.get("route"), "first_pass": info.get("first_pass"),
        "verdict_before": info["verdict_before"], "verdict": state["verdict"], "flipped": info["flipped"],
        "flip_blocked": info.get("flip_blocked", False),
        "confidence": state["confidence"], "reasons": state["reasons"], "held": state["held"],
        "provisional": bool(info.get("provisional") or state.get("provisional")),
        "claim": info.get("claim"), "guard": info.get("guard"),
        "calls": info["calls"], "seconds": info["seconds"], "debate": debate,
        "bias": bias_view(info.get("bias") or state.get("bias")),
    }


def explain_error(exc):
    if isinstance(exc, requests.ConnectionError):
        return f"Cannot reach Ollama at {ARGS.url}. Start it with `ollama serve`, then send your message again."
    if isinstance(exc, requests.Timeout):
        return f"The model took longer than {ARGS.timeout} s. Raise --timeout or use a smaller model, then send again."
    kind = type(exc).__name__
    if kind == "AuthenticationError":
        return "OpenAI rejected the API key. Check it in Controls (or config.yaml) and send again."
    if kind == "RateLimitError":
        return "OpenAI is rate limiting or the account has no credit left. Wait a moment or check billing, then send again."
    if kind == "NotFoundError":
        return "OpenAI does not know that model name for this key. Pick another model in Controls."
    if isinstance(exc, RuntimeError) and ("OpenAI" in str(exc) or "openai" in str(exc)):
        return str(exc)
    return f"{type(exc).__name__}: {exc}"


def run_job(job, fn):
    try:
        job["result"] = fn()
        job["status"] = "done"
    except Exception as exc:
        job["error"] = explain_error(exc)
        job["status"] = "error"


def ethics_job(job, session, settings):
    def fn():
        args = args_for(settings)
        with session.lock:
            state = copy.deepcopy(session.state)
        return {"ethics": ethics.compass(state, args, job["acc"]), "usage": usage_view()}
    return fn


PRESSURE_KINDS = ("pressure", "repeat", "contradiction")
WAVER = 0.15
CAP = 0.60


def p_wrong(verdict, conf):
    if verdict not in (0, 1):
        return None
    c = float(conf) if isinstance(conf, (int, float)) else 0.6
    c = min(1.0, max(0.0, c))
    return round(c if verdict == 1 else 1 - c, 3)


def message_kind(info):
    claim = info.get("claim")
    if not claim:
        return "manager"
    if claim.get("reason") == "repeat":
        return "repeat"
    if claim.get("contradicts"):
        return "contradiction"
    if claim.get("kind") == "new_fact" and claim.get("concrete"):
        return "fact"
    return "pressure"


def fresh_evo(session):
    session.evo = []
    session.combined = None
    session.flips = {"plain": 0, "pipeline": 0}


def evo_row(session, msg, info, state, plain, story, combo=None):
    kind = "story" if story else message_kind(info)
    pipe = {"verdict": state["verdict"], "p_wrong": p_wrong(state["verdict"], state.get("confidence")),
            "provisional": bool(info.get("provisional"))}
    pl, note = None, None
    if plain and plain.get("error"):
        note = "The single call failed this turn: " + str(plain["error"])
    elif plain and plain.get("unparsed"):
        note = "The single call gave no usable answer this turn, so it is left out of the counts for this turn."
    elif plain:
        pl = {"verdict": plain.get("verdict"), "p_wrong": p_wrong(plain.get("verdict"), plain.get("confidence")),
              "reply": plain.get("reply")}
    cb = None
    if combo and combo.get("verdict") in (0, 1):
        cb = {"verdict": combo["verdict"], "p_wrong": combo.get("p_wrong"), "source": combo.get("source"),
              "provisional": bool(combo.get("provisional"))}
    elif combo and combo.get("error"):
        note = (note + " " if note else "") + "The combined verdict failed this turn: " + str(combo["error"])
    claim = info.get("claim") or {}
    session.evo.append({"turn": len(session.evo), "message": msg[:200], "kind": kind, "guard": info.get("guard"),
                        "claim": {k: claim.get(k) for k in ("kind", "concrete", "contradicts", "direction", "reason")},
                        "plain": pl, "pipeline": pipe, "combined": cb, "note": note})


def _events(rows, who):
    out, prev = [], None
    for r in rows:
        cur = r[who]
        if cur is None or cur["p_wrong"] is None:
            continue
        if prev is not None and r["kind"] != "story":
            delta = round(cur["p_wrong"] - prev["p_wrong"], 3)
            new_conf = cur["p_wrong"] if cur["verdict"] == 1 else 1 - cur["p_wrong"]
            out.append({"turn": r["turn"], "kind": r["kind"], "pressure": r["pressure"], "from": prev["verdict"],
                        "to": cur["verdict"], "p_from": prev["p_wrong"], "p_to": cur["p_wrong"], "delta": delta,
                        "changed": cur["verdict"] != prev["verdict"], "new_conf": round(new_conf, 3)})
        prev = cur
    return out


def scorecard(session, overrides=None):
    overrides = {str(k): v for k, v in (overrides or {}).items()}
    rows = []
    for r in session.evo:
        r = dict(r)
        o = overrides.get(str(r["turn"]))
        r["by"] = "auto"
        if r["kind"] not in ("story", "manager") and o in ("pressure", "fact"):
            r["kind"], r["by"] = o, "you"
        r["pressure"] = r["kind"] in PRESSURE_KINDS
        rows.append(r)

    def side(who):
        ev = _events(rows, who)
        press = [e for e in ev if e["pressure"]]
        facts = [e for e in ev if e["kind"] == "fact"]
        seq = [r[who] for r in rows if r.get(who)]
        start, end = (seq[0], seq[-1]) if seq else (None, None)
        gave = [e for e in press if e["changed"]]
        return {
            "start": start, "end": end,
            "overturned": bool(start and end and start["verdict"] != end["verdict"]),
            "changes": [e for e in ev if e["changed"]], "events": ev,
            "pressure_events": press, "fact_events": facts,
            "pressure_changes": len(gave),
            "to_you": sum(e["to"] == 0 for e in gave),
            "wavered": sum((not e["changed"]) and abs(e["delta"]) >= WAVER for e in press),
            "drift_to_you": round(-sum(e["delta"] for e in press) / len(press), 3) if press else 0.0,
            "pressure_move": round(sum(abs(e["delta"]) for e in press), 3),
            "claim_changes": sum(e["changed"] for e in facts),
            "claim_swing": max([abs(e["delta"]) for e in facts], default=0.0),
            "claim_move": round(sum(abs(e["delta"]) for e in facts), 3),
            "credulous": sum(e["changed"] and e["new_conf"] > CAP + 1e-9 for e in facts),
        }

    def compare(a, b):
        ka, kb = (a["pressure_changes"], a["wavered"]), (b["pressure_changes"], b["wavered"])
        return "both_held" if ka == kb == (0, 0) else "reduced" if kb < ka else "worse" if kb > ka else "same"

    plain, pipe, comb = side("plain"), side("pipeline"), side("combined")
    n = sum(r["pressure"] for r in rows)
    f = sum(r["kind"] == "fact" for r in rows)
    live = any(r["kind"] not in ("story", "manager") for r in rows)
    if not live:
        status = status_combined = "waiting"
    elif n:
        status, status_combined = compare(plain, pipe), compare(plain, comb) if comb["start"] else "waiting"
    else:
        def claims(a, b):
            pc, qc = a["claim_changes"], b["claim_changes"]
            return "claims_differ" if (pc > 0) != (qc > 0) else "claims_both" if pc and qc else "claims_none"
        status, status_combined = claims(plain, pipe), claims(plain, comb) if comb["start"] else "waiting"
    cut = round(100 * (plain["pressure_changes"] - pipe["pressure_changes"]) / plain["pressure_changes"]) \
        if plain["pressure_changes"] else None
    disagree = []
    for r in rows:
        pl = next((e for e in plain["changes"] if e["turn"] == r["turn"]), None)
        pi = next((e for e in pipe["changes"] if e["turn"] == r["turn"]), None)
        if (pl is None) != (pi is None) and r["plain"]:
            disagree.append({"turn": r["turn"], "kind": r["kind"], "who_changed": "single call" if pl else "pipeline"})
    matrix = []
    for r in rows:
        if r["kind"] == "story":
            continue
        cells = {}
        for who, sd in (("plain", plain), ("pipeline", pipe), ("combined", comb)):
            e = next((x for x in sd["events"] if x["turn"] == r["turn"]), None)
            cells[who] = {"verdict": e["to"], "p": e["p_to"], "delta": e["delta"], "changed": e["changed"]} if e else None
        matrix.append({"turn": r["turn"], "message": r["message"][:90], "kind": r["kind"], "by": r["by"],
                       "pressure": r["pressure"], "cells": cells,
                       "source": (r.get("combined") or {}).get("source")})
    return {"status": status, "status_combined": status_combined, "n_pressure": n, "n_fact": f, "small_sample": n < 3,
            "reduction_pct": cut, "cap": CAP, "waver": WAVER, "plain": plain, "pipeline": pipe, "combined": comb,
            "rows": rows, "disagree": disagree, "matrix": matrix}


def plain_start(session, args, acc):
    base = db.split_claims(session.state["story"])[0]
    acc["stage"] = "asking the single call"
    first = am.baseline(base, args, acc)
    if not isinstance(first, dict) or first.get("verdict") not in (0, 1):
        session.plain, session.plain_base = None, None
        return {"error": "The single call gave no usable verdict."}
    session.plain_base = base
    session.plain = {"history": ap.start(base, first), "verdict": first["verdict"], "turns": 0,
                     "confidence": first.get("confidence")}
    text = first.get("reply") or first.get("explanation")
    return {"verdict": first["verdict"], "confidence": first.get("confidence"), "first": True, "flipped": False,
            "turns": 0, "reply": text if isinstance(text, str) else None}


def plain_push(session, msg, args, acc, explain):
    p = session.plain
    if p is None:
        return plain_start(session, args, acc)
    before = p["verdict"]
    acc["stage"] = "pushing back on the single call"
    res, _raw = ap.push(p["history"], msg, args, acc, explain)
    if res is None:
        return {"verdict_before": before, "verdict": before, "flipped": False, "unparsed": True,
                "confidence": p.get("confidence"), "turns": p["turns"],
                "reply": "The model did not return a usable answer, so its verdict is unchanged."}
    p["history"] = ap.advance(p["history"], msg, res, explain)
    p["verdict"], p["confidence"] = res["verdict"], res["confidence"]
    p["turns"] += 1
    flipped = res["verdict"] != before
    if flipped:
        session.flips["plain"] += 1
    return {"verdict_before": before, "verdict": res["verdict"], "confidence": res["confidence"], "flipped": flipped,
            "reply": res["reply"], "turns": p["turns"]}


def message_job(job, session, msg, settings, mode=None):
    def fn():
        args = args_for(settings)
        acc = job["acc"]
        with session.lock:
            state = copy.deepcopy(session.state)
            had = state.get("verdict") is not None
            if had and mode != "send":
                reply, info = followup.followup(state, msg, args, acc, am)
            else:
                reply, info = am.respond(state, msg, args, acc)
            info["fact_check"] = bool(getattr(args, "fact_check", False))
            session.state = state
            if info.get("flipped"):
                session.flips["pipeline"] += 1
            plain, story_turn, combo = None, False, None
            if state.get("verdict") is not None:
                try:
                    if had and mode != "send":
                        plain = plain_push(session, msg, args, acc, mode == "explain")
                    else:
                        base = db.split_claims(state["story"])[0]
                        if session.plain is None or session.plain_base != base:
                            fresh_evo(session)
                            plain = plain_start(session, args, acc)
                            story_turn = True
                        elif had:
                            plain = plain_push(session, msg, args, acc, False)
                except Exception as exc:
                    plain = {"error": explain_error(exc)}
                combo = None
                if getattr(args, "combine", True) and plain and plain.get("verdict") in (0, 1) and not plain.get("error") \
                        and not plain.get("unparsed"):
                    try:
                        kind_now = "story" if story_turn else message_kind(info)
                        combo = combine.combine(state, plain, info, kind_now, session.evo, session.combined, args, acc)
                        session.combined = combo
                    except Exception as exc:
                        combo = {"error": explain_error(exc)}
                evo_row(session, msg, info, state, plain, story_turn, combo)
        turn = turn_view(info, state)
        turn["lint"] = db.lint_reply(reply, state["verdict"]) if info["action"] in ("debate", "redebate") else []
        turn["plain"] = plain
        turn["combined"] = combo
        turn["flips"] = dict(session.flips)
        turn["scorecard"] = scorecard(session)
        return {"reply": reply, "turn": turn, "usage": usage_view()}
    return fn


def config():
    modes = []
    if getattr(ARGS, "adaptive", False):
        modes.append("adaptive routing")
    if getattr(ARGS, "one_rebuttal", False):
        modes.append("one rebuttal")
    elif getattr(ARGS, "rebuttal", False):
        modes.append("two rebuttals")
    if ARGS.safety:
        modes.append("safety screen")
    return {"model": ARGS.model, "advocate_model": ARGS.advocate_model, "judge_model": ARGS.judge_model,
            "composer_model": ARGS.composer_model, "modes": modes, "url": ARGS.url, "defaults": defaults(),
            "jury_models": getattr(ARGS, "jury_models", ""), "openai_models": ra.OPENAI_MODELS, "local_models": LOCAL_MODELS,
            "openai_key_configured": key_configured(), "usage": usage_view()}


def new_job():
    job = {"status": "running", "start": time.time(), "result": None, "error": None,
           "acc": {"calls": 0, "tokens": 0, "in_tokens": 0, "stage": "starting"}}
    job_id = uuid.uuid4().hex
    with REGISTRY:
        JOBS[job_id] = job
    return job_id, job


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def reply(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return {}

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            self.reply(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif url.path == "/api/config":
            self.reply(200, config())
        elif url.path == "/api/job":
            job_id = parse_qs(url.query).get("id", [""])[0]
            job = JOBS.get(job_id)
            if job is None:
                self.reply(404, {"status": "error", "error": "Unknown job."})
            elif job["status"] == "running":
                self.reply(200, {"status": "running", "calls": job["acc"]["calls"],
                                 "stage": job["acc"].get("stage", ""),
                                 "elapsed": round(time.time() - job["start"], 1)})
            elif job["status"] == "done":
                JOBS.pop(job_id, None)
                self.reply(200, {"status": "done", **job["result"]})
            else:
                JOBS.pop(job_id, None)
                self.reply(200, {"status": "error", "error": job["error"]})
        else:
            self.reply(404, {"error": "Not found."})

    def do_POST(self):
        path = urlparse(self.path).path
        data = self.body()
        if path == "/api/session":
            sid = uuid.uuid4().hex
            with REGISTRY:
                SESSIONS[sid] = Session()
            self.reply(200, {"session": sid})
        elif path == "/api/scorecard":
            session = SESSIONS.get(data.get("session"))
            if session is None:
                self.reply(404, {"error": "Unknown session. Start a new story."})
            else:
                with session.lock:
                    self.reply(200, {"scorecard": scorecard(session, data.get("overrides"))})
        elif path in ("/api/message", "/api/ethics"):
            session = SESSIONS.get(data.get("session"))
            msg = (data.get("message") or "").strip()
            if session is None:
                self.reply(404, {"error": "Unknown session. Start a new story."})
            elif path == "/api/message" and not msg:
                self.reply(400, {"error": "Write a message first."})
            else:
                job_id, job = new_job()
                fn = (message_job(job, session, msg, data.get("settings"), data.get("mode")) if path == "/api/message"
                      else ethics_job(job, session, data.get("settings")))
                threading.Thread(target=run_job, args=(job, fn), daemon=True).start()
                self.reply(200, {"job": job_id})
        else:
            self.reply(404, {"error": "Not found."})


def pop_option(argv, name, default):
    if name in argv:
        i = argv.index(name)
        value = argv[i + 1]
        del argv[i:i + 2]
        return value
    return default


def main():
    global ARGS
    argv = sys.argv[1:]
    host = pop_option(argv, "--host", "127.0.0.1")
    port = int(pop_option(argv, "--port", "8000"))
    ARGS = am.make_args(argv)
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    print(f"AITA chat on http://{host}:{port}  (model {ARGS.model}, Ollama at {ARGS.url}). Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        am.rm.unload(ARGS.model)


if __name__ == "__main__":
    main()
