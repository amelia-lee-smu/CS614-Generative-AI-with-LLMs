#!/usr/bin/env python3
"""
Interactive demo of the debate chatbot: safety check -> parallel advocates -> shuffled judge -> composer.
Prints ONE kind, direct reply (no two opposing answers). Local Ollama models only; nothing is stored.

    python3 aita_chat.py --model llama3.1:8b
    python3 aita_chat.py --model llama3.1:8b --advocate-model qwen2.5:3b --rebuttal --debate-critic-rounds 1
    add --show-trace to also print the hidden debate and the judge's verdict.

Paste your story, then finish with an empty line. Ctrl-D or an empty story quits.
"""
import json
import sys

import run_agentic as ra

argv = sys.argv[1:]
show = "--show-trace" in argv
argv = [a for a in argv if a != "--show-trace"]
args = ra.build_parser().parse_args(["--pipeline", "debate", "--safety", "--compose"] + argv)
args.schema = ra.rm.make_schema(False)
ds = json.loads(json.dumps(args.schema))
ds.setdefault("properties", {}).update({"confidence": {"type": "number"},
                                        "key_reasons": {"type": "array", "items": {"type": "string"}}})
ds["required"] = list(dict.fromkeys(list(ds.get("required", [])) + ["confidence", "key_reasons"]))
args.debate_schema = ds

print("Tell me your story (finish with an empty line):")
while True:
    lines = []
    try:
        while True:
            line = input()
            if not line.strip():
                break
            lines.append(line)
    except EOFError:
        break
    story = "\n".join(lines).strip()
    if not story:
        break
    rec = ra.run_task(args.model, "chat", "prompt", story, args)
    if rec["error"]:
        print(f"\n[error] {rec['error']}\n")
    else:
        print(f"\n{rec['final_reply']}\n")
        if show:
            print(f"--- verdict (1 = in the wrong): {rec['asker_is_asshole']}  confidence: {rec['judge_confidence']}  "
                  f"calls: {rec['n_calls']}  {rec['seconds']}s")
            print(json.dumps(json.loads(rec["agent_trace"]), indent=2, ensure_ascii=False), "\n")
    print("Another story? (empty line to quit)")
