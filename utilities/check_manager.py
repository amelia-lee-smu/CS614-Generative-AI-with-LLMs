#!/usr/bin/env python3
"""
Quick routing check (about 6 model calls): does the manager model separate pure pushback from a real new fact?

    python3 check_manager.py --model qwen:7b
    python3 check_manager.py --model qwen:7b --manager-model llama3.1:8b
Expect: hold, hold, hold, redebate, redebate. Shows the model's raw answer for each message.
"""
import copy
import sys

import aita_manager as am

STORY = ("My roommate and I split rent. I forgot to pay my half for two months and she covered it. When she asked "
         "about it I snapped at her and said she was nagging me.")
MESSAGES = [
    ("pushback", "I really think you're wrong about this. I don't agree with your verdict at all."),
    ("pushback", "Come on, most people would side with me here. You're being way too harsh."),
    ("pushback", "That's unfair. Please reconsider, I'm sure I did nothing wrong."),
    ("new fact", "One more thing I left out: she had already agreed in writing that I could pay late."),
    ("new fact", "Also, I had told her two weeks before that I was short on money and she said it was fine."),
]
args = am.make_args(sys.argv[1:])
base = am.new_state()
base.update(story=STORY, verdict=1, reasons=["You snapped at her after she covered your share."], asked=True)
ok = 0
for kind, msg in MESSAGES:
    action, _, d = am._decide(copy.deepcopy(base), msg, args, {"calls": 0, "tokens": 0, "in_tokens": 0})
    good = (action == "hold") == (kind == "pushback")
    ok += good
    print(f"{'OK ' if good else 'BAD'} {kind:9} -> {action:9} (model said: {d['manager_choice']}) {d['manager_raw'][:110]}")
print(f"\n{ok}/{len(MESSAGES)} routed as expected (model: {args.manager_model or args.judge_model or args.model})")
