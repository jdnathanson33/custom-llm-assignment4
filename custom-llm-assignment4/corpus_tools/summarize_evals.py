"""Build results/eval_comparison.md: every one of the 48 cases across all four result sets."""
import json, glob
from pathlib import Path
RUNS = {"starter": glob.glob("experiments/starter/llm_runs/*/")[0], "expanded": glob.glob("experiments/expanded/llm_runs/*/")[0]}
res = {(e, s): json.load(open(f"{d}language_evals/{s}/eval_results.json")) for e, d in RUNS.items() for s in ("untrained", "final")}
summ = {(e, s): json.load(open(f"{d}language_evals/{s}/eval_summary.json")) for e, d in RUNS.items() for s in ("untrained", "final")}
def cell(c):
    if c["status"] != "scored":
        return "OOV: " + ", ".join(c["unknown_prompt_words"] + [w for w in c["unknown_choices"] if w not in c["unknown_prompt_words"]])
    p = c["choice_probabilities"][c["predicted_choice"]]
    return f"{'✅' if c['score'] else '❌'} {c['predicted_choice']} ({p:.3f})"
out = ["# All 48 cases × four result sets", "",
       "✅/❌ = four-choice score (chosen word and its probability). OOV = unscorable (unknown words listed; counts as 0).",
       "Free continuations are the unconstrained text generated after the prompt (temperature 0.8, fixed seed), and they are not scored.", "",
       "| id | group / category | prompt → expected | starter untrained | starter trained | expanded untrained | expanded trained | starter trained continuation | expanded trained continuation |",
       "|---|---|---|---|---|---|---|---|---|"]
for i, c in enumerate(res[("starter", "untrained")]):
    row = [c["id"], f"{c['group']} / {c['category']}", f"`{c['prompt']}` → **{c['expected']}**"]
    row += [cell(res[k][i]) for k in [("starter", "untrained"), ("starter", "final"), ("expanded", "untrained"), ("expanded", "final")]]
    row += [f"`{res[('starter','final')][i]['generated_text'] or '[empty]'}`", f"`{res[('expanded','final')][i]['generated_text'] or '[empty]'}`"]
    out.append("| " + " | ".join(x.replace("|", "\\|") for x in row) + " |")
out += ["", "## Category breakdown (correct / scorable / total)", "",
        "| category | starter untrained | starter trained | expanded untrained | expanded trained |", "|---|---|---|---|---|"]
for cat in summ[("starter", "final")]["by_category"]:
    out.append(f"| {cat} | " + " | ".join(f"{summ[k]['by_category'][cat]['correct']} / {summ[k]['by_category'][cat]['scorable']} / {summ[k]['by_category'][cat]['total']}" for k in summ) + " |")
Path("results/eval_comparison.md").write_text("\n".join(out) + "\n", encoding="utf-8")
for k, v in summ.items():
    print(k, v["overall"], v["model_sha256"][:12])
