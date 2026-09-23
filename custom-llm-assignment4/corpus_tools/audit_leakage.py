"""Extra leakage audit (beyond the notebook's exact-prefix check).
For every eval case, report the longest run of consecutive tokens shared between the eval
prompt+answer and any corpus passage, and flag any passage that contains the full prompt
or prompt followed by the answer."""
import json, re, sys
from pathlib import Path
tok = lambda t: re.findall(r"\w+(?:['’]\w+)*|[^\w\s]", t.lower())
suite = json.load(open("evals/language_evals.json"))["cases"]
passages = []
for f in sorted(Path("corpus").rglob("*.txt")):
    passages += [tok(l) for l in f.read_text().splitlines() if l.strip()]
def longest(a, b):
    best = 0; prev = [0]*(len(b)+1)
    for i in range(len(a)):
        cur = [0]*(len(b)+1)
        for j in range(len(b)):
            if a[i] == b[j]:
                cur[j+1] = prev[j]+1; best = max(best, cur[j+1])
        prev = cur
    return best
rows = []
joined = [" ".join(p) for p in passages]
for c in suite:
    target = tok(c["prompt"]) + [c["answer"]]
    lcs = max(longest(target, p) for p in passages) if passages else 0
    full_prompt = any(" ".join(tok(c["prompt"])) in j for j in joined)
    rows.append({"id": c["id"], "category": c["category"], "prompt_tokens": len(target)-1,
                 "longest_shared_run_with_prompt_plus_answer": lcs,
                 "full_prompt_found": full_prompt})
out = {"corpus_passages_checked": len(passages), "any_full_prompt_found": any(r["full_prompt_found"] for r in rows), "cases": rows}
Path(sys.argv[1]).write_text(json.dumps(out, indent=2))
for r in rows:
    if r["category"] in ("opposites","spatial_relations") or r["longest_shared_run_with_prompt_plus_answer"]>=5:
        print(r)
print("any full prompt found:", out["any_full_prompt_found"])
