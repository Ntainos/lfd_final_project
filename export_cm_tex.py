# export_cm_tex.py
import json, sys, pathlib, os

inp = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("outputs/transformer_fast_distilroberta_cls_dev_report.json")
out = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path("outputs/cm_best.tex")

if not inp.exists():
    raise SystemExit(f"Input JSON not found: {inp}")

with open(inp, "r", encoding="utf-8") as f:
    rep = json.load(f)

cm = rep.get("confusion_matrix")
labels = rep.get("labels", ["NOT","OFF"])
if not cm:
    raise SystemExit("No confusion_matrix in the report JSON.")

os.makedirs(out.parent, exist_ok=True)
tex = []
tex += [r"\begin{table}[ht]\centering", r"\small", r"\begin{tabular}{lcc}", r"\toprule"]
tex += [f"& Pred {labels[0]} & Pred {labels[1]} \\\\", r"\midrule"]
tex += [f"Gold {labels[0]} & {cm[0][0]} & {cm[0][1]} \\\\"]
tex += [f"Gold {labels[1]} & {cm[1][0]} & {cm[1][1]} \\\\", r"\bottomrule", r"\end{tabular}"]
tex += [r"\caption{Confusion matrix on development for the selected model.}", r"\end{table}"]

out.write_text("\n".join(tex), encoding="utf-8")
print(f"[OK] wrote {out}")
