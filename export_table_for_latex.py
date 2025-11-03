# export_table_for_latex.py
import json, glob, pandas as pd, pathlib, os

os.makedirs("outputs", exist_ok=True)

rows = []
for path in sorted(glob.glob("outputs/*_dev_report.json")):
    p = pathlib.Path(path)
    try:
        with open(p, "r", encoding="utf-8") as f:
            rep = json.load(f)
        per = rep.get("per_class", {})
        rows.append({
            "Model": p.stem,  # filename without .json
            "Macro F1": rep.get("macro_f1", float("nan")),
            "F1 NOT": per.get("NOT", {}).get("f1-score", float("nan")),
            "F1 OFF": per.get("OFF", {}).get("f1-score", float("nan")),
        })
    except Exception as e:
        print(f"[WARN] skipping {p.name}: {e}")

df = pd.DataFrame(rows)
if df.empty:
    # Safe placeholder so LaTeX compiles
    latex = r"""
\begin{table}[ht]
\centering
\caption{Development results (macro-F1).}
\label{tab:dev-results}
\begin{tabular}{lccc}
\toprule
Model & Macro F1 & F1 (NOT) & F1 (OFF) \\
\midrule
\textit{(no results yet)} & -- & -- & -- \\
\bottomrule
\end{tabular}
\end{table}
""".strip()
else:
    df = df.sort_values("Macro F1", ascending=False)
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Development results (macro-F1).}")
    lines.append(r"\label{tab:dev-results}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Model & Macro F1 & F1 (NOT) & F1 (OFF) \\")
    lines.append(r"\midrule")
    for _, r in df.iterrows():
        def fmt(x): 
            try: return f"{float(x):.4f}"
            except: return "--"
        lines.append(f"{r['Model']} & {fmt(r['Macro F1'])} & {fmt(r['F1 NOT'])} & {fmt(r['F1 OFF'])} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    latex = "\n".join(lines)

out_path = "outputs/dev_results_table.tex"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(latex)
print(f"[OK] Wrote LaTeX table to {out_path}")
