# make_error_examples_tex.py
import pandas as pd, os, html, re

def latex_escape(s: str) -> str:
    repl = {
        '&':'\\&','%':'\\%','$':'\\$','#':'\\#','_':'\\_','{':'\\{','}':'\\}','~':'\\textasciitilde{}','^':'\\textasciicircum{}','\\':'\\textbackslash{}'
    }
    return ''.join(repl.get(c,c) for c in s)

os.makedirs("outputs", exist_ok=True)
df = pd.read_csv("outputs/dev_errors_xfmr_fast.csv")
# take 3 FNs (OFF->NOT) and 2 FPs (NOT->OFF) if available
fns = df[df["error_type"]=="FN"].head(3)
fps = df[df["error_type"]=="FP"].head(2)
sel = pd.concat([fns, fps], ignore_index=True)

lines = [r"\begin{table}[ht]\centering", r"\small", r"\begin{tabular}{lll}", r"\toprule",
         r"Gold $\rightarrow$ Pred & Error & Text \\", r"\midrule"]
for _, r in sel.iterrows():
    pair = f"{r['gold']}\\,$\\rightarrow$\\,{r['pred']}"
    et = r["error_type"]
    txt = latex_escape(str(r["text"])[:140])  # truncate for layout
    lines.append(f"{pair} & {et} & {txt} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\caption{Representative development errors (truncated for space).}", r"\end{table}"]
open("outputs/example_errors.tex","w",encoding="utf-8").write("\n".join(lines))
print("[OK] wrote outputs/example_errors.tex")
