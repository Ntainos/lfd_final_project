# export_data_summary_tex.py
import argparse, os, pandas as pd, numpy as np
from utils_data import read_tsv, prepare_dataframe

def summarize(df, text_col, label_col):
    n = len(df)
    lens_tok = df[text_col].astype(str).apply(lambda s: len(s.split()))
    lens_chr = df[text_col].astype(str).apply(len)
    stats = {
        "N": n,
        "Avg tokens": lens_tok.mean(),
        "Avg chars": lens_chr.mean(),
    }
    by = df[label_col].value_counts(dropna=False)
    for lab, cnt in by.items():
        stats[f"{lab}"] = cnt
        stats[f"{lab}%"] = 100.0 * cnt / max(1, n)
    return stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--normalize_twitter", type=int, default=1)
    ap.add_argument("--mask_lexicon", type=int, default=0)
    args = ap.parse_args()

    os.makedirs("outputs", exist_ok=True)

    tr_raw = read_tsv(args.train); dv_raw = read_tsv(args.dev); te_raw = read_tsv(args.test)
    tr, tr_text, tr_label = prepare_dataframe(tr_raw, bool(args.normalize_twitter), bool(args.mask_lexicon))
    dv, dv_text, dv_label = prepare_dataframe(dv_raw, bool(args.normalize_twitter), bool(args.mask_lexicon))[0:3]
    te, te_text, _        = prepare_dataframe(te_raw, bool(args.normalize_twitter), bool(args.mask_lexicon))

    rows = []
    rows.append(("Train", summarize(tr, tr_text, tr_label)))
    if dv_label is None:
        raise SystemExit("Dev labels not found; cannot summarize dev.")
    rows.append(("Dev", summarize(dv, dv_text, dv_label)))
    rows.append(("Test", {"N": len(te), "Avg tokens": te[te_text].astype(str).apply(lambda s: len(s.split())).mean(),
                          "Avg chars": te[te_text].astype(str).apply(len).mean()}))

    # Build LaTeX
    cols = ["Split","N","NOT","NOT%","OFF","OFF%","Avg tokens","Avg chars"]
    lines = [r"\begin{table}[ht]\centering", r"\small", r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"Split & N & NOT & NOT(\%) & OFF & OFF(\%) & Avg tok & Avg char \\",
             r"\midrule"]
    for split, st in rows:
        N = int(st.get("N",0))
        NOT = int(st.get("NOT",0)); NOTp = st.get("NOT%",0.0)
        OFF = int(st.get("OFF",0)); OFFp = st.get("OFF%",0.0)
        tavg = st.get("Avg tokens",0.0); cavg = st.get("Avg chars",0.0)
        lines.append(f"{split} & {N} & {NOT} & {NOTp:.1f} & {OFF} & {OFFp:.1f} & {tavg:.1f} & {cavg:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\caption{Dataset summary and basic statistics.}", r"\end{table}"]
    with open("outputs/data_summary.tex","w",encoding="utf-8") as f: f.write("\n".join(lines))
    print("[OK] wrote outputs/data_summary.tex")

if __name__ == "__main__":
    main()
