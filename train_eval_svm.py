import argparse, json, os, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from utils_data import read_tsv, prepare_dataframe
from utils_eval import full_report

# --- mapping helpers ---
LIKELY_LABEL_NAMES = ("label","subtask_a","target","class","y")
def _try_map_to_not_off(series):
    def map_one(v):
        s = str(v).strip().upper()
        if s.startswith("N"): return "NOT"
        if s.startswith("O"): return "OFF"
        if s in {"0","NOT_OFFENSIVE"}: return "NOT"
        if s in {"1","OFFENSIVE"}:    return "OFF"
        return None
    mapped = [map_one(v) for v in series]
    ok_ratio = sum(m is not None for m in mapped) / max(1,len(mapped))
    mapped_filled = [m if m is not None else str(v) for m, v in zip(mapped, series)]
    return mapped_filled, ok_ratio >= 0.80

def infer_label_col(df: pd.DataFrame, text_col: str) -> str | None:
    # 1) prefer known label-like names
    for c in df.columns:
        if c == text_col: continue
        name = c.lower()
        if any(k in name for k in LIKELY_LABEL_NAMES):
            mapped, ok = _try_map_to_not_off(df[c])
            if ok:
                df[c] = mapped
                return c
    # 2) small-cardinality columns (<=3 unique) that map well
    for c in df.columns:
        if c == text_col: continue
        uniq = pd.Series(df[c]).dropna().unique()
        if len(uniq) <= 3:
            mapped, ok = _try_map_to_not_off(df[c])
            if ok:
                df[c] = mapped
                return c
    return None

def build_union(word_ngram=(1,2), char_ngram=(3,5), min_df=2, max_df=0.95):
    word = TfidfVectorizer(ngram_range=word_ngram, min_df=min_df, max_df=max_df, sublinear_tf=True, lowercase=True)
    char = TfidfVectorizer(analyzer='char', ngram_range=char_ngram, min_df=min_df, sublinear_tf=True, lowercase=True)
    return FeatureUnion([('word', word), ('char', char)])

def fit_and_eval(cfg, X_train, y_train, X_dev, y_dev):
    pipe = Pipeline([
        ('tfidf', build_union(cfg['word_ngram'], cfg['char_ngram'], cfg['min_df'], cfg['max_df'])),
        ('clf', LinearSVC(C=cfg['C'], class_weight=cfg['class_weight']))
    ])
    pipe.fit(X_train, y_train)
    y_dev_pred = pipe.predict(X_dev)
    report = full_report(y_dev, y_dev_pred)
    return report['macro_f1'], report, pipe

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True)
    ap.add_argument('--dev', required=True)
    ap.add_argument('--test', required=True)
    ap.add_argument('--normalize_twitter', type=int, default=1)
    ap.add_argument('--mask_lexicon', type=int, default=0)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    # Load
    train_raw = read_tsv(args.train)
    dev_raw   = read_tsv(args.dev)
    test_raw  = read_tsv(args.test)

    # Prepare
    train, tr_text, tr_label = prepare_dataframe(train_raw, bool(args.normalize_twitter), bool(args.mask_lexicon))
    dev,   dv_text, dv_label_guess = prepare_dataframe(dev_raw,   bool(args.normalize_twitter), bool(args.mask_lexicon))
    test,  te_text, _              = prepare_dataframe(test_raw,  bool(args.normalize_twitter), bool(args.mask_lexicon))

    dv_label = dv_label_guess if (dv_label_guess in dev.columns) else infer_label_col(dev, dv_text)

    X_train, y_train = train[tr_text].tolist(), train[tr_label].tolist()
    X_dev = dev[dv_text].tolist()
    y_dev = dev[dv_label].tolist() if dv_label else None
    X_test = test[te_text].tolist()

    os.makedirs('outputs', exist_ok=True)

    # If no dev labels, just train a solid default and export predictions
    if y_dev is None:
        pipe = Pipeline([
            ('tfidf', build_union()),
            ('clf', LinearSVC(C=1.0, class_weight='balanced'))
        ])
        pipe.fit(X_train, y_train)
        y_test_pred = pipe.predict(X_test)
        pd.DataFrame({"pred": y_test_pred}).to_csv('outputs/svm_test_predictions.tsv', sep='\t', index=False)
        print("[INFO] Dev labels not found; exported test predictions with default SVM.")
        return

    # Tiny, fast grid (≈ a few seconds)
    grid = [
        # slightly different char ranges and C; always try class_weight='balanced'
        {"word_ngram": (1,2), "char_ngram": (3,5), "min_df": 1, "max_df": 0.95, "C": 1.0, "class_weight": "balanced"},
        {"word_ngram": (1,3), "char_ngram": (3,6), "min_df": 1, "max_df": 0.95, "C": 1.0, "class_weight": "balanced"},
        {"word_ngram": (1,2), "char_ngram": (2,5), "min_df": 1, "max_df": 0.95, "C": 2.0, "class_weight": "balanced"},
        {"word_ngram": (1,3), "char_ngram": (3,6), "min_df": 2, "max_df": 0.90, "C": 2.0, "class_weight": "balanced"},
        {"word_ngram": (1,2), "char_ngram": (3,6), "min_df": 1, "max_df": 0.95, "C": 0.5, "class_weight": "balanced"},
        {"word_ngram": (1,3), "char_ngram": (2,5), "min_df": 1, "max_df": 0.95, "C": 4.0, "class_weight": "balanced"},
    ]

    best = {"score": -1, "cfg": None, "report": None, "pipe": None}
    for i, cfg in enumerate(grid, 1):
        try:
            score, rep, pipe = fit_and_eval(cfg, X_train, y_train, X_dev, y_dev)
            print(f"[{i}/{len(grid)}] macro-F1={score:.4f} cfg={cfg}")
            if score > best["score"]:
                best = {"score": score, "cfg": cfg, "report": rep, "pipe": pipe}
        except Exception as e:
            print(f"[WARN] cfg failed {cfg} -> {e}")

    # Save best
    if best["report"] is not None:
        with open('outputs/svm_dev_report.json','w') as f:
            json.dump(best["report"], f, indent=2)
        print(json.dumps(best["report"], indent=2))
        # test
        y_test_pred = best["pipe"].predict(X_test)
        pd.DataFrame({"pred": y_test_pred}).to_csv('outputs/svm_test_predictions.tsv', sep='\t', index=False)
        print(f"[OK] Saved best SVM (macro-F1={best['score']:.4f}) and test predictions.\nBest cfg: {best['cfg']}")
    else:
        # fallback
        pipe = Pipeline([('tfidf', build_union()), ('clf', LinearSVC(C=1.0, class_weight='balanced'))])
        pipe.fit(X_train, y_train)
        y_test_pred = pipe.predict(X_test)
        pd.DataFrame({"pred": y_test_pred}).to_csv('outputs/svm_test_predictions.tsv', sep='\t', index=False)
        print("[INFO] Fallback SVM trained; dev selection unavailable.")
    
if __name__ == '__main__':
    main()
