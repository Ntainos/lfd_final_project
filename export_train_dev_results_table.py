# export_train_dev_results_table.py
import os, argparse, numpy as np, pandas as pd, torch
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding
from utils_data import read_tsv, prepare_dataframe

# --- SVM features ---
def build_svm_features(X_train, X_eval):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion
    word = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_df=0.95, sublinear_tf=True, lowercase=True)
    char = TfidfVectorizer(analyzer='char', ngram_range=(3,6), min_df=1, sublinear_tf=True, lowercase=True)
    fu = FeatureUnion([('word', word), ('char', char)])
    Xtr = fu.fit_transform(X_train); Xev = fu.transform(X_eval)
    return Xtr, Xev

# --- Frozen encoder dataset ---
class EncodedTextDataset(Dataset):
    def __init__(self, texts, tok, max_len):
        self.texts = texts; self.tok = tok; self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        return self.tok(self.texts[i], truncation=True, max_length=self.max_len, add_special_tokens=True)

@torch.no_grad()
def featurize(model, loader, device, pooling="cls"):
    model.eval(); outs=[]
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        last = model(**batch).last_hidden_state
        if pooling == "cls":
            emb = last[:,0,:]
        else:
            mask = batch["attention_mask"].unsqueeze(-1)
            emb = (last*mask).sum(1) / mask.sum(1).clamp(min=1)
        outs.append(emb.cpu().numpy())
    return np.vstack(outs) if outs else None

def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", labels=["NOT","OFF"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True); ap.add_argument("--dev", required=True)
    args = ap.parse_args()
    os.makedirs("outputs", exist_ok=True)

    tr_raw = read_tsv(args.train); dv_raw = read_tsv(args.dev)
    tr, tr_text, tr_label = prepare_dataframe(tr_raw, True, False)
    dv, dv_text, dv_label = prepare_dataframe(dv_raw, True, False)[0:3]
    if dv_label is None: raise SystemExit("Dev labels missing.")

    # ----- SVM (norm OFF) -----
    tr_no = prepare_dataframe(tr_raw, False, False)[0]
    dv_no = prepare_dataframe(dv_raw, False, False)[0]
    Xtr, Xdv = build_svm_features(tr_no[tr_text].tolist(), dv_no[dv_text].tolist())
    ytr = tr_no[tr_label].tolist(); ydv = dv_no[dv_label].tolist()
    svm = LinearSVC(C=0.5, class_weight="balanced").fit(Xtr, ytr)
    ytr_hat = svm.predict(Xtr); ydv_hat = svm.predict(Xdv)
    svm_train_f1 = macro_f1(ytr, ytr_hat); svm_dev_f1 = macro_f1(ydv, ydv_hat)

    # ----- Frozen DistilRoBERTa CLS + LinearSVC -----
    tok = AutoTokenizer.from_pretrained("distilroberta-base", use_fast=True)
    enc = AutoModel.from_pretrained("distilroberta-base")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc.to(dev).eval()
    coll = DataCollatorWithPadding(tokenizer=tok)
    tr_ds = EncodedTextDataset(tr[tr_text].tolist(), tok, 128)
    dv_ds = EncodedTextDataset(dv[dv_text].tolist(), tok, 128)
    tr_loader = DataLoader(tr_ds, batch_size=64, shuffle=False, collate_fn=coll)
    dv_loader = DataLoader(dv_ds, batch_size=64, shuffle=False, collate_fn=coll)
    Xtr2 = featurize(enc, tr_loader, dev, "cls"); Xdv2 = featurize(enc, dv_loader, dev, "cls")
    ytr2 = tr[tr_label].tolist(); ydv2 = dv[dv_label].tolist()
    lin = LinearSVC().fit(Xtr2, ytr2)
    ytr2_hat = lin.predict(Xtr2); ydv2_hat = lin.predict(Xdv2)
    xf_train_f1 = macro_f1(ytr2, ytr2_hat); xf_dev_f1 = macro_f1(ydv2, ydv2_hat)

    # ---- write LaTeX table ----
    lines = [r"\begin{table}[ht]\centering", r"\small", r"\begin{tabular}{lcc}", r"\toprule",
             r"Model & Macro F1 (train) & Macro F1 (dev) \\", r"\midrule",
             f"SVM (norm off) & {svm_train_f1:.4f} & {svm_dev_f1:.4f} \\\\",
             f"Frozen DistilRoBERTa + CLS + LinearSVC & {xf_train_f1:.4f} & {xf_dev_f1:.4f} \\\\",
             r"\bottomrule", r"\end{tabular}",
             r"\caption{Train vs development macro-F1 for the two strongest models.}", r"\end{table}"]
    open("outputs/train_dev_results_table.tex","w",encoding="utf-8").write("\n".join(lines))
    print("[OK] wrote outputs/train_dev_results_table.tex")

if __name__ == "__main__":
    main()
