# export_fp_fn_examples.py
import argparse, os, pandas as pd, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from utils_data import read_tsv, prepare_dataframe

class EncodedTextDataset(Dataset):
    def __init__(self, texts, tok, max_len):
        self.texts = texts; self.tok = tok; self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        return self.tok(self.texts[i], truncation=True, max_length=self.max_len, add_special_tokens=True)

@torch.no_grad()
def featurize(model, loader, device, pooling="mean"):
    model.eval(); outs=[]
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        last = model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1)
        if pooling == "cls":
            emb = last[:,0,:]
        else:
            emb = (last*mask).sum(1) / mask.sum(1).clamp(min=1)
        outs.append(emb.cpu().numpy())
    return np.vstack(outs) if outs else np.zeros((0, model.config.hidden_size))

def build_svm_features(X_train, X_dev, cfg):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion
    word = TfidfVectorizer(ngram_range=cfg.get("word_ngram",(1,2)), min_df=cfg.get("min_df",1),
                           max_df=cfg.get("max_df",0.95), sublinear_tf=True, lowercase=True)
    char = TfidfVectorizer(analyzer='char', ngram_range=cfg.get("char_ngram",(3,6)),
                           min_df=cfg.get("min_df",1), sublinear_tf=True, lowercase=True)
    fu = FeatureUnion([('word', word), ('char', char)])
    Xtr = fu.fit_transform(X_train)
    Xdv = fu.transform(X_dev)
    return Xtr, Xdv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--model_type", choices=["svm","xfmr_fast"], required=True)
    ap.add_argument("--xfmr_name", default="distilroberta-base")
    ap.add_argument("--pooling", choices=["mean","cls"], default="mean")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--topn", type=int, default=40)
    ap.add_argument("--normalize_twitter", type=int, default=1)
    ap.add_argument("--mask_lexicon", type=int, default=0)
    args = ap.parse_args()

    tr_raw = read_tsv(args.train); dv_raw = read_tsv(args.dev)
    tr, tr_text, tr_label = prepare_dataframe(tr_raw, bool(args.normalize_twitter), bool(args.mask_lexicon))
    dv, dv_text, dv_label = prepare_dataframe(dv_raw, bool(args.normalize_twitter), bool(args.mask_lexicon))[0:3]
    if dv_label is None:
        raise SystemExit("Dev split has no labels; cannot export FP/FN examples.")

    y_tr = tr[tr_label].tolist()
    X_tr_text = tr[tr_text].tolist()
    y_dv = dv[dv_label].tolist()
    X_dv_text = dv[dv_text].tolist()

    id2lab = {0:"NOT", 1:"OFF"}; lab2id = {"NOT":0, "OFF":1}

    if args.model_type == "svm":
        # your best SVM cfg (norm toggle is handled by prepare_dataframe via --normalize_twitter flag)
        cfg = {"word_ngram":(1,2), "char_ngram":(3,6), "min_df":1, "max_df":0.95}
        Xtr, Xdv = build_svm_features(X_tr_text, X_dv_text, cfg)
        clf = LinearSVC(C=0.5, class_weight="balanced").fit(Xtr, [lab2id[y] for y in y_tr])
        preds = clf.predict(Xdv)
    else:
        tok = AutoTokenizer.from_pretrained(args.xfmr_name, use_fast=True)
        enc = AutoModel.from_pretrained(args.xfmr_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        enc.to(device).eval()
        coll = DataCollatorWithPadding(tokenizer=tok)
        tr_ds = EncodedTextDataset(X_tr_text, tok, args.max_len)
        dv_ds = EncodedTextDataset(X_dv_text, tok, args.max_len)
        tr_loader = DataLoader(tr_ds, batch_size=64, shuffle=False, collate_fn=coll)
        dv_loader = DataLoader(dv_ds, batch_size=64, shuffle=False, collate_fn=coll)
        Xtr = featurize(enc, tr_loader, device, pooling=args.pooling)
        Xdv = featurize(enc, dv_loader, device, pooling=args.pooling)
        clf = LinearSVC().fit(Xtr, [lab2id[y] for y in y_tr])
        preds = clf.predict(Xdv)

    gold_ids = [lab2id[y] for y in y_dv]
    rows = []
    for i, (g, p, txt) in enumerate(zip(gold_ids, preds, X_dv_text)):
        if g != p:
            rows.append({
                "idx": i,
                "gold": id2lab[g],
                "pred": id2lab[p],
                "error_type": "FN" if (g==1 and p==0) else ("FP" if (g==0 and p==1) else "other"),
                "text": txt
            })

    # prioritize OFF-related errors first (FNs), then FPs
    rows.sort(key=lambda r: (r["error_type"]!="FN", r["error_type"]!="FP"))
    df = pd.DataFrame(rows[:args.topn])

    os.makedirs("outputs", exist_ok=True)
    outp = f"outputs/dev_errors_{args.model_type}.csv"
    df.to_csv(outp, index=False)

    cm = confusion_matrix(gold_ids, preds, labels=[0,1])
    print("Confusion matrix [NOT, OFF] rows vs cols:\n", cm)
    print(f"[OK] wrote top {len(df)} errors to {outp}")

if __name__ == "__main__":
    main()
