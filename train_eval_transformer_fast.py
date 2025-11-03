import argparse, os, json, random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding
from sklearn.svm import LinearSVC
from utils_data import read_tsv, prepare_dataframe
from utils_eval import full_report

# -------- label helpers (same mapping as SVM/LSTM) --------
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
    # 1) name-based
    for c in df.columns:
        if c == text_col: continue
        if any(k in c.lower() for k in LIKELY_LABEL_NAMES):
            mapped, ok = _try_map_to_not_off(df[c])
            if ok:
                df[c] = mapped
                return c
    # 2) small-cardinality
    for c in df.columns:
        if c == text_col: continue
        uniq = pd.Series(df[c]).dropna().unique()
        if len(uniq) <= 3:
            mapped, ok = _try_map_to_not_off(df[c])
            if ok:
                df[c] = mapped
                return c
    return None

def set_seeds(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# -------- dataset for embedding extraction --------
class EncodedTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        return self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
            add_special_tokens=True
        )

@torch.no_grad()
def featurize(model, loader, device, pooling="mean"):
    model.eval()
    feats = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        last = out.last_hidden_state                      # [B,T,H]
        mask = batch["attention_mask"].unsqueeze(-1)      # [B,T,1]
        if pooling == "cls":
            emb = last[:, 0, :]                           # [B,H]
        else:
            summed = (last * mask).sum(dim=1)             # [B,H]
            counts = mask.sum(dim=1).clamp(min=1)         # [B,1]
            emb = summed / counts
        feats.append(emb.cpu().numpy())
    return np.vstack(feats) if feats else np.zeros((0, model.config.hidden_size), dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True)
    ap.add_argument('--dev', required=True)
    ap.add_argument('--test', required=True)
    ap.add_argument('--model', default='prajjwal1/bert-tiny')  # FAST default
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--max_len', type=int, default=96)
    ap.add_argument('--pooling', choices=['mean','cls'], default='mean')
    ap.add_argument('--normalize_twitter', type=int, default=1)
    ap.add_argument('--mask_lexicon', type=int, default=0)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    set_seeds(args.seed)

    # Load & prepare
    train_raw = read_tsv(args.train)
    dev_raw   = read_tsv(args.dev)
    test_raw  = read_tsv(args.test)

    train, tr_text, tr_label = prepare_dataframe(train_raw, bool(args.normalize_twitter), bool(args.mask_lexicon))
    dev,   dv_text, dv_label_guess = prepare_dataframe(dev_raw,   bool(args.normalize_twitter), bool(args.mask_lexicon))
    test,  te_text, _              = prepare_dataframe(test_raw,  bool(args.normalize_twitter), bool(args.mask_lexicon))

    if tr_label is None or tr_label not in train.columns:
        raise ValueError("Could not find a label column in TRAIN after preparation. Please check train.tsv.")
    dv_label = dv_label_guess if (dv_label_guess in dev.columns) else infer_label_col(dev, dv_text)

    label2id = {'NOT':0,'OFF':1}
    id2label = {0:'NOT',1:'OFF'}

    # Tokenizer & base encoder (frozen)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    encoder   = AutoModel.from_pretrained(args.model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder.to(device)
    encoder.eval()

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Build loaders
    train_ds = EncodedTextDataset(train[tr_text].tolist(), tokenizer, args.max_len)
    dev_ds   = EncodedTextDataset(dev[dv_text].tolist(), tokenizer, args.max_len)
    test_ds  = EncodedTextDataset(test[te_text].tolist(), tokenizer, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=False, collate_fn=collator, num_workers=0)
    dev_loader   = DataLoader(dev_ds,   batch_size=args.batch, shuffle=False, collate_fn=collator, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False, collate_fn=collator, num_workers=0)

    # Extract embeddings quickly
    with torch.no_grad():
        X_train = featurize(encoder, train_loader, device, pooling=args.pooling)
        X_dev   = featurize(encoder, dev_loader,   device, pooling=args.pooling)
        X_test  = featurize(encoder, test_loader,  device, pooling=args.pooling)

    y_train = np.array([label2id[v] for v in train[tr_label].tolist()], dtype=np.int64)
    y_dev   = np.array([label2id[v] for v in dev[dv_label].tolist()],   dtype=np.int64) if dv_label else None

    # Train a fast linear head
    clf = LinearSVC()
    clf.fit(X_train, y_train)

    os.makedirs('outputs', exist_ok=True)

    # Dev report
    if y_dev is not None:
        dev_pred = clf.predict(X_dev)
        rep = full_report([id2label[i] for i in y_dev], [id2label[i] for i in dev_pred])
        with open('outputs/transformer_fast_dev_report.json','w') as f:
            json.dump(rep, f, indent=2)
        print(json.dumps(rep, indent=2))
    else:
        print("[INFO] Dev labels not found; skipping dev report.")

    # Test predictions
    test_pred = clf.predict(X_test)
    pd.DataFrame({"pred": [id2label[i] for i in test_pred]}).to_csv('outputs/transformer_fast_test_predictions.tsv', sep='\t', index=False)

if __name__ == "__main__":
    main()
