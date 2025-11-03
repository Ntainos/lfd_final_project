import argparse, os, json, re
import pandas as pd
import numpy as np
from collections import Counter
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from utils_data import read_tsv, prepare_dataframe
from utils_eval import full_report

# ---- Tokenization ----
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
def tokenize(s: str):
    return TOKEN_RE.findall(s.lower())

# ---- Dataset ----
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=80):
        self.texts = texts
        self.labels = labels  # list[str] with 'NOT'/'OFF' or None for test
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self): return len(self.texts)

    def encode(self, text):
        ids = [self.vocab.get(t, 1) for t in tokenize(text)][:self.max_len]
        if len(ids) < self.max_len:
            ids += [0] * (self.max_len - len(ids))
        return np.array(ids, dtype=np.int64)

    def __getitem__(self, idx):
        x = self.encode(self.texts[idx])
        if self.labels is None:
            y = -1
        else:
            lab = self.labels[idx]
            y = 0 if lab == 'NOT' else 1
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

# ---- Model ----
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim=200, hidden=192, bidir=True, pad_idx=0, dropout=0.30):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=bidir)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * (2 if bidir else 1), 2)

    def forward(self, x):
        emb = self.emb(x)                       # [B, T, E]
        out, (h, c) = self.lstm(emb)            # h: [num_layers*dir, B, H]
        if self.lstm.bidirectional:
            h_cat = torch.cat([h[-2], h[-1]], dim=1)  # [B, 2H]
        else:
            h_cat = h[-1]                        # [B, H]
        logits = self.fc(self.dropout(h_cat))    # [B, 2]
        return logits

# ---- Vocab ----
def build_vocab(all_texts, max_vocab=50000, min_freq=1):
    cnt = Counter()
    for s in all_texts:
        cnt.update(tokenize(s))
    vocab = {"<pad>":0, "<unk>":1}
    for tok, c in cnt.most_common(max_vocab):
        if c >= min_freq and tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab

# ---- Training / Prediction ----
def train_epoch(model, loader, optim, device, loss_fn):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optim.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optim.step()
        loss_sum += loss.detach().item() * xb.size(0)
        preds = logits.argmax(1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)
    return loss_sum/total, correct/total

@torch.no_grad()
def predict_ids(model, loader, device):
    model.eval()
    preds, golds = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        pred = logits.argmax(1).cpu().numpy().tolist()
        preds.extend(pred)
        golds.extend(yb.numpy().tolist())
    return preds, golds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True)
    ap.add_argument('--dev', required=True)
    ap.add_argument('--test', required=True)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--max_len', type=int, default=80)
    ap.add_argument('--emb_dim', type=int, default=200)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--normalize_twitter', type=int, default=1)
    ap.add_argument('--mask_lexicon', type=int, default=0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--class_weight', type=int, default=1)  # 1 = use weighted CE
    ap.add_argument('--patience', type=int, default=2)      # early stopping patience
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_df = read_tsv(args.train)
    dev_df   = read_tsv(args.dev)
    test_df  = read_tsv(args.test)

    train_df, tr_text, tr_label = prepare_dataframe(train_df, bool(args.normalize_twitter), bool(args.mask_lexicon))
    dev_df,   dv_text, dv_label_guess = prepare_dataframe(dev_df,   bool(args.normalize_twitter), bool(args.mask_lexicon))
    test_df,  te_text, _              = prepare_dataframe(test_df,  bool(args.normalize_twitter), bool(args.mask_lexicon))

    # infer dev label if needed
    dv_label = dv_label_guess if (dv_label_guess in dev_df.columns) else dv_label_guess

    # Build vocab
    all_texts = pd.concat([train_df[tr_text], dev_df[dv_text], test_df[te_text]]).tolist()
    vocab = build_vocab(all_texts, max_vocab=50000, min_freq=1)

    # Datasets / loaders
    ds_train = TextDataset(train_df[tr_text].tolist(), train_df[tr_label].tolist(), vocab, max_len=args.max_len)
    ds_dev   = TextDataset(dev_df[dv_text].tolist(),   dev_df[dv_label].tolist() if dv_label else None, vocab, max_len=args.max_len)
    ds_test  = TextDataset(test_df[te_text].tolist(),  None, vocab, max_len=args.max_len)

    train_loader = DataLoader(ds_train, batch_size=args.batch, shuffle=True,  num_workers=0)
    dev_loader   = DataLoader(ds_dev,   batch_size=args.batch, shuffle=False, num_workers=0)
    test_loader  = DataLoader(ds_test,  batch_size=args.batch, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMClassifier(vocab_size=len(vocab), emb_dim=args.emb_dim, hidden=192, bidir=True, pad_idx=0, dropout=0.30).to(device)

    # Class weights to help OFF recall
    if args.class_weight:
        counts = Counter(train_df[tr_label].tolist())
        n_total = sum(counts.values())
        w_not = n_total / (2.0 * counts.get('NOT', 1))
        w_off = n_total / (2.0 * counts.get('OFF', 1))
        weights = torch.tensor([w_not, w_off], dtype=torch.float32, device=device)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
    else:
        loss_fn = nn.CrossEntropyLoss()

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs('outputs', exist_ok=True)

    # Train with early stopping on dev macro-F1
    best_f1, best_state, bad_epochs = -1.0, None, 0
    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optim, device, loss_fn)

        # Dev eval
        if dv_label:
            dev_pred_ids, dev_gold_ids = predict_ids(model, dev_loader, device)
            id2lab = {0:'NOT', 1:'OFF'}
            dev_pred = [id2lab[i] for i in dev_pred_ids]
            dev_gold = [id2lab[i] for i in dev_gold_ids]
            rep = full_report(dev_gold, dev_pred)
            macro = rep["macro_f1"]
            print(f"Epoch {epoch}/{args.epochs} - train loss {tr_loss:.4f} | acc {tr_acc:.4f} | dev macro-F1 {macro:.4f}")

            # keep best
            if macro > best_f1:
                best_f1 = macro
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                with open('outputs/lstm_dev_report.json','w') as f:
                    json.dump(rep, f, indent=2)
                bad_epochs = 0
            else:
                bad_epochs += 1

            if bad_epochs > args.patience:
                print(f"[EARLY STOP] No dev improvement for {args.patience} epochs.")
                break
        else:
            print(f"Epoch {epoch}/{args.epochs} - train loss {tr_loss:.4f} | acc {tr_acc:.4f}")

    # Load best state if we had dev
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"[INFO] Loaded best LSTM weights (dev macro-F1={best_f1:.4f})")

    # Test predictions
    test_pred_ids, _ = predict_ids(model, test_loader, device)
    id2lab = {0:'NOT', 1:'OFF'}
    test_pred = [id2lab[i] for i in test_pred_ids]
    pd.DataFrame({"pred": test_pred}).to_csv('outputs/lstm_test_predictions.tsv', sep='\t', index=False)

if __name__ == '__main__':
    main()
