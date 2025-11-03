import re, pandas as pd
from typing import Tuple, Iterable, Optional, List

# --- small lexicon for the masking experiment ---
OFFENSIVE_LEXICON = set([
    "fuck","fucking","bitch","idiot","moron","dumb","stupid","asshole","shit","cocksucker","retard"
])

TWITTER_URL = re.compile(r'http\S+|www\.\S+')
TWITTER_USER = re.compile(r'@\w+')

# ---------- helpers for robust TSV loading ----------

def _suspicious_header(colnames: List[str]) -> bool:
    """
    Heuristics: if any column name looks like actual tweet text or a label value ('NOT'/'OFF'),
    we assume the file had NO header and pandas used the first data row as header.
    """
    for name in colnames:
        n = str(name)
        if n.upper() in {"NOT","OFF"}:
            return True
        if "@" in n or "http" in n or "#" in n:
            return True
        if " " in n and len(n) > 15:  # long, spacey column name looks like free text
            return True
    return False

def read_tsv(path: str) -> pd.DataFrame:
    """
    Robust TSV reader:
    1) try header=0;
    2) if header looks suspicious, reload with header=None and generic col names.
    """
    df = pd.read_csv(path, sep='\t', engine='python', dtype=str)
    if _suspicious_header(list(df.columns)):
        df = pd.read_csv(path, sep='\t', engine='python', header=None, dtype=str)
        df.columns = [f"c{i}" for i in range(df.shape[1])]
    return df

# ---------- label/text guessing ----------

def _try_map_to_not_off(series: Iterable[str]) -> tuple[list[str], float]:
    """
    Attempt to map values to NOT/OFF. Return (mapped_values, ratio_mapped).
    """
    def map_one(v):
        s = str(v).strip().upper()
        if s.startswith("N"): return "NOT"
        if s.startswith("O"): return "OFF"
        if s in {"0","NOT_OFFENSIVE"}: return "NOT"
        if s in {"1","OFFENSIVE"}: return "OFF"
        return None
    mapped = [map_one(v) for v in series]
    ok_ratio = sum(m is not None for m in mapped) / max(1, len(mapped))
    mapped_filled = [m if m is not None else str(v) for m, v in zip(mapped, series)]
    return mapped_filled, ok_ratio

def _mean_strlen(series: Iterable[str]) -> float:
    s = pd.Series(series, dtype="string")
    return float(s.str.len().fillna(0).mean())

def guess_cols(df: pd.DataFrame) -> Tuple[str, Optional[str]]:
    """
    Pick the *text* column as the object/string column with the largest average length.
    Then pick a *label* column (if any) that maps to NOT/OFF with >= 0.8 ratio.
    """
    candidates = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype) == "string"]
    if not candidates:
        text_col = df.columns[-1]
    else:
        lens = {c: _mean_strlen(df[c]) for c in candidates}
        text_col = max(lens, key=lens.get)

    label_col: Optional[str] = None
    best_ratio = 0.0
    for c in df.columns:
        if c == text_col:
            continue
        mapped, ratio = _try_map_to_not_off(df[c])
        if ratio > best_ratio and ratio >= 0.8:
            best_ratio = ratio
            label_col = c
            df[c] = mapped
    return text_col, label_col

# ---------- normalization / masking pipeline ----------

def normalize_label(y: Iterable[str]) -> Iterable[str]:
    def norm(v):
        s = str(v).strip().upper()
        if s.startswith('N'): return 'NOT'
        if s.startswith('O'): return 'OFF'
        return s
    return [norm(v) for v in y]

def basic_twitter_normalize(s: str, normalize_twitter: bool = True) -> str:
    if not isinstance(s, str):
        s = str(s)
    if normalize_twitter:
        s = TWITTER_URL.sub('URL', s)
        s = TWITTER_USER.sub('USER', s)
    return s

def mask_offensive(s: str, do_mask: bool) -> str:
    if not do_mask:
        return s
    toks = re.findall(r"\w+|[^\w\s]", s, flags=re.UNICODE)
    out = ["OFFENSIVE" if t.lower() in OFFENSIVE_LEXICON else t for t in toks]
    return ' '.join(out)

def prepare_dataframe(df: pd.DataFrame, normalize_twitter: bool = True, mask_lexicon: bool = False) -> tuple[pd.DataFrame, str, Optional[str]]:
    df = df.copy()
    text_col, label_col = guess_cols(df)
    df[text_col] = df[text_col].astype(str).map(lambda s: mask_offensive(basic_twitter_normalize(s, normalize_twitter), mask_lexicon))
    if label_col and label_col in df.columns:
        df[label_col] = list(normalize_label(df[label_col]))
    return df, text_col, label_col
