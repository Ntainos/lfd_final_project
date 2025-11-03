from typing import Dict
from sklearn.metrics import f1_score, classification_report, confusion_matrix

def macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average='macro')

def full_report(y_true, y_pred, labels=('NOT','OFF')) -> Dict:
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    rep = classification_report(y_true, y_pred, labels=list(labels), output_dict=True, zero_division=0)
    return {
        "macro_f1": macro_f1(y_true, y_pred),
        "labels": list(labels),
        "confusion_matrix": cm.tolist(),
        "per_class": {k:v for k,v in rep.items() if k in labels},
    }
