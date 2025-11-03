import json, glob, pandas as pd

rows = []
for path in sorted(glob.glob("outputs/*_dev_report.json")):
    with open(path) as f:
        rep = json.load(f)
    macro = rep.get("macro_f1")
    per = rep.get("per_class", {})
    rows.append({
        "report": path,
        "macro_f1": round(macro, 4) if macro is not None else None,
        "NOT_f1": round(per.get("NOT", {}).get("f1-score", 0), 4),
        "OFF_f1": round(per.get("OFF", {}).get("f1-score", 0), 4)
    })

df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
print("\n=== DEV RESULTS SUMMARY ===\n")
print(df.to_string(index=False))
print("\nCopy this table into your report.")
