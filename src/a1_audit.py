"""Part A task 2 - audit the three raw tables (shape, dtypes, missing values, planted defects)."""
import pandas as pd
from common import RAW

pd.set_option("display.width", 160)
tables = {n: pd.read_csv(RAW / f"{n}.csv") for n in ("customers", "products", "orders")}

for name, df in tables.items():
    print(f"\n=== {name}: shape={df.shape}")
    audit = pd.DataFrame({"dtype": df.dtypes.astype(str),
                          "missing": df.isna().sum(),
                          "missing_pct": (df.isna().mean() * 100).round(2)})
    print(audit.to_string())

o, c = tables["orders"], tables["customers"]
amt = pd.to_numeric(o["order_amount"], errors="coerce")
dm = pd.to_numeric(o["delivery_minutes"], errors="coerce")
age = pd.to_numeric(c["age"], errors="coerce")
fig = {
    "raw orders rows": len(o),
    "exact duplicate rows": int(o.duplicated().sum()),
    "rows with status pending": int((o["status"] == "pending").sum()),
    "distinct order_id": int(o["order_id"].nunique()),
    "non-numeric order_amount": int(amt.isna().sum()),
    "negative delivery_minutes": int((dm < 0).sum()),
    "blank delivery_minutes": int(dm.isna().sum()),
    "customers: blank age": int(age.isna().sum()),
    "customers: age outside 10-100": int(((age < 10) | (age > 100)).sum()),
    "customers: distinct raw city spellings": int(c["city"].nunique()),
    "customers: distinct non-blank gender spellings": int(c["gender"].dropna().nunique()),
    "customers: blank gender": int(c["gender"].isna().sum()),
}
print("\n=== audit figures")
for k, v in fig.items():
    print(f"{k:50s} {v:>8,}")
