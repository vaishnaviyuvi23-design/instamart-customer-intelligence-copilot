"""Part A task 7 - pandas cross-check of Q2 and exploratory statistics on completed order_amount."""
import pandas as pd
from common import CLEAN

pd.set_option("display.width", 160)
o = pd.read_csv(CLEAN / "orders.csv"); c = pd.read_csv(CLEAN / "customers.csv"); p = pd.read_csv(CLEAN / "products.csv")
d = o[o["status"] == "completed"].merge(c[["customer_id", "city"]], on="customer_id").merge(p[["product_id", "category"]], on="product_id")

pt = pd.pivot_table(d, values="order_amount", index="city", columns="category", aggfunc="sum", margins=True, margins_name="Total")
print("(a) pivot grand total:", round(pt.loc["Total", "Total"], 2), "(SQL Q2 total: 2998668.80)")
print(pt.round(2).to_string())

by_city = d.groupby("city").agg(total=("order_amount", "sum"), mean=("order_amount", "mean"), count=("order_amount", "count"))
print("\nper-city completed order_amount:"); print(by_city.round(2).to_string())

x = d["order_amount"]
q1, q3 = x.quantile(.25), x.quantile(.75); iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print(f"\n(b) mean={x.mean():.2f} median={x.median():.2f} skew={x.skew():.2f} Q1={q1:.2f} Q3={q3:.2f}")
print(f"    fences: lower={lo:.2f} upper={hi:.2f}; outliers below={(x < lo).sum()} above={(x > hi).sum()}")
corr = d[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr()
print(corr.round(3).to_string())
cols = list(corr.columns)
pairs = {(a, b): corr.loc[a, b] for i, a in enumerate(cols) for b in cols[i + 1:]}
best = max(pairs, key=lambda k: abs(pairs[k]))
print("    largest |r| pair:", best, round(pairs[best], 3))
