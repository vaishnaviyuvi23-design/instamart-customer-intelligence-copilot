"""Part A tasks 3-5 - clean customers and orders, write data/clean/*.csv and data/instamart.db."""
import sqlite3
import pandas as pd
from common import RAW, CLEAN, DB

# ---------------------------------------------------------------- customers
cust = pd.read_csv(RAW / "customers.csv")
prod = pd.read_csv(RAW / "products.csv")

n_city_changed = int((cust["city"] != cust["city"].str.strip().str.title()).sum())
cust["city"] = cust["city"].str.strip().str.title()                                   # (a)

g = cust["gender"].fillna("").str.strip().str.lower()                                   # (b)
cust["gender"] = g.map({"m": "male", "male": "male", "f": "female", "female": "female"}).fillna("unknown")

cust["age"] = pd.to_numeric(cust["age"], errors="coerce")                               # (c)
bad_age = (cust["age"] < 10) | (cust["age"] > 100)
n_bad_age = int(bad_age.sum())
cust.loc[bad_age, "age"] = float("nan")
n_missing_age = int(cust["age"].isna().sum())
cust["age"] = cust["age"].fillna(cust.groupby("city")["age"].transform("median"))
cust["signup_date"] = pd.to_datetime(cust["signup_date"])                               # (d)

print("CUSTOMERS")
print(f"  city values changed by strip/title-case: {n_city_changed}")
print(f"  impossible ages set to missing: {n_bad_age}; ages imputed in total: {n_missing_age}")
print("  cities:", cust["city"].value_counts().to_dict())
print("  genders:", cust["gender"].value_counts().to_dict())
print(f"  missing ages: {int(cust['age'].isna().sum())}; age range {cust['age'].min()}-{cust['age'].max()}")

# ---------------------------------------------------------------- orders
o = pd.read_csv(RAW / "orders.csv", dtype={"order_amount": str})
n0 = len(o)
o["order_ts"] = pd.to_datetime(o["order_ts"])                                           # rule 1

n = len(o); o = o.drop_duplicates(); r2 = n - len(o)                                    # rule 2
n = len(o); o = o.sort_values(["order_id", "order_ts"]).drop_duplicates("order_id", keep="last"); r3 = n - len(o)  # rule 3
o = o.reset_index(drop=True)

amt = pd.to_numeric(o["order_amount"], errors="coerce")                                 # rule 4
r4 = int(amt.isna().sum())
o["order_amount"] = amt
price = o["product_id"].map(prod.set_index("product_id")["price_inr"])
calc = (price * o["quantity"] * (1 - o["discount_pct"] / 100)).round(2)
o["order_amount"] = o["order_amount"].fillna(calc)

o["delivery_minutes"] = pd.to_numeric(o["delivery_minutes"], errors="coerce")           # rule 5
neg = o["delivery_minutes"] < 0
r5a = int(neg.sum())
o.loc[neg, "delivery_minutes"] = float("nan")
cat = o["product_id"].map(prod.set_index("product_id")["category"])
done = o["status"] == "completed"
med = o["delivery_minutes"].where(done).groupby(cat).transform("median")               # completed-only median per category
need = done & o["delivery_minutes"].isna()
r5b = int(need.sum())
o.loc[need, "delivery_minutes"] = med[need]
o = o.sort_values("order_ts").reset_index(drop=True)

print("\nORDERS")
print(f"  raw rows {n0}; rule2 exact duplicates removed: {r2}; rule3 earlier records removed (pending): {r3}")
print(f"  rule4 unparseable amounts refilled from price: {r4}")
print(f"  rule5 negative delivery set missing: {r5a}; completed-order delivery imputed (blank+negative): {r5b}")
print(f"  rows={len(o)}; status counts: {o['status'].value_counts().to_dict()}")
done = o["status"] == "completed"          # recompute: o was re-sorted, so the old mask is stale
c = o[done]
print(f"  missing order_amount: {int(o['order_amount'].isna().sum())}")
print(f"  missing delivery: completed={int(c['delivery_minutes'].isna().sum())}, "
      f"cancelled={int(o.loc[~done, 'delivery_minutes'].isna().sum())}")
print(f"  completed revenue: INR {c['order_amount'].sum():,.2f}")

# ---------------------------------------------------------------- store
cust.to_csv(CLEAN / "customers.csv", index=False, date_format="%Y-%m-%d")
prod.to_csv(CLEAN / "products.csv", index=False)
o.to_csv(CLEAN / "orders.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")

db_c, db_o = cust.copy(), o.copy()
db_c["signup_date"] = db_c["signup_date"].dt.strftime("%Y-%m-%d")
db_o["order_ts"] = db_o["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
if DB.exists():
    DB.unlink()
with sqlite3.connect(DB) as con:
    db_c.to_sql("customers", con, index=False)
    prod.to_sql("products", con, index=False)
    db_o.to_sql("orders", con, index=False)
print("\nwrote data/clean/*.csv and data/instamart.db")
