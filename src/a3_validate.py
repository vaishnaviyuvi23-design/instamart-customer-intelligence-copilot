"""Part A task 5 - assert the reference checkpoints on the clean CSVs and on the SQLite database."""
import sqlite3
import pandas as pd
from common import CLEAN, DB

c = pd.read_csv(CLEAN / "customers.csv")
o = pd.read_csv(CLEAN / "orders.csv")
done = o[o["status"] == "completed"]

checks = {
    "city counts": c["city"].value_counts().to_dict() == {"Pune": 227, "Delhi": 210, "Bengaluru": 197, "Chennai": 191, "Hyderabad": 191, "Mumbai": 184},
    "gender counts": c["gender"].value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30},
    "no missing ages": c["age"].isna().sum() == 0,
    "ages within 18-60": c["age"].between(18, 60).all(),
    "orders rows == 12309": len(o) == 12309,
    "order_id unique": o["order_id"].is_unique,
    "11110 completed": (o["status"] == "completed").sum() == 11110,
    "1199 cancelled": (o["status"] == "cancelled").sum() == 1199,
    "no pending": (o["status"] == "pending").sum() == 0,
    "no missing order_amount": o["order_amount"].isna().sum() == 0,
    "completed: no missing delivery": done["delivery_minutes"].isna().sum() == 0,
    "cancelled: 1199 missing delivery": o.loc[o["status"] == "cancelled", "delivery_minutes"].isna().sum() == 1199,
    "completed revenue 2,998,668.80": round(done["order_amount"].sum(), 2) == 2998668.80,
}
with sqlite3.connect(DB) as con:
    tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
    checks["db has 3 tables"] = {"customers", "products", "orders"} <= tables
    checks["db orders == 12309"] = con.execute("select count(*) from orders").fetchone()[0] == 12309
    checks["db revenue matches"] = round(con.execute(
        "select sum(order_amount) from orders where status='completed'").fetchone()[0], 2) == 2998668.80

for name, ok in checks.items():
    print(("PASS  " if ok else "FAIL  ") + name)
assert all(checks.values()), "some checkpoints failed"
print("ALL CHECKPOINTS PASSED")
