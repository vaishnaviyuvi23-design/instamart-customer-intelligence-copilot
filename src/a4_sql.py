"""Part A task 6 - run sql/queries.sql against data/instamart.db and report the stated results."""
import re
import sqlite3
import pandas as pd
from common import DB, ROOT

pd.set_option("display.width", 160)
text = (ROOT / "sql" / "queries.sql").read_text()
parts = re.split(r"^-- (Q\d):.*$", text, flags=re.M)
queries = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}
assert list(queries) == [f"Q{i}" for i in range(1, 8)]

with sqlite3.connect(DB) as con:
    r = {q: pd.read_sql_query(sql, con) for q, sql in queries.items()}

print("Q1 customers:", len(r["Q1"]))
print("Q2 rows:", len(r["Q2"]), "| sum:", round(r["Q2"]["revenue"].sum(), 2))
print("   largest cell:", r["Q2"].iloc[0].to_dict())
print("Q3 products above average price:", len(r["Q3"]))
print("Q4 customers without completed order:", len(r["Q4"]))
print("Q5 customers above average:", len(r["Q5"]), "| average spend:", r["Q5"]["avg_spend"].iloc[0])
q6 = r["Q6"]
print("Q6 top customer per city:")
print(q6[q6["row_num"] == 1][["city", "customer_id", "total_spend"]].to_string(index=False))
print("   rows where RANK() != ROW_NUMBER():", int((q6["rnk"] != q6["row_num"]).sum()))
print("Q7 daily rows:", len(r["Q7"]), "| last cumulative:", r["Q7"]["cumulative_revenue"].iloc[-1])
