"""Part B tasks 1-3 - RFM table, K-means selection, segment profiling/naming, PCA, outputs/customer_segments.csv."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from common import CLEAN, OUT

pd.set_option("display.width", 160)
SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
FINAL_K = 4

o = pd.read_csv(CLEAN / "orders.csv", parse_dates=["order_ts"])
d = o[o["status"] == "completed"]

# ---- task 1: RFM
g = d.groupby("customer_id")
rfm = pd.DataFrame({"recency": (SNAPSHOT - g["order_ts"].max()).dt.days,
                    "frequency": g["order_id"].nunique(),
                    "monetary": g["order_amount"].sum().round(2)}).reset_index()
print(f"RFM customers: {len(rfm)}")
print(f"recency median={rfm.recency.median()} min={rfm.recency.min()} max={rfm.recency.max()}")
print(f"frequency mean={rfm.frequency.mean():.2f} max={rfm.frequency.max()}")
print(f"monetary mean={rfm.monetary.mean():.2f} max={rfm.monetary.max():.2f}")

# ---- task 2: K-means sweep
X = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])
rows = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    rows.append({"K": k, "inertia": round(km.inertia_, 1), "silhouette": round(silhouette_score(X, km.labels_), 3)})
sweep = pd.DataFrame(rows)
sweep["inertia_drop_vs_prev"] = (-sweep["inertia"].diff()).round(1)
print("\nK sweep:"); print(sweep.to_string(index=False))

# ---- task 3: final model, profile, name
km = KMeans(n_clusters=FINAL_K, n_init=10, random_state=42).fit(X)
rfm["segment_id"] = km.labels_
prof = rfm.groupby("segment_id").agg(size=("customer_id", "count"), recency=("recency", "mean"),
                                     frequency=("frequency", "mean"), monetary=("monetary", "mean")).round(2)
print(f"\nSegment profile (K={FINAL_K}):"); print(prof.to_string())

# names are assigned from the profile itself, so they survive any permutation of the arbitrary cluster ids
names, left = {}, set(prof.index)
premium = prof["monetary"].idxmax(); names[premium] = "Premium High-Spenders"; left.discard(premium)
loyal = prof.loc[list(left), "frequency"].idxmax(); names[loyal] = "Loyal Frequent Buyers"; left.discard(loyal)
dormant = prof.loc[list(left), "recency"].idxmax(); names[dormant] = "Dormant At-Risk Buyers"; left.discard(dormant)
for s in left:
    names[s] = "Occasional Light Buyers"
rfm["segment_name"] = rfm["segment_id"].map(names)
rfm["is_premium"] = rfm["segment_id"] == premium
prof["name"] = prof.index.map(names)
print(prof[["name", "size"]].to_string())

pca = PCA(n_components=2).fit(X)
print("\nPCA explained_variance_ratio_:", pca.explained_variance_ratio_.round(3), "total", round(pca.explained_variance_ratio_.sum(), 3))

rfm[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]].to_csv(
    OUT / "customer_segments.csv", index=False)
print("premium segment id:", premium, "| premium customers:", int(rfm.is_premium.sum()))
