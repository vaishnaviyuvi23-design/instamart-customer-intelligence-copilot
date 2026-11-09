"""Part B tasks 4-8 - lapse label, leakage-free features, tuned LR/RF, class-balance experiment, interpretation, persistence."""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from common import CLEAN, OUT

pd.set_option("display.width", 160)
CUTOFF, WINDOW_END = pd.Timestamp("2025-09-01"), pd.Timestamp("2025-10-01")

o = pd.read_csv(CLEAN / "orders.csv", parse_dates=["order_ts"])
cust = pd.read_csv(CLEAN / "customers.csv", parse_dates=["signup_date"])
prod = pd.read_csv(CLEAN / "products.csv")
o = o.merge(prod[["product_id", "category"]], on="product_id")

# ------------------------------------------------------------------ task 4: label + features
hist = o[o["order_ts"] < CUTOFF]                    # ONLY orders before the cutoff feed the features
hist_done = hist[hist["status"] == "completed"]
g = hist_done.groupby("customer_id")
fav = hist_done.groupby(["customer_id", "category"]).size().reset_index(name="n").sort_values(
    ["customer_id", "n", "category"], ascending=[True, False, True]).drop_duplicates("customer_id").set_index("customer_id")["category"]
feat = pd.DataFrame({
    "recency": (CUTOFF - g["order_ts"].max()).dt.days,
    "frequency": g["order_id"].nunique(),
    "monetary": g["order_amount"].sum(),
    "avg_discount_pct": g["discount_pct"].mean(),
    "avg_delivery_minutes": g["delivery_minutes"].mean(),
    "fav_category": fav,
})
canc = hist.assign(c=(hist["status"] == "cancelled")).groupby("customer_id")["c"].mean()
feat["cancellation_rate"] = canc.reindex(feat.index).fillna(0.0)
feat = feat.reset_index().merge(cust, on="customer_id")
feat["tenure_days"] = (CUTOFF - feat["signup_date"]).dt.days

future = o[(o["order_ts"] >= CUTOFF) & (o["order_ts"] < WINDOW_END) & (o["status"] == "completed")]["customer_id"].unique()
feat["lapsed"] = (~feat["customer_id"].isin(future)).astype(int)

NUM = ["recency", "frequency", "monetary", "avg_discount_pct", "cancellation_rate", "avg_delivery_minutes", "age", "tenure_days"]
CAT = ["city", "gender", "fav_category"]
X, y, ids = feat[NUM + CAT], feat["lapsed"], feat["customer_id"]
assert not X.isna().any().any(), "feature matrix has missing values"
print(f"cohort={len(feat)} lapsed={int(y.sum())} ({y.mean()*100:.2f}%)  missing feature cells: {int(X.isna().sum().sum())}")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"train={len(X_tr)} test={len(X_te)}  train lapsed={int(y_tr.sum())}")

def metrics(y_true, y_pred):
    return dict(precision=precision_score(y_true, y_pred, zero_division=0), recall=recall_score(y_true, y_pred, zero_division=0),
                f1=f1_score(y_true, y_pred, zero_division=0))

base_pred = np.zeros(len(y_te), dtype=int)
print(f"baseline always-not-lapsed: accuracy={accuracy_score(y_te, base_pred):.3f} lapsed F1={f1_score(y_te, base_pred, zero_division=0):.3f}")

# ------------------------------------------------------------------ task 5: preprocessing (fitted on train rows only, inside a Pipeline)
def prep():
    return ColumnTransformer([("num", StandardScaler(), NUM),
                              ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), CAT)])
probe = prep().fit(X_tr)
learned = probe.named_transformers_["num"].mean_[NUM.index("frequency")]
print(f"scaler learned mean of frequency (train only)={learned:.4f} | full-cohort mean={feat['frequency'].mean():.4f}")
n_out = probe.transform(X_tr).shape[1]
print(f"design matrix width: {len(NUM)} numeric + {n_out-len(NUM)} one-hot (k-1) = {n_out}")

# ------------------------------------------------------------------ task 6: tune
lr_gs = GridSearchCV(Pipeline([("prep", prep()), ("clf", LogisticRegression(max_iter=1000))]),
                     {"clf__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1").fit(X_tr, y_tr)
rf_gs = GridSearchCV(Pipeline([("prep", prep()), ("clf", RandomForestClassifier(n_estimators=100, random_state=42))]),
                     {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 10, 20]}, cv=5, scoring="f1").fit(X_tr, y_tr)
print(f"\nLR best params={lr_gs.best_params_} cv F1={lr_gs.best_score_:.4f}")
print(f"RF best params={rf_gs.best_params_} cv F1={rf_gs.best_score_:.4f}")
res = {}
for name, gs in (("LogisticRegression", lr_gs), ("RandomForest", rf_gs)):
    pred = gs.predict(X_te)
    res[name] = metrics(y_te, pred)
    print(f"\n{name} test: " + ", ".join(f"{k}={v:.3f}" for k, v in res[name].items()))
    print("confusion matrix [[TN FP],[FN TP]]:", confusion_matrix(y_te, pred).tolist())

# selection is made on cross-validated F1 (training data only), not on the test set
selected_name = "RandomForest" if rf_gs.best_score_ >= lr_gs.best_score_ else "LogisticRegression"
selected = rf_gs if selected_name == "RandomForest" else lr_gs
print(f"\nselected by CV F1: {selected_name}; test F1={res[selected_name]['f1']:.3f}")
assert res[selected_name]["f1"] >= 0.55

# ------------------------------------------------------------------ task 7: class-balance experiment (training rows only)
lap_idx = y_tr[y_tr == 1].index
extra = pd.Series(lap_idx).sample(n=int((y_tr == 0).sum() - (y_tr == 1).sum()), replace=True, random_state=42).values
bal_idx = list(y_tr.index) + list(extra)
X_bal, y_bal = X.loc[bal_idx], y.loc[bal_idx]
print(f"\nbalanced train: {int((y_bal==0).sum())} not lapsed / {int((y_bal==1).sum())} lapsed (test untouched, n={len(y_te)})")
lr_best = lr_gs.best_estimator_
lr_bal = Pipeline([("prep", prep()), ("clf", LogisticRegression(max_iter=1000, C=lr_gs.best_params_["clf__C"]))]).fit(X_bal, y_bal)
m0, m1 = metrics(y_te, lr_best.predict(X_te)), metrics(y_te, lr_bal.predict(X_te))
print(pd.DataFrame({"unbalanced": m0, "oversampled": m1}).round(3).to_string())
print("confusion oversampled:", confusion_matrix(y_te, lr_bal.predict(X_te)).tolist())

# ------------------------------------------------------------------ task 8: interpret, persist, use
names = rf_gs.best_estimator_.named_steps["prep"].get_feature_names_out()
imp = pd.Series(rf_gs.best_estimator_.named_steps["clf"].feature_importances_, index=names).sort_values(ascending=False)
print("\nTop-5 RF importances:"); print(imp.head(5).round(4).to_string())
coef = pd.Series(lr_gs.best_estimator_.named_steps["clf"].coef_[0], index=lr_gs.best_estimator_.named_steps["prep"].get_feature_names_out())
print("\nLR coefficients (standardised features):")
print(coef[["num__recency", "num__frequency", "num__cancellation_rate", "num__monetary", "num__tenure_days"]].round(3).to_string())

joblib.dump(selected.best_estimator_, OUT / "lapse_model.joblib")
reloaded = joblib.load(OUT / "lapse_model.joblib")
same = np.array_equal(reloaded.predict(X_te), selected.best_estimator_.predict(X_te))
print("\nreloaded model reproduces test predictions exactly:", same); assert same

seg = pd.read_csv(OUT / "customer_segments.csv")
premium_ids = set(seg.loc[seg["is_premium"], "customer_id"])
te = pd.DataFrame({"customer_id": ids.loc[X_te.index].values,
                   "lapse_probability": reloaded.predict_proba(X_te)[:, 1].round(4),
                   "actually_lapsed": y_te.values})
te = te[te["customer_id"].isin(premium_ids)].sort_values("lapse_probability", ascending=False)
top = te.head(10)
top.to_csv(OUT / "retention_list.csv", index=False)
print(f"\npremium customers in test set: {len(te)}; retention list rows: {len(top)}; truly lapsed in list: {int(top.actually_lapsed.sum())}")
