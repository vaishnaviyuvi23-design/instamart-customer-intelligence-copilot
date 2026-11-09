# Instamart Customer Intelligence & Support Copilot

## Overview

One repository, four connected Parts, all on synthetic and fictional data (the brand name only frames the scenario). All money is in INR.

| Part | What it does | Main scripts |
|---|---|---|
| A | Audit and clean the raw exports, store them in SQLite, answer business questions in SQL and pandas | `src/a1_audit.py` … `src/a5_pandas_eda.py`, `sql/queries.sql` |
| B | RFM segments (K-means) and a lapse-prediction model | `src/b1_segments.py`, `src/b2_lapse.py` |
| C | Support-ticket triage with privacy, retry, validation and budget guardrails | `src/c1_triage.py` |
| D | Policy RAG assistant and a tool-using refund agent with safety limits and crash-resume | `src/rag.py`, `src/agent.py`, `src/d1_rag_eval.py`, `src/d2_agent_demo.py` |

Each Part consumes the previous one: Part B reads `data/clean/*.csv`, Part C reads `outputs/customer_segments.csv`, Part D's agent reads `data/instamart.db` and reuses the Part D retriever. `generate_data.py` and `mock_llm.py` are the supplied files, unmodified. No network, API key or paid service is used anywhere; every model call goes through `mock_llm.py`. Libraries: Python standard library, pandas, numpy, scikit-learn, joblib. Built and tested with Python 3.12 (3.11+ required) on pinned versions in `requirements.txt`.

## Reproduce

Run every command from the repository root. Each script changes into the repo root itself, so the order below is the only thing that matters.

1. `python -m venv .venv`
2. `source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. `python generate_data.py` (last line: `generated: customers=1200 products=40 orders_raw=13084 tickets=164`)
5. `python src/a1_audit.py` (Part A audit figures)
6. `python src/a2_clean.py` (cleans, writes `data/clean/*.csv` and `data/instamart.db`)
7. `python src/a3_validate.py` (prints `ALL CHECKPOINTS PASSED`)
8. `python src/a4_sql.py` (runs the seven queries in `sql/queries.sql`)
9. `python src/a5_pandas_eda.py` (pivot cross-check and exploratory statistics)
10. `python src/b1_segments.py` (writes `outputs/customer_segments.csv`)
11. `python src/b2_lapse.py` (writes `outputs/lapse_model.joblib`, `outputs/retention_list.csv`)
12. `python src/c1_triage.py` (writes `outputs/triage_results.csv`, `outputs/triage_budget_demo.csv`, `outputs/triage_retry_log.txt`)
13. `python src/d1_rag_eval.py` (chunking, retrieval and grounding evaluation)
14. `python src/d2_agent_demo.py` (writes `outputs/agent_runs.jsonl`, `outputs/refund_ledger.jsonl`, `state/*.json`)

Steps 12 and 14 sleep briefly (retry back-off) and take a few seconds. Random seeds: generator 42, K-means/split/forest/oversampling 42, retry jitter `random.seed(42)`.

## Part A

### Audit of the raw data

| Table | Shape | Columns with missing values (pandas default parsing) |
|---|---|---|
| customers.csv | 1200 × 5 | age 61 (5.08%), gender 30 (2.50%); others 0 |
| products.csv | 40 × 4 | none |
| orders.csv | 13084 × 9 | order_amount 53 (0.41%), delivery_minutes 2541 (19.42%); others 0 |

Dtypes: ids, dates, city, gender, status are text; `age`, `delivery_minutes` float; `quantity`, `discount_pct`, `price_inr` int; `order_amount` is text because it contains unparseable entries (pandas treats the literal `N/A` as missing, which is why only 53 of the 120 bad amounts show as missing here; the other 67 are the word `error`).

| Audit figure | Value |
|---|---|
| raw orders rows | 13,084 |
| exact duplicate rows | 119 |
| rows with status pending | 656 |
| distinct order_id | 12,309 |
| non-numeric order_amount | 120 |
| negative delivery_minutes | 106 |
| blank delivery_minutes | 2,541 |
| customers: blank age | 61 |
| customers: age outside 10–100 | 6 |
| customers: distinct raw city spellings | 24 |
| customers: distinct non-blank gender spellings | 10 |
| customers: blank gender | 30 |

### Cleaning rules and what each changed

Customers: (a) `city` stripped and title-cased, 476 values changed, exactly six cities remain (Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184). (b) gender mapped to male/female/unknown (blank → unknown): female 610, male 560, unknown 30. (c) 6 impossible ages set to missing, then all 67 missing ages (61 blank + 6) filled with the city median via `groupby().transform("median")`; 0 missing, range 18–60. (d) `signup_date` converted to datetime.

Orders, in this order:

| # | Rule | Rows / values changed |
|---|---|---|
| 1 | `order_ts` converted to datetime | all rows |
| 2 | exact duplicate rows removed | 119 rows |
| 3 | per `order_id` keep the record with the latest `order_ts` (drops the earlier pending record) | 656 rows (all the pending ones) |
| 4 | `order_amount` converted to numeric, unparseable → missing → `round(price × quantity × (1 − discount/100), 2)` | 112 values (the other 8 of the 120 sat on rows removed by rules 2–3) |
| 5 | negative `delivery_minutes` → missing (105 left after rules 2–3); completed orders with missing delivery filled with the median of that product category among completed orders (`groupby().transform`); cancelled orders left missing | 774 completed-order values imputed (105 negative + 669 blank); 1,199 cancelled left missing |

Result: 12,309 rows (one per order_id); 11,110 completed, 1,199 cancelled, 0 pending; 0 missing `order_amount`; 0 missing delivery among completed, 1,199 among cancelled; completed `order_amount` sums to INR 2,998,668.80. 13,084 − 119 − 656 = 12,309.

### Store and validate

`src/a2_clean.py` writes `data/clean/customers.csv`, `products.csv`, `orders.csv` and the SQLite database `data/instamart.db` (tables `customers`, `products`, `orders`; `order_ts` stored as text `YYYY-MM-DD HH:MM:SS`). `src/a3_validate.py` asserts 16 checkpoints (city and gender counts, no missing ages, age range, 12,309 unique orders, 11,110/1,199 statuses, no pending, no missing amounts, delivery missingness for both statuses, the INR 2,998,668.80 total, the three DB tables, DB row count and DB revenue) and prints `ALL CHECKPOINTS PASSED`.

### SQL analysis (`sql/queries.sql`, run with `pd.read_sql_query`)

| Query | Technique | Result |
|---|---|---|
| Q1 | WHERE then HAVING | 30 customers |
| Q2 | three-table join | 30 rows; sum INR 2,998,668.80; largest cell Bengaluru / Electronics INR 233,909.45 |
| Q3 | scalar subquery | 11 products above the average price |
| Q4 | LEFT JOIN | 25 customers without a completed order |
| Q5 | two CTEs | 550 customers above the average total of INR 2,552.06 |
| Q6 | RANK / DENSE_RANK / ROW_NUMBER per city | 8 rows with RANK ≠ ROW_NUMBER; top per city: Bengaluru C0082 (12,856.20), Chennai C0930 (10,151.40), Delhi C0291 (13,742.60), Hyderabad C0743 (12,730.05), Mumbai C0693 (14,766.55), Pune C0426 (10,377.10) |
| Q7 | running total | 183 daily rows; last cumulative INR 2,998,668.80 |

Q6 ties: when two customers in a city have exactly the same total spend, `RANK()` gives them the same number (and skips the next), while `ROW_NUMBER()` still numbers them 1, 2, 3…, so every row after a tie differs between the two.

### pandas cross-check and exploratory analysis

`pd.pivot_table(..., margins=True)` of completed order_amount by city × category has a grand total of INR 2,998,668.80, equal to the SQL total. Per city (completed): 

| City | Sum (INR) | Mean (INR) | Count |
|---|---|---|---|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

Completed `order_amount`: mean 269.91, median 135.00, skewness 4.61, Q1 64.80, Q3 297.00, fences −283.50 / 645.30; 0 outliers below, 897 above. Correlations over completed orders: order_amount–discount_pct −0.095, order_amount–quantity +0.087, order_amount–delivery_minutes −0.003, discount_pct–quantity 0.000, discount_pct–delivery −0.010, quantity–delivery −0.013.

**What is this data about?** Six months (April–September 2025) of quick-commerce orders from 1,200 customers in six cities, across 40 products in five categories. **Distributions?** Order value is strongly right-skewed (mean 269.91 well above median 135.00, skewness 4.61): most baskets are small, a few are large. **Outliers?** 897 orders exceed the upper fence of 645.30 (mostly electronics), none fall below; they are genuine high-value orders, not errors. **Relationships?** All correlations are weak; the largest in absolute value is order_amount with discount_pct (−0.095), just ahead of order_amount with quantity (+0.087). Correlation this small does not establish causation: bigger discounts do not demonstrably shrink baskets, and the sign may simply reflect which products carry discounts.

## Part B

### RFM (snapshot 2025-10-01 00:00:00, completed orders only)

1,175 customers. Recency: median 19, min 0, max 180 days. Frequency: mean 9.46, max 33. Monetary: mean INR 2,552.06, max INR 14,766.55.

### K-means (StandardScaler, `KMeans(n_clusters=K, n_init=10, random_state=42)`)

| K | Inertia | Silhouette | Inertia drop vs K−1 |
|---|---|---|---|
| 2 | 1820.8 | 0.465 | – |
| 3 | 1268.2 | 0.445 | 552.6 |
| 4 | 743.1 | 0.526 | 525.1 |
| 5 | 617.4 | 0.480 | 125.7 |
| 6 | 503.2 | 0.482 | 114.2 |
| 7 | 408.5 | 0.416 | 94.7 |
| 8 | 366.5 | 0.412 | 42.0 |

**Chosen K = 4.** The elbow is at K=4: the inertia falls by about 525 from K=3 to K=4 but only by 126 from K=4 to K=5, and silhouette peaks at 0.526 for K=4. Four segments are also a number a retention team can staff campaigns for, each with a distinct behaviour (below). The inertia of K=8 is always lower than that of K=2 because more centroids can only bring every point closer to some centroid (with K = number of customers the inertia is zero), so inertia falls monotonically and cannot pick K alone; it has to be read together with the elbow, silhouette and interpretability.

### Segment profile (K = 4)

Cluster numbers are only labels with no ordering (re-running with another seed can permute them); names and the Premium flag below are assigned from the profile, not from the number.

| segment_id | Name | Size | Mean recency (days) | Mean frequency | Mean monetary (INR) | Recommended action |
|---|---|---|---|---|---|---|
| 1 | **Premium High-Spenders** (is_premium = true) | 138 | 24.12 | 6.77 | 7,129.70 | Early access to electronics launches, concierge support, protect with the retention list |
| 2 | Loyal Frequent Buyers | 419 | 8.77 | 19.40 | 3,410.02 | Subscription and bundle offers on groceries and dairy; keep delivery fast |
| 0 | Occasional Light Buyers | 436 | 28.86 | 3.77 | 1,057.62 | Basket-builder nudges and free-delivery thresholds to raise order frequency |
| 3 | Dormant At-Risk Buyers | 182 | 109.04 | 2.20 | 685.99 | Win-back voucher; low priority for expensive offers |

PCA(n_components=2) on the scaled RFM matrix: explained_variance_ratio_ = 0.634 and 0.218, total 0.852. `outputs/customer_segments.csv` holds `customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium` (138 premium customers, exactly the highest-monetary segment).

### Lapse problem

Cutoff 2025-09-01. Cohort: 1,156 customers with at least one completed order before the cutoff. `lapsed = 1` if there is no completed order from 2025-09-01 through 2025-09-30: 443 lapsed (38.32%). Features (only orders dated before the cutoff plus the customer table): recency, frequency, monetary, average discount_pct, cancellation rate, average delivery_minutes, favourite category, city, age, gender, tenure_days. Split `train_test_split(test_size=0.2, stratify=y, random_state=42)`: 924 train, 232 test. **No data dated 2025-09-01 or later entered any feature, and the Part B segment labels were not used as features.** Baseline always "not lapsed": accuracy 61.6%, lapsed-class F1 0.000.

### Preprocessing without leakage

A scikit-learn `Pipeline` holds a `ColumnTransformer` (`StandardScaler` on 8 numeric columns; `OneHotEncoder(drop="first")` on city, gender and favourite category → 11 k−1 columns; 19 columns in total). **No encoder, scaler or resampler saw test rows**: all are fitted inside the pipeline on the training rows only and merely applied to the test rows. The learned scaler mean of `frequency` is 8.1147 (training rows) against 8.0285 for the full cohort; they differ because the scaler sees only the 924 training customers, a different sample from the 1,156 cohort, and using the full-cohort value would leak test information into training. Assertion in code: the feature matrix has 0 missing cells.

### Tuning and test evaluation (lapsed class = positive)

| Model | Best parameters | Best CV F1 | Test precision | Test recall | Test F1 | Confusion [[TN, FP], [FN, TP]] |
|---|---|---|---|---|---|---|
| LogisticRegression, C ∈ {0.01, 0.1, 1, 10, 100}, cv=5 | C = 100 | 0.651 | 0.644 | 0.652 | 0.648 | [[111, 32], [31, 58]] |
| RandomForest (100 trees), max_depth ∈ {3, 5, 8} × min_samples_leaf ∈ {5, 10, 20}, cv=5 (45 fits) | depth 8, leaf 10 | 0.681 | 0.659 | 0.652 | 0.655 | [[113, 30], [31, 58]] |

**Selected model: the random forest** (chosen on cross-validated F1, which uses training data only; its test F1 is 0.655, above the 0.55 requirement and far above the baseline F1 of 0). Missing a customer who is about to lapse (a false negative) loses all their future orders, while wrongly targeting a loyal customer (a false positive) costs only one voucher, so recall matters at least as much as precision, and F1 is a reasonable balance. The forest has the same recall as the logistic regression (0.652), slightly better precision (0.659 versus 0.644) and the better CV score. The two models are close on this data, so the choice is not decisive; the forest also captures non-linear effects of frequency and recency without manual feature work.

### Class-balance experiment

Over-sampling the training set only (354 lapsed rows duplicated with replacement, `random_state=42`, to 570 vs 570) and refitting the best logistic regression (C = 100), evaluated on the unchanged test set:

| Logistic regression | Precision | Recall | F1 | Confusion |
|---|---|---|---|---|
| un-balanced | 0.644 | 0.652 | 0.648 | [[111, 32], [31, 58]] |
| over-sampled | 0.589 | 0.854 | 0.697 | [[90, 53], [13, 76]] |

**Recall moved most** (+0.20), precision fell (−0.055) and did not rise. For the retention team this means over-sampling catches many more lapsing customers (76 instead of 58 of 89) but flags 21 more loyal customers by mistake (53 false alarms instead of 32): worth it if a voucher is cheap relative to a lost customer, wasteful if the offer budget is tight.

### Interpretation, persistence and retention list

Top-5 random-forest feature importances: frequency 0.280, monetary 0.271, recency 0.115, avg_delivery_minutes 0.061, cancellation_rate 0.052. Business reading: how often and how much a customer has bought dominate lapse risk; slow delivery and cancellations are minor secondary signals.

Logistic-regression coefficients (standardised features): `frequency` −1.698 (more past orders → less likely to lapse), `recency` +0.040 (a longer gap since the last order nudges lapse risk up, but the effect is negligible once frequency is known), `cancellation_rate` −0.124 (weak and counter-intuitive; it is correlated with frequency, so do not read it causally).

The selected pipeline is saved with `joblib.dump` to `outputs/lapse_model.joblib`; reloading with `joblib.load` reproduces the original test predictions exactly (checked with `np.array_equal`). `outputs/retention_list.csv` (`customer_id, lapse_probability, actually_lapsed`) lists the 10 highest-probability customers among the 40 Premium-segment customers in the test set; **2 of the 10 truly lapsed**. This list is illustrative and not a leakage-free evaluation: the Premium flag comes from the whole-window RFM of task 3, which includes the September window that defines the lapse label.

## Part C

Pipeline in `src/c1_triage.py`, processing tickets in file order.

**Guardrails.** Empty/whitespace tickets → `skipped_empty`; `est_tokens(text) > 400` → `flagged_long`; neither ever reaches the model. Found: 4 empty, 4 long (each estimates to 734 tokens).

**Redaction** (email, then phone, then order id, using exactly the specified patterns): 55 emails → `[EMAIL]`, 91 phones → `[PHONE]`, 132 order ids → `[ORDER_ID]`; 278 in total; re-applying the three patterns to the redacted texts finds 0. The user message is exactly the redacted text, and every call is guarded by an assertion that no message (system or user) matches any of the three patterns.

**Prompts.** Zero-shot (32 est. tokens) and few-shot (125 est. tokens, `EXAMPLES:` line followed by four labelled examples including a fraud one); both contain `CLASSIFY`, list the four categories and demand a JSON-only reply.

**Retries.** Every call goes through a retry wrapper: at most 3 retries, wait `0.05 × 2**n × random.uniform(0.9, 1.15)`, one log line per retry. `mock_llm.reset()` precedes every full run. A full run over the 156 eligible tickets had 19 `TransientLLMError`s, 19 retry log lines (all retry number 0, waits 0.0454 to 0.0574 s, inside 0.045–0.0575), all recovered; no ticket ended with a permanent error (`outputs/triage_retry_log.txt` holds the lines of the final run).

**Validation.** JSON with an allowed category → `classified_json`; non-JSON → lowercase words via `re.findall(r"[a-z]+", ...)`, exactly one allowed word → `classified_keyword`; everything else (including JSON with a category such as `shipping`) → `flagged_invalid_output`, never guessed. Both full runs: 135 `classified_json`, 12 `classified_keyword`, 9 `flagged_invalid_output` (plus 4 `skipped_empty`, 4 `flagged_long`).

### Zero-shot versus few-shot (no budget)

| Run | Classified | Correct | Accuracy | Total tokens | Tokens per classified ticket |
|---|---|---|---|---|---|
| zero-shot | 147 | 126 | 85.7% | 9,400 | 63.9 |
| few-shot | 147 | 147 | 100.0% | 23,908 | 162.6 |

Zero-shot confusion (rows = true, columns = predicted):

| true \ predicted | billing | delivery | fraud | technical |
|---|---|---|---|---|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| fraud | 11 | 0 | 23 | 0 |
| technical | 10 | 0 | 0 | 29 |

The few-shot prompt costs about 99 more tokens per classified ticket (162.6 versus 63.9, 14,508 extra tokens in total) because its 93 extra prompt tokens are re-sent with every call. It fixed all 21 errors, which is about 690 extra tokens per corrected ticket. It is worth paying for here because the zero-shot errors are not random: 11 fraud tickets were misrouted to billing (and 10 technical ones too), and a fraud ticket has a 2-hour security SLA, so a wrong label has a real cost. For a low-stakes category with few errors, the zero-shot prompt at one third of the cost would be enough.

### Budget, routing and outputs

`TOKEN_BUDGET` check before each call (no call is made once the running total ≥ the budget). Zero-shot run with `TOKEN_BUDGET = 6000` → `outputs/triage_budget_demo.csv`: **56 tickets `budget_exhausted`** (88 `classified_json`, 7 `classified_keyword`, 5 `flagged_invalid_output`, 4 empty, 4 long; 6,038 tokens spent).

`outputs/triage_results.csv` (164 rows; few-shot run, `TOKEN_BUDGET = 100000`; columns `ticket_id, customer_id, status, category, segment_name, priority, route, tokens_used`). Routing: skipped/flagged/budget-exhausted → `human_review`, NORMAL; else fraud → `security_team`, HIGH; else premium (from `outputs/customer_segments.csv`, missing customer = not premium) → `premium_desk`, HIGH; else `support_queue`, NORMAL.

| Route | Priority | Tickets |
|---|---|---|
| security_team | HIGH | 34 |
| premium_desk | HIGH | 11 |
| support_queue | NORMAL | 102 |
| human_review | NORMAL | 17 |

**What was rule-validated and what went to humans.** 147 tickets (135 JSON + 12 keyword) passed the output rules and were routed automatically. 17 went to human review: 9 because the model's output was invalid, 4 empty and 4 over-long tickets that were never sent to the model.

## Part D

### Chunking and retrieval

40-word chunks with 10-word overlap (step 30; stop after the first chunk whose end reaches the last word): **15 chunks**: refund 4, delivery 3, escalation 3, payment 3, privacy 2. `TfidfVectorizer()` is fitted on the chunk texts, queries are transformed with the same vectoriser and ranked by cosine similarity (`retrieve(query, k=3)` returns `source, score, text`). For `refund policy late delivery` the top three chunks all come from `refund_policy.txt`.

TF (term frequency) measures how often a word appears in a chunk; IDF (inverse document frequency) down-weights words that appear in many chunks, so rare, informative words count more. A limitation: TF-IDF matches words, not meaning, so a question phrased with different words than the policy (for instance "money back" against "refund") scores poorly.

`answer(question)` retrieves three chunks and sends two messages to `mock_llm.chat`: a system message containing `POLICY ASSISTANT` (answer only from context, otherwise say it could not find the information) and a user message `CONTEXT:`, newline, the three chunks each prefixed with `[<source file>]` on its own line, newline, `QUESTION:`, newline, the question. It returns the text and the sources.

### Evaluation (`data/raw/policy_eval.csv`, 12 answerable + 2 not covered)

| Setting | Chunks | Answerable: expected doc in top 3 | Answerable: expected fact in answer | Unanswerable: fallback "I could not find…" |
|---|---|---|---|---|
| overlap 10 | 15 | 12 / 12 | 12 / 12 | 2 / 2 |
| overlap 0 | 13 | 12 / 12 | 10 / 12 | 2 / 2 |

With zero overlap two questions break because the answer sentence is split across a chunk boundary. "What delivery time makes a completed order eligible for a full refund?" is answered with "receives a full refund of the order amount." and misses "more than 45 minutes", which ended in the previous chunk. "How soon is money returned when a payment fails but is deducted?" is answered with "…the amount is returned automatically" and misses "within 48 hours", which fell into the next chunk. The correct file is still retrieved, but the fact is cut off.

### Tools

`TOOLS_SPEC` (name, one-sentence description, JSON-style parameters) and `TOOL_MAP` in `src/agent.py`: `get_order` (reads `data/instamart.db`; `{"error": "order not found"}` if missing), `search_policy` (the Part D retriever, key `results`), `calculate_refund` (delivered in more than 45 minutes → 100% of the order amount, else 0), `issue_refund` and `escalate_to_human` (each appends a timestamped JSON line with tool name, run id, order id and amount to `outputs/refund_ledger.jsonl`). The loop looks the requested name up in `TOOL_MAP` and calls it with the model's arguments. A tool that raises becomes an `{"error": ...}` observation instead of crashing the run.

### ReAct loop results (`run_agent`, defaults: 6 iterations, 6000 tokens, 10 s)

| Scenario | Status | Refund (INR) | Model calls | Tokens used |
|---|---|---|---|---|
| S1 | REFUND | 128.25 | 5 | 1,112 |
| S2 | REFUND | 135.00 | 5 | 1,112 |
| S3 | ESCALATE | 1,999.00 | 5 | 1,099 |
| S4 | DENIED | 0.00 | 4 | 777 |
| S5 | DENIED | 0.00 | 3 | 466 |
| S6 | NOT_FOUND | 0.00 | 2 | 158 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,258 |

Every refund equals `expected_refund_inr` within 0.01 (asserted in `src/d2_agent_demo.py`). S7 (a `[LOOP]` request that keeps calling `search_policy`) uses all six model calls and 3,258 estimated tokens, so the default 6000-token limit does not stop it early. Each run appends one line to `outputs/agent_runs.jsonl` (scenario id, status, final answer, calls, tokens, every step's tool, arguments and observation); the file also holds the lines of the demos below, marked with a `demo` field.

### Stop safely

| Demonstration | Result |
|---|---|
| S1 with `token_limit=200` | `FALLBACK_TOKEN_LIMIT` after 3 model calls (483 tokens); the tool requested by the third call was not executed; 2 steps returned |
| S1 with `time_limit_seconds=0` | `FALLBACK_TIMEOUT` before the first model call (0 calls, 0 steps) |

The iteration limit gives `FALLBACK_MAX_ITERATIONS` for S7. Every fallback returns the steps completed so far and a human-handoff summary in `final_answer`.

### Crash and resume

After every tool observation the loop writes `state/<run_id>.json` (run id, scenario id, full message list, tokens used, step number, tools executed so far, plus the steps and limits needed to continue). For S1 a simulated crash is raised right after the second tool execution (checkpoint already saved); `resume_agent("S1-crash")` reloads the checkpoint and continues. Tools executed across both parts, in order, four executions and none repeated:

1. `get_order`
2. `search_policy` (crash raised here, after the checkpoint was saved)
3. `calculate_refund`
4. `issue_refund`

Final status REFUND with INR 128.25, 5 model calls in total. The time limit restarts on resume; call and token counters carry over. Because the S1 refund is also issued by the main S1 run, `outputs/refund_ledger.jsonl` contains that simulated refund twice (tagged with different `run_id`s, `S1-main` and `S1-crash`); the ledger has 4 lines in total (S1, S2, S1 replay as refunds, S3 as an escalation).

### ReAct versus plan-and-execute

My ReAct agent makes one model call per step: 5 for a refund (S1), 5 for an escalation (S3), 4 for an on-time denial (S4), 3 for a cancelled order (S5) and 2 for an unknown order (S6), with 1,112, 1,099, 777, 466 and 158 tokens. Because the whole message history is re-sent on every call, tokens grow faster than the call count. A plan-and-execute design asks the model once for the whole list (get_order, search_policy, calculate_refund, issue or escalate), so a refund costs about 2 calls (plan and final answer) and far fewer tokens, since the history is not re-sent four times. Its weakness is the unexpected observation: with S6 (NOT_FOUND) or S5 (cancelled) the fixed plan would carry on to a refund calculation on missing data, so it needs an explicit replan call whenever an observation contradicts the plan, which is extra logic to write and test. ReAct handles these cases for free, because it reads each observation before choosing the next step: it stopped after 2 calls for S6 and 3 for S5. For this refund workflow I would choose ReAct, with the limits and checkpoints above. The workflow is short (at most 5 calls), branches on data (status, delivery time, the INR 1000 threshold), and money is involved, so deciding after each observation is safer than committing to a plan. If volume grew and token cost mattered, I would move to plan-and-execute with a replan step on any error observation.

## Limitations and Next Steps

- **All data is synthetic**, generated by a fixed script; archetypes (premium/frequent/rare) make segments cleaner than real customers would be, and the effects (for example the 38% lapse rate) are properties of the generator.
- **Premium flag and the retention list.** The Premium flag comes from the whole-window RFM of task 3, which includes the September window that defines the lapse label, so it is a post-hoc whole-window label: the retention list is illustrative and not a leakage-free evaluation. The model's own features use only orders before 2025-09-01. A proper evaluation would compute segments at the cutoff.
- **Lapse model:** one split of 1,156 customers (232 test rows) gives noisy test F1 (a handful of customers moves it by a point); the logistic regression and forest are statistically close. Next step: repeated cross-validation, probability calibration and a cost-based threshold instead of the default 0.5.
- **Imputation:** delivery times and ages are imputed with medians, which shrinks variance; imputed values are not flagged in the clean tables.
- **Part C:** the mock classifier is keyword-based, so the 100% few-shot accuracy says more about the mock than about real LLMs. The redaction regexes cover only Indian 10-digit mobile numbers, plain emails and `ORD-` ids; names and addresses are not redacted. The jitter wait is real `time.sleep`, so runs take a few seconds.
- **Part D:** TF-IDF cannot match synonyms (embeddings would), the retrieval index holds only five short documents, the refund window (7 days) is assumed satisfied, and the refund ledger is not idempotent across separate runs (a real system would reject a second refund for the same order). Resumed runs restart the wall-clock limit.
- **Reproducibility note:** results were produced with the versions pinned in `requirements.txt` (pandas 3.x, scikit-learn 1.9); scikit-learn estimator outputs (K-means, forests) can differ slightly on other versions.
