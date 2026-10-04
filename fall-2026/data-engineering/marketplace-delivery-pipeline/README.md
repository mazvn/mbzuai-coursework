# Marketplace Delivery Pipeline

**Task:** Build a reproducible pipeline on a marketplace SQL extract that loads and cleans the data, defines a late-delivery label, analyzes what drives late deliveries and produces a leakage-free feature table. Measure the time, memory and storage of each step, and benchmark loading methods and file formats.

**What I did:** Items and payments are aggregated per order in SQLite before joining, so nothing is counted twice. Cleaning handles the planted problems (placeholder values, prices scaled by 1000, fake delivery dates) with every rule logged. Orders get a late label only when it can really be known (16,491 of 20,000). The analysis uses Wilson intervals, a logistic regression with state and month controls and Beta posteriors for small states, plus sensitivity scenarios. Step 4 builds 82 features known at approval time with a leakage check. Resource needs are estimated from 10/20/40% pilot runs and compared to the full run, and two fresh runs give identical outputs (SHA-256).

Main results: 6.7% of orders were late, each extra week in the delivery window lowers the odds of being late by about 36%, and Parquet/Feather were the fastest formats that also kept the data types intact. Charts are in `figures/`.

```
pip install -r requirements.txt
python3 run_pipeline.py
```

It needs `data/raw.sql` and `setup_db.py` from the course, which are not included. `--skip-estimates --skip-benchmarks` runs only the data steps.
