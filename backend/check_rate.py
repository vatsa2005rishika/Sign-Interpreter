import pandas as pd, glob
for f in glob.glob("backend/data/ALL_DATA/*/*.parquet"):
    ts = pd.to_datetime(pd.read_parquet(f)["ts_device"], errors='coerce')
    gap = ts.diff().dt.total_seconds().dropna() * 1000   # milliseconds between readings
    mean_gap = gap.mean()
    freq_str = f"{1000/mean_gap:.1f} Hz" if mean_gap > 0 else "N/A"
    q95 = gap.quantile(.95) if not gap.empty else 0
    print(f.split("ALL_DATA")[1], freq_str, f"slowest 5% gap: {q95:.0f} ms")