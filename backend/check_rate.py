import pandas as pd, glob
for f in glob.glob("backend/data/ALL_DATA/*/*.parquet")[:10]:
    gap = pd.read_parquet(f)["ts_device"].diff().dropna()   # milliseconds between readings
    print(f.split("ALL_DATA")[1], f"{1000/gap.mean():.1f} Hz", f"slowest 5% gap: {gap.quantile(.95):.0f} ms")