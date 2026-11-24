
import pandas as pd
from pathlib import Path

csv_path = Path("results/quality_benchmark.csv")
df = pd.read_csv(csv_path)

methods = ["uniform", "adaptive_v1", "adaptive_v3", "self_adaptive_grid"]

subset = df[df["method"].isin(methods)]

pivot = subset.pivot(
    index="frame",
    columns="method",
    values=["mean_error_m", "p95_error_m", "neighborhood_recall"],
)

print("\nWorst self-adaptive-grid frames by P95 error:\n")
print(
    subset[subset["method"] == "self_adaptive_grid"]
    .sort_values("p95_error_m", ascending=False)
    [["frame", "mean_error_m", "p95_error_m",
      "neighborhood_recall", "runtime_s"]]
    .head(5)
    .to_string(index=False)
)

print("\nComparison for the worst frame:\n")
worst_frame = (
    subset[subset["method"] == "self_adaptive_grid"]
    .sort_values("p95_error_m", ascending=False)
    .iloc[0]["frame"]
)

print(
    subset[subset["frame"] == worst_frame]
    [["frame", "method", "mean_error_m", "p95_error_m",
      "neighborhood_recall", "runtime_s"]]
    .to_string(index=False)
)
