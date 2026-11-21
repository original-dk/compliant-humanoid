"""Generate a 30 Hz synthetic symmetric knee-bend clip in G1 CSV layout."""
import csv
import math
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "data/demo/knee_bend.csv"
with path.open("w", newline="") as f:
    writer = csv.writer(f)
    for i in range(121):
        angle = 0.12 + 0.08 * (1 - math.cos(math.pi * i / 60))
        leg = [-angle, 0, 0, 2 * angle, -angle, 0]
        row = [0, 0, 0.80 - 0.6 * (1-math.cos(angle)), 0, 0, 0, 1]
        writer.writerow(row + leg * 2 + [0] * 17)
