import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

GAPS = ["00", "05", "10", "15", "20", "25", "30"]
gaps, errs, idsw = [], [], []
for g in GAPS:
    m = json.loads(Path(f"results/gaps/gap_{g}/metrics.json").read_text())
    gaps.append(int(g))
    errs.append(m["entry_count_error"] + m["exit_count_error"])
    idsw.append(m["id_switches"])

fig, ax1 = plt.subplots(figsize=(7, 4))
ax1.plot(gaps, errs, "o-", color="tab:red", label="ошибка (входы+выходы)")
ax1.set_xlabel("длина пропуска, кадров")
ax1.set_ylabel("суммарная ошибка", color="tab:red")
ax1.tick_params(axis="y", labelcolor="tab:red")
ax1.grid(True, alpha=0.4)

ax2 = ax1.twinx()
ax2.plot(gaps, idsw, "s--", color="tab:blue", label="id_switches")
ax2.set_ylabel("id_switches", color="tab:blue")
ax2.tick_params(axis="y", labelcolor="tab:blue")

ax1.axvline(8, color="gray", linestyle=":", label="max_missed=8")
ax1.set_title("Ошибка подсчёта vs длина пропуска (easy.yaml)")
fig.tight_layout()
Path("results/gaps").mkdir(parents=True, exist_ok=True)
fig.savefig("results/gaps/plot.png", dpi=150)
print("saved results/gaps/plot.png")
