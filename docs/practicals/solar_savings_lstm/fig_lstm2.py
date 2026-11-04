import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
w = pd.read_csv('data/dingle_weather_2010_2021.csv', parse_dates=['time']).set_index('time')
win = w['shortwave_radiation']['2018-06-11 12:00':'2018-06-12 11:00']      # 24 hours
truth_next = w['shortwave_radiation']['2018-06-12 12:00']
fig = plt.figure(figsize=(13, 4.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.6, 1.35], wspace=0.08)
# ---- left: the window
ax = fig.add_subplot(gs[0])
hours = np.arange(1, 25)
ax.bar(hours, win.values, color="#ffd166", edgecolor="#b08900", width=0.8)
ax.set_xticks([1, 7, 13, 19, 24]); ax.set_xticklabels(["1\n12:00", "7\n18:00", "13\n00:00", "19\n06:00", "24\n11:00"]); ax.set_xlabel("hour of the window (11 June 2018 12:00 to 12 June 11:00)")
ax.set_ylabel("radiation, W/m²"); ax.set_title("1. The window: the last 24 hours", fontsize=10, loc="left")
ax.set_ylim(0, 950); ax.grid(alpha=0.3, axis="y")
ax.annotate("", xy=(24.6, 880), xytext=(0.5, 880), arrowprops=dict(arrowstyle="->", color="#555"))
ax.text(12.5, 905, "the LSTM reads the hours in this order", ha="center", fontsize=8, color="#555")
# ---- middle: one LSTM with a memory loop
ax2 = fig.add_subplot(gs[1]); ax2.set_xlim(0, 10); ax2.set_ylim(0, 10); ax2.axis("off")
ax2.set_title("2. One LSTM, used 24 times with the same weights", fontsize=10, loc="left")
ax2.add_patch(FancyBboxPatch((3.0, 3.6), 4.0, 2.8, boxstyle="round,pad=0.1", facecolor="#f4a261", edgecolor="#555"))
ax2.text(5.0, 5.4, "LSTM", ha="center", va="center", fontsize=13, fontweight="bold")
ax2.text(5.0, 4.4, "reads the inputs of one hour\nand updates the memory", ha="center", va="center", fontsize=8.5)
# input arrow
ax2.annotate("", xy=(2.95, 5.0), xytext=(0.3, 5.0), arrowprops=dict(arrowstyle="->", linewidth=1.8))
ax2.text(1.6, 5.5, "inputs of\nhour t", ha="center", fontsize=8.5)
# memory loop
from matplotlib.patches import Arc
ax2.add_patch(Arc((5.0, 6.4), 3.6, 3.2, theta1=10, theta2=170, color="#e76f51", linewidth=2.4))
ax2.annotate("", xy=(6.75, 6.75), xytext=(6.6, 7.3), arrowprops=dict(arrowstyle="-|>", color="#e76f51", linewidth=2.4, mutation_scale=18))
ax2.text(5.0, 8.75, "memory of `units` numbers,\npassed from one hour to the next", ha="center", va="center", fontsize=8.5, color="#e76f51")
# output arrow
ax2.annotate("", xy=(9.7, 5.0), xytext=(7.05, 5.0), arrowprops=dict(arrowstyle="->", linewidth=1.8))
ax2.text(8.4, 5.5, "memory after\nhour 24", ha="center", fontsize=8.5)
ax2.text(5.0, 1.9, "At hour 1 the memory starts at zero. At each hour the LSTM\ndecides what to add to it and what to forget. After hour 24\nthe memory summarises the whole window.", ha="center", va="center", fontsize=8.5, color="#333")
# ---- right: the head and the forecast
ax3 = fig.add_subplot(gs[2])
ax3.bar(hours, win.values, color="#ffd166", edgecolor="#b08900", width=0.8, alpha=0.5)
ax3.bar([25], [truth_next], color="none", edgecolor="#2a9d8f", width=0.8, linestyle="--")
ax3.plot([25], [truth_next * 0.93], marker="o", color="#90be6d", markersize=9)
ax3.set_xticks([1, 13, 25]); ax3.set_xticklabels(["1", "13", "25\n12:00"])
ax3.set_xlabel("hour"); ax3.set_ylim(0, 950); ax3.grid(alpha=0.3, axis="y"); ax3.set_yticklabels([])
ax3.set_title("3. LastHour + Linear give the forecast of hour 25", fontsize=10, loc="left")
ax3.annotate("forecast\n(green dot)", xy=(25, truth_next * 0.93), xytext=(16.5, 820), fontsize=8.5, arrowprops=dict(arrowstyle="->", color="#555"))
ax3.annotate("truth, known\nonly later", xy=(25.4, truth_next), xytext=(17.5, 560), fontsize=8.5, color="#2a9d8f", arrowprops=dict(arrowstyle="->", color="#2a9d8f"))
ax3.text(12.5, 60, "LastHour keeps the memory after hour 24.\nLinear turns its `units` numbers into one number.", ha="center", fontsize=8, color="#333")
plt.savefig("p6/assets/lstm_architecture.png", dpi=130, bbox_inches="tight"); print("ok", truth_next)
