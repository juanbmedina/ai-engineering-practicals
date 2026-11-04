import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, FancyBboxPatch, Arc, FancyArrowPatch, Circle

# ---------------- Figure A: a typical installation and the tilt of the roof
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})
# house side view (looking from the east): south is to the right
ax.add_patch(Rectangle((2, 0), 6, 3, facecolor="#f1e9dc", edgecolor="#555"))           # wall
ax.add_patch(Polygon([[1.6, 3], [5, 5.2], [8.4, 3]], closed=True, facecolor="#b5651d", edgecolor="#555"))  # roof
ax.add_patch(Rectangle((3.2, 0), 1.0, 1.9, facecolor="#8ecae6", edgecolor="#555"))    # door
ax.add_patch(Rectangle((5.8, 1.2), 1.2, 1.0, facecolor="#cde7f0", edgecolor="#555"))  # window
# panels on the south slope (right), parallel to the roof
slope = np.array([8.4 - 5, 3 - 5.2]); slope = slope / np.linalg.norm(slope)
normal = np.array([-slope[1], slope[0]])
for k in range(4):
    p0 = np.array([5.35, 4.98]) + slope * (k * 0.78)
    off = normal * 0.12
    quad = [p0 + off, p0 + off + slope * 0.72, p0 + off + slope * 0.72 + normal * 0.22, p0 + off + normal * 0.22]
    ax.add_patch(Polygon(quad, closed=True, facecolor="#1d3557", edgecolor="#a8dadc", linewidth=1.2))
# tilt angle
ax.plot([5, 9.2], [5.2 - 0, 5.2 - 0], linestyle=":", color="#333", linewidth=1)   # horizontal reference from the ridge
ax.plot([5, 8.4], [5.2, 3.0], color="#333", linewidth=1)
angle = np.degrees(np.arctan2(2.2, 3.4))
ax.add_patch(Arc((5, 5.2), 2.6, 2.6, theta1=-angle, theta2=0, color="#e76f51", linewidth=2))
ax.text(6.55, 4.75, f"tilt ≈ 30°", color="#e76f51", fontsize=11, fontweight="bold")
# sun paths
ax.text(8.3, 5.45, "horizontal", fontsize=8, color="#333", va="center")
ax.annotate("", xy=(7.0, 4.25), xytext=(10.3, 7.4), arrowprops=dict(arrowstyle="->", color="#f4a261", linewidth=2))
ax.add_patch(Circle((10.6, 7.6), 0.42, color="#f4a261")); ax.text(10.6, 8.35, "sun in June\n(high, 60° above the horizon)", ha="center", fontsize=8.5)
ax.annotate("", xy=(7.4, 4.0), xytext=(11.2, 5.0), arrowprops=dict(arrowstyle="->", color="#f4a261", linewidth=2, linestyle="--"))
ax.add_patch(Circle((11.5, 5.05), 0.35, color="#f4a261", alpha=0.6)); ax.text(11.5, 4.3, "sun in December\n(low, 15°)", ha="center", fontsize=8.5)
ax.annotate("south", xy=(12.2, 0.6), xytext=(9.6, 0.6), arrowprops=dict(arrowstyle="->", color="#333"), fontsize=10, va="center")
ax.text(1.2, 0.6, "north", fontsize=10, va="center")
ax.plot([0, 13], [0, 0], color="#555", linewidth=1.5)
ax.set_xlim(0, 13); ax.set_ylim(-0.3, 9.3); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("A rooftop installation seen from the side: panels on the slope that faces south", fontsize=10)
# right: yearly radiation vs tilt for Dingle (approximate, from the data: 1027 horizontal, 1182 at 30 south)
tilts = np.array([0, 10, 20, 30, 40, 50, 60, 90])
south = np.array([1.00, 1.08, 1.13, 1.16, 1.15, 1.12, 1.07, 0.80]) * 1027
east = np.array([1.00, 1.01, 1.00, 0.97, 0.93, 0.88, 0.82, 0.60]) * 1027
north = np.array([1.00, 0.93, 0.85, 0.76, 0.67, 0.58, 0.50, 0.30]) * 1027
ax2.plot(tilts, south, marker="o", color="#e76f51", label="roof faces south")
ax2.plot(tilts, east, marker="s", color="#2a9d8f", label="roof faces east or west")
ax2.plot(tilts, north, marker="^", color="#264653", label="roof faces north")
ax2.axvline(30, color="gray", linestyle=":", linewidth=1); ax2.text(31, 420, "typical Irish roof\n30° to 35°", fontsize=8, color="gray")
ax2.set_xlabel("tilt of the roof (degrees)"); ax2.set_ylabel("kWh / m² per year on the panel")
ax2.set_title("Radiation received by a panel in Dingle, by tilt and orientation (approximate)", fontsize=10)
ax2.set_ylim(0, 1300); ax2.legend(fontsize=8, loc="lower left"); ax2.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("p6/assets/panel_installation_tilt.png", dpi=130); plt.close()

# ---------------- Figure B: the LSTM read as a chain over the window
fig, ax = plt.subplots(figsize=(12, 4.2))
hours = ["hour 1", "hour 2", "hour 3", "…", "hour 24"]
xs = [1.0, 3.2, 5.4, 7.6, 9.8]
for x, h in zip(xs, hours):
    if h == "…":
        ax.text(x + 0.6, 2.3, "…", fontsize=22, ha="center", va="center"); ax.text(x + 0.6, 0.55, "…", fontsize=22, ha="center", va="center"); continue
    # inputs box
    ax.add_patch(FancyBboxPatch((x, 0.1), 1.3, 0.9, boxstyle="round,pad=0.05", facecolor="#ffd166", edgecolor="#555"))
    ax.text(x + 0.65, 0.55, f"inputs of\n{h}", ha="center", va="center", fontsize=8.5)
    # LSTM block
    ax.add_patch(FancyBboxPatch((x, 1.8), 1.3, 1.0, boxstyle="round,pad=0.05", facecolor="#f4a261", edgecolor="#555"))
    ax.text(x + 0.65, 2.3, "LSTM\nsame weights", ha="center", va="center", fontsize=8.5)
    ax.annotate("", xy=(x + 0.65, 1.78), xytext=(x + 0.65, 1.02), arrowprops=dict(arrowstyle="->"))
    # memory out
    ax.add_patch(FancyBboxPatch((x, 3.6), 1.3, 0.7, boxstyle="round,pad=0.05", facecolor="#e9ecef", edgecolor="#555"))
    ax.text(x + 0.65, 3.95, "memory\n(units numbers)", ha="center", va="center", fontsize=8)
    ax.annotate("", xy=(x + 0.65, 3.58), xytext=(x + 0.65, 2.82), arrowprops=dict(arrowstyle="->"))
# memory arrows between hours
for a, b in [(xs[0], xs[1]), (xs[1], xs[2]), (xs[2], xs[3]), (xs[3], xs[4])]:
    ax.annotate("", xy=(b - 0.02, 2.3), xytext=(a + 1.32, 2.3), arrowprops=dict(arrowstyle="->", color="#e76f51", linewidth=2))
ax.text(2.75, 1.4, "the memory passes on", color="#e76f51", fontsize=8, ha="center")
# last hour -> linear -> forecast
ax.add_patch(FancyBboxPatch((11.6, 3.6), 1.4, 0.7, boxstyle="round,pad=0.05", facecolor="#8ecae6", edgecolor="#555"))
ax.text(12.3, 3.95, "Linear\nunits -> 1", ha="center", va="center", fontsize=8.5)
ax.annotate("", xy=(11.58, 3.95), xytext=(11.12, 3.95), arrowprops=dict(arrowstyle="->"))
ax.add_patch(FancyBboxPatch((13.5, 3.6), 1.4, 0.7, boxstyle="round,pad=0.05", facecolor="#90be6d", edgecolor="#555"))
ax.text(14.2, 3.95, "forecast of\nhour 25", ha="center", va="center", fontsize=8.5)
ax.annotate("", xy=(13.48, 3.95), xytext=(13.02, 3.95), arrowprops=dict(arrowstyle="->"))
ax.text(10.45, 4.55, "LastHour keeps only this memory", fontsize=8.5, ha="center", color="#333")
ax.text(0.2, 0.55, "window", fontsize=9, rotation=90, va="center"); ax.text(0.2, 2.3, "LSTMBlock", fontsize=9, rotation=90, va="center")
ax.set_xlim(0, 15.2); ax.set_ylim(-0.1, 4.9); ax.axis("off")
ax.set_title("The model reads the window one hour at a time with the same weights, carries a memory forward, and turns the last memory into the forecast", fontsize=10)
plt.tight_layout(); plt.savefig("p6/assets/lstm_architecture.png", dpi=130); plt.close()
print("ok")
