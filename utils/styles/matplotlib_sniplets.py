import matplotlib.pyplot as plt

plt.style.use("styles/boardstate-dark.mplstyle")  # or light
def boardstate_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax
plt.rcParams["figure.figsize"] = (7.2, 4.0)   # ~16:9-ish
plt.rcParams["figure.constrained_layout.use"] = True
CALLOUT = {
    "decision": "#7C5CFF",
    "result":   "#2ECC71",
    "warning":  "#F2C94C",
    "danger":   "#EB5757",
}

ax.annotate(
    "OOD spike here",
    xy=(10, 0.82), xycoords="data",
    xytext=(0.65, 0.9), textcoords="axes fraction",
    arrowprops=dict(arrowstyle="->"),
    bbox=dict(boxstyle="round,pad=0.35", fc="none", ec=CALLOUT["warning"], lw=1.2),
)
