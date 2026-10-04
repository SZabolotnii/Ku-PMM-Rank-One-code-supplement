"""Draw the data figures of section 8 from the frozen result files only (no rerun).

    fig_rho_panel_{uk,en}.{pdf,png}  variance ratio along u at fixed rho (results/k1_rho_direction_panel.csv)
    fig_s3_sweep_{uk,en}.{pdf,png}   excess of the feasible g over its prediction, S = 3
                                     (results/k1_s3_rho_sweep.csv, diagnostic run)

Vector PDF goes into LaTeX, 600 dpi PNG is for a DOCX build. No hatch fills: pattern
tiles extend the bounding box past the page (gate G8).

    .venv/bin/python scripts/make_paper_figures.py
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, MultipleLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
# the manuscript repository writes into paper/, the public supplement into output/
OUT = ROOT / ("paper" if (ROOT / "paper").is_dir() else "output") / "figures"

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#9a9890"
GRID = "#e6e5e0"
EMPIRICAL = "#256abf"
# ordinal blue ramp for n = 500, 2000, 8000; validated with dataviz validate_palette.js --ordinal
RAMP = ("#5598e7", "#256abf", "#0d366b")
MARKERS = ("o", "s", "^")

TEXT = {
    "uk": {
        "rho": r"параметр базису $\rho$",
        "ratio": r"відношення вздовж $\mathbf{u}$",
        "pred": "передбачення",
        "emp": r"емпірика, 95 % інтервал",
        "excess": r"надлишок над передбаченням $g_{\mathrm{fe}}$",
        "deg": r"$\rho = 1/2$",
        "n": r"$n = {}$",
    },
    "en": {
        "rho": r"basis parameter $\rho$",
        "ratio": r"ratio along $\mathbf{u}$",
        "pred": "prediction",
        "emp": "empirical, 95% interval",
        "excess": r"excess over the prediction $g_{\mathrm{fe}}$",
        "deg": r"$\rho = 1/2$",
        "n": r"$n = {}$",
    },
}


def _read(name):
    with (RESULTS / name).open() as handle:
        return list(csv.DictReader(handle))


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9,
        "axes.edgecolor": INK_2, "axes.linewidth": 0.6, "axes.labelcolor": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "legend.frameon": False, "pdf.fonttype": 42,
    })


def _comma(lang, nd):
    if lang == "uk":
        return FuncFormatter(lambda x, _: f"{x:.{nd}f}".replace(".", ",").replace("-", "−"))
    return FuncFormatter(lambda x, _: f"{x:.{nd}f}".replace("-", "−"))


def _split(x, *ys):
    """Two runs, rho < 1/2 and rho > 1/2: the degenerate point rho = 1/2 is not on the grid,
    so no line is drawn across it."""
    for keep in (lambda r: r < 0.5, lambda r: r > 0.5):
        idx = [i for i, r in enumerate(x) if keep(r)]
        yield ([x[i] for i in idx],) + tuple([y[i] for i in idx] for y in ys)


def _axes(ax, lang, ynd):
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.xaxis.set_major_formatter(_comma(lang, 1))
    ax.yaxis.set_major_formatter(_comma(lang, ynd))
    ax.set_xlim(-0.04, 1.04)


def rho_panel_data():
    rows = _read("k1_rho_direction_panel.csv")
    data = {}
    for sid in ("A2L", "A4L"):
        blk = sorted((r for r in rows if r["scenario_id"] == sid), key=lambda r: float(r["rho"]))
        data[sid] = {k: [float(r[c]) for r in blk] for k, c in (
            ("rho", "rho"), ("pred", "predicted_ratio_along_u"), ("emp", "empirical_ratio_along_u"),
            ("lo", "ratio_u_ci_low"), ("hi", "ratio_u_ci_high"))}
    return data


def s3_sweep_data():
    rows = _read("k1_s3_rho_sweep.csv")
    data = {}
    for sid in ("S200", "S100"):
        for n in (500, 2000, 8000):
            blk = sorted((r for r in rows if r["scenario_id"] == sid and int(r["sample_size"]) == n),
                         key=lambda r: float(r["rho"]))
            data[(sid, n)] = ([float(r["rho"]) for r in blk],
                              [float(r["excess_over_feasible_theory"]) for r in blk])
    return data


def fig_rho_panel(lang):
    t = TEXT[lang]
    data = rho_panel_data()
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.5), sharey=True, constrained_layout=True)
    for ax, sid in zip(axes, ("A2L", "A4L")):
        d = data[sid]
        ax.axhline(1.0, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
        for k, (x, y) in enumerate(_split(d["rho"], d["pred"])):
            ax.plot(x, y, color=INK_2, linewidth=1.4, label=t["pred"] if k == 0 else None, zorder=2)
        err = [[e - lo for e, lo in zip(d["emp"], d["lo"])], [hi - e for e, hi in zip(d["emp"], d["hi"])]]
        ax.errorbar(d["rho"], d["emp"], yerr=err, fmt="o", markersize=4, color=EMPIRICAL,
                    ecolor=EMPIRICAL, elinewidth=1.0, capsize=0, label=t["emp"], zorder=3)
        ax.set_title(sid, loc="left", color=INK)
        ax.set_xlabel(t["rho"])
        _axes(ax, lang, 2)
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
    axes[0].set_ylabel(t["ratio"])
    axes[0].legend(loc="upper left", handlelength=1.6)
    return fig


def fig_s3_sweep(lang):
    t = TEXT[lang]
    data = s3_sweep_data()
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.5), sharey=True, constrained_layout=True)
    for ax, sid in zip(axes, ("S200", "S100")):
        ax.axhline(0.0, color=MUTED, linewidth=0.8)
        ax.axvline(0.5, color=MUTED, linewidth=0.8, linestyle=(0, (1, 2)))
        for color, marker, n in zip(RAMP, MARKERS, (500, 2000, 8000)):
            for k, (x, y) in enumerate(_split(*data[(sid, n)])):
                ax.plot(x, y, color=color, marker=marker, markersize=4, linewidth=1.4,
                        label=t["n"].format(n) if k == 0 else None)
        ax.set_title(sid, loc="left", color=INK)
        ax.set_xlabel(t["rho"])
        _axes(ax, lang, 1)
    axes[0].text(0.52, 0.40, t["deg"], color=INK_2, fontsize=8)
    axes[0].set_ylabel(t["excess"])
    axes[1].legend(loc="upper right", handlelength=2.2)
    return fig


def main():
    _style()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, make in (("fig_rho_panel", fig_rho_panel), ("fig_s3_sweep", fig_s3_sweep)):
        for lang in ("uk", "en"):
            fig = make(lang)
            for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
                path = OUT / f"{name}_{lang}.{ext}"
                fig.savefig(path, **kw)
                print(f"wrote {path.relative_to(ROOT)}")
            plt.close(fig)


if __name__ == "__main__":
    main()
