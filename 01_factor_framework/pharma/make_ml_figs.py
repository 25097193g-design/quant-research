#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
画医药 ML 加分两张图（补 pharma_report.pdf 最大短板：XGBoost 只有表格没图）。
读 outputs CSV -> matplotlib -> PNG。
用法: conda run -n rqsdk --no-capture-output python make_ml_figs.py
产出: outputs/ml_phase_robust.png, outputs/rqalpha_compare.png
"""
import csv
import pathlib

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang HK", "Heiti TC", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np

BASE = pathlib.Path(__file__).resolve().parent
OUT = BASE / "outputs"

BLUE = "#1f5aa5"
GRAY1 = "#a8b3c4"
GRAY2 = "#c8d0dc"


def fig_phase():
    """XGB vs baseline 三相位年化收益（3 相位全胜，排除相位运气）"""
    rows = []
    with (OUT / "ml_phase_robust.csv").open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["phase"] in ("0", "1", "2"):
                rows.append(row)
    phases = ["0", "1", "2"]
    base = [float(next(r for r in rows if r["模型"] == "baseline" and r["phase"] == p)["年化"]) * 100 for p in phases]
    xgb = [float(next(r for r in rows if r["模型"] == "XGBoost" and r["phase"] == p)["年化"]) * 100 for p in phases]

    x = np.arange(len(phases))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=150)
    b1 = ax.bar(x - w / 2, base, w, label="baseline（无 ML）", color=GRAY1)
    b2 = ax.bar(x + w / 2, xgb, w, label="XGBoost（19 簇合成）", color=BLUE)
    ax.set_xticks(x)
    ax.set_xticklabels([f"相位 {p}" for p in phases])
    ax.set_ylabel("年化收益 (%)")
    ax.set_title("XGBoost 在 3 个季度起始相位全胜 baseline", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    for bars in (b1, b2):
        for bar in bars:
            ax.annotate(f"{bar.get_height():.1f}%", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 2), textcoords="offset points", ha="center", fontsize=8.5)
    ax.axhline(0, color="#666", lw=0.6)
    fig.tight_layout()
    fig.savefig(OUT / "ml_phase_robust.png", bbox_inches="tight")
    plt.close(fig)
    print("OK  outputs/ml_phase_robust.png")


def fig_rqalpha():
    """rqalpha 真实撮合指标对比 24F / 4F / XGB19（XGB19 复现 ML 加分）"""
    rows = []
    with (OUT / "rqalpha_compare.csv").open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    label_map = {"24F季": "24F 季", "4F季": "4F 季", "XGB19(19簇)": "XGB19"}
    labels = [label_map.get(r["信号"], r["信号"]) for r in rows]
    ann = [float(r["annualized_returns"]) * 100 for r in rows]
    alpha = [float(r["alpha"]) * 100 for r in rows]
    ir = [float(r["information_ratio"]) for r in rows]
    colors = [GRAY1, GRAY2, BLUE]

    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.1), dpi=150)
    for ax, (title, vals, fmt) in zip(axes, [
        ("年化收益 (%)", ann, "{:.1f}"),
        ("Alpha (%)", alpha, "{:.1f}"),
        ("信息比率 IR", ir, "{:.2f}"),
    ]):
        bars = ax.bar(range(len(labels)), vals, color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(fmt.format(h), xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points", ha="center", fontsize=8)
    fig.suptitle("rqalpha 真实撮合：XGB19 复现 ML 加分（年化 18.8% / Alpha 19.0% / IR 0.95）",
                 fontsize=10.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "rqalpha_compare.png", bbox_inches="tight")
    plt.close(fig)
    print("OK  outputs/rqalpha_compare.png")


if __name__ == "__main__":
    fig_phase()
    fig_rqalpha()
