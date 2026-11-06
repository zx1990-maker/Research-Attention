"""Create a Chinese, evidence-linked summary from finished benchmark outputs."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PAPER = {
    "PCA+OU": {1:(.40,-2.54),3:(1.26,-2.72),5:(.99,-3.44),8:(.78,-4.15),
               10:(.80,-4.32),15:(.51,-5.24),30:(.18,-6.45),100:(-.35,-7.05)},
    "PCA+LongConv": {1:(2.26,1.19),3:(2.76,1.57),5:(2.41,1.30),8:(2.64,1.50),
                     10:(2.66,1.52),15:(2.56,1.41),30:(2.79,1.57),100:(2.66,1.44)},
}


def plot_results(metrics, results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    colors = {"PCA+OU":"#C2653C", "PCA+LongConv":"#247C8F"}
    for model in PAPER:
        m = metrics[metrics.model==model].sort_values("K")
        if len(m):
            axes[0].plot(m.K, m.gross_sharpe, "o-", label=f"{model} reproduced", color=colors[model])
            axes[1].plot(m.K, m.net_sharpe, "o-", label=f"{model} reproduced", color=colors[model])
        ks = np.array(list(PAPER[model])); vals = np.array(list(PAPER[model].values()))
        axes[0].plot(ks, vals[:,0], "--", alpha=.6, label=f"{model} paper", color=colors[model])
        axes[1].plot(ks, vals[:,1], "--", alpha=.6, label=f"{model} paper", color=colors[model])
    for ax,title in zip(axes,["Gross annualized Sharpe", "Net annualized Sharpe"]):
        ax.axhline(0,color="#999999",lw=.8); ax.set_xscale("log"); ax.set_xticks(list(PAPER["PCA+OU"]));
        ax.set_xticklabels(list(PAPER["PCA+OU"])); ax.set_xlabel("Number of PCA factors K")
        ax.set_title(title); ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.savefig(results/"sharpe_comparison.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5,4.1), constrained_layout=True)
    for model,color in colors.items():
        slug = "ou" if model=="PCA+OU" else "longconv"
        p = results/f"{slug}_k30"/"daily.csv"
        if not p.exists(): continue
        d = pd.read_csv(p, parse_dates=["date"])
        wealth = (1+d.net_excess.to_numpy()).cumprod()
        ax.plot(d.date, wealth, label=f"{model} K=30 net",color=color)
    ax.axhline(1,color="#999999",lw=.8); ax.set_yscale("log")
    ax.set_ylabel("Cumulative excess-return wealth (log scale)")
    ax.set_title("Reproduced daily out-of-sample paths, 1998–2021")
    ax.grid(alpha=.2); ax.legend()
    fig.savefig(results/"net_wealth_k30.png", dpi=180); plt.close(fig)


def main():
    results=ROOT/"results"; metrics=pd.read_csv(results/"metrics.csv")
    plot_results(metrics,results)
    cache=json.loads((ROOT/"cache"/"complete.json").read_text())
    audit=json.loads((ROOT/"audit.json").read_text())
    lines=["# PCA+OU 与 PCA+LongConv 复现报告", "",
           "本次运行完整覆盖论文的 1998 年 1 月至 2021 年 12 月样本外区间，逐日更新 252 日 PCA，"
           "并分别运行 OU 阈值策略和每年重训的 LongConv。表中“本项目”是实际运行结果；“论文”是原文 Table 2 的比较数字。",
           "", "## 样本与模型", "",
           f"- 样本外交易日：{int(metrics.days.max()):,}；原始面板 1985–2021 年。月度股票池按前月末市值 top 500 形成。",
           f"- PCA 因子最大 K=100；逐日合格股票数均值 {cache['eligible_mean']:.1f}，最低 {cache['eligible_min']}。",
           f"- 缺少当期收益、但仅凭过去信息仍合格的股票日：{cache['missing_realized_eligible_asset_days']}。"
           "这些日期按零超额收益估值并单独记录缺失仓位；当日因子残差也受到该估值选择影响。",
           "- 收益采用 RET_ADJ−rf，RET_ADJ 已包含退市回报；调仓成本为股票仓位差 L1 的 5bp，做空成本为做空绝对仓位的每日 1bp。",
           "- LongConv 固定论文公布的 30 日回看、一层 32 维、30 epoch、Adam 学习率 0.003、dropout 0.1；1990–1995 年拟合、1996–1997 年验证，用于确定输入收益单位；1998–2021 年每年以前 8 年重训。",
           "- 网络输入把小数残差乘 100，转为百分数单位；交易收益、成本和最终绩效仍用小数计。选择证据见 `results/input_scale_validation.csv`。",
           "", "## 主要结果", "",
           "年化收益率与波动率以百分数表示；Sharpe 为年化。年收益是 252×日均超额收益，不是复利 CAGR。",
           "", "| 模型 | K | 本项目毛 Sharpe | 论文毛 Sharpe | 本项目净 Sharpe | 论文净 Sharpe | 本项目年化毛收益 | 本项目年化净收益 | 年换手成本 |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for model in PAPER:
        for k in PAPER[model]:
            m=metrics[(metrics.model==model)&(metrics.K==k)]
            if len(m)!=1: raise RuntimeError(f"Missing or duplicate run {model} K={k}")
            row=m.iloc[0]; pg,pn=PAPER[model][k]
            lines.append(f"| {model} | {k} | {row.gross_sharpe:.2f} | {pg:.2f} | {row.net_sharpe:.2f} | {pn:.2f} | "
                         f"{100*row.gross_annual_excess_return:.2f}% | {100*row.net_annual_excess_return:.2f}% | "
                         f"{100*row.annual_transaction_cost:.2f}% |")
    l30=metrics[(metrics.model=="PCA+LongConv")&(metrics.K==30)].iloc[0]
    o30=metrics[(metrics.model=="PCA+OU")&(metrics.K==30)].iloc[0]
    lines += ["", f"K=30 时，本项目 LongConv 净 Sharpe 为 {l30.net_sharpe:.2f}，低于论文的 1.57；"
              f"年化毛收益 {100*l30.gross_annual_excess_return:.2f}% 和波动率 {100*l30.gross_annual_volatility:.2f}% "
              "也低于论文的 15.15% 与 5.42%。因此差异不只来自交易费用，因子残差和策略风险敞口亦有差别。",
              f"OU 的本项目 K=30 净 Sharpe 为 {o30.net_sharpe:.2f}，论文为 −6.45；"
              f"两者都显示毛收益经高换手后转负，本项目年化换手成本为 {100*o30.annual_transaction_cost:.2f}%。"]
    validation=pd.read_csv(results/"input_scale_validation.csv")
    lines += ["", "## 样本外前的输入单位验证（K=30）", "",
              "两种输入单位均只用 1990–1995 年拟合，并在 1996–1997 年验证；事前选取净 Sharpe 较高的单位用于全部样本外年份。",
              "", "| 输入单位 | 验证期毛 Sharpe | 验证期净 Sharpe |", "|---|---:|---:|"]
    for _, row in validation.iterrows():
        lines.append(f"| 小数残差 × {row.input_scale:g} | {row.gross_sharpe:.2f} | {row.net_sharpe:.2f} |")
    lines += ["", "![本项目与论文的毛、净 Sharpe 对照](results/sharpe_comparison.png)", "",
              "![K=30 的样本外净收益路径](results/net_wealth_k30.png)", "",
              "## K=30 的交易成本分解", "",
              "| 模型 | 年化毛收益 | 年换手成本 | 年做空成本 | 年化净收益 | 每日 L1 换手 | 缺失收益仓位日期 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for model in PAPER:
        row=metrics[(metrics.model==model)&(metrics.K==30)].iloc[0]
        lines.append(f"| {model} | {row.gross_annual_excess_return*100:.2f}% | "
                     f"{row.annual_transaction_cost*100:.2f}% | {row.annual_short_cost*100:.2f}% | "
                     f"{row.net_annual_excess_return*100:.2f}% | {row.mean_daily_turnover_l1:.3f} | "
                     f"{int(row.days_missing_return_exposure)} |")
    lines += ["", "## 与原文数字不同的原因及解释边界", "",
              "论文主文没有公布两个 PCA 基准的完整代码。本文选用严格滞后的协方差 PCA 和正交投影；原文对协方差与相关性 PCA 的措辞不一致，"
              "其参考文献 [19] 的公开实现采用相关矩阵 PCA 和 60 日 OLS beta。两者的残差、交易仓位和成本会不同。",
              "LongConv 的输入投影、激活与输出映射未在论文公开，本文用共享标量输入层、因果 LongConv、GELU、dropout 和标量输出层；"
              "优化批次与随机种子也是本项目设定。模型完整定义见 `models.py` 和 `run_manifest.json`。",
              "本项目数据虽然覆盖相同年月与 top500 规则，但无法证明与作者原始 CRSP/Compustat 文件逐行一致。"
              "提供数据中存在少数缺失实现收益，且退市后旧月度名单可能仍保留股票；本文在已知退市事件下一交易日移出，并披露剩余缺失估值。",
              "原始三份 parquet 最初位于微信临时目录，在完整数据审计与逐日 PCA 缓存生成后，该临时目录于 2026-11-06 被清理。"
              "本次结果使用已完成并以 SHA-256 封存的缓存继续计算；`cache/seal.json` 记录缓存哈希及审计时的原始文件哈希。"
              "如需从原始 parquet 重新计算 PCA，必须重新取得这些文件。",
              "市场 beta 采用本面板可观测股票的前日市值加权代理，不能与论文的市场因子 beta 作逐数字比较。",
              "因此本次成果是可审计的**方法复现与样本外运行**。任何差异应首先检查 PCA/beta 约定、网络外围、数据版本和缺失值处理；"
              "不应根据 1998–2021 年样本外表现反向调参。",
              "", "## 复核与文件", "",
              "`results/metrics.csv` 是本表的数值来源。各模型目录的 `daily.csv` 与 `weights.npz` 可逐日复算收益、成本和股票仓位；"
              "`yearly.csv` 便于查看时间稳定性。`audit.json` 记录原始文件 SHA-256 与数据质量审计，"
              "`run_manifest.json` 记录配置、库版本和 PCA 缓存指纹。`results/verification.json` 记录从逐股仓位重新计算损益的核对。"
              "运行方法与工程选择见 `README.md`，论文与演示核对见 `notes/`。",
              "", "论文：[Epstein et al. (2025)](https://arxiv.org/abs/2510.11616)；OU 规则的原始参考："
              "[Guijarro-Ordonez et al. (2021)](https://arxiv.org/abs/2106.04028)。", ""]
    (ROOT/"report.md").write_text("\n".join(lines))
    print(ROOT/"report.md")


if __name__=="__main__": main()
