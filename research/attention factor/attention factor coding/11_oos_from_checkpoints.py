"""
11：用队友训练好的模型（每年一个 model_{y}.pt）重新算样本外每日收益，补上 Beta 和原文口径的 μ、σ

为什么要做：队友只给了 summary.txt 和模型文件，没有 oos_daily.csv。
  模型文件里有每年的模型参数和水平量标准化参数，所以不用重新训练，
  只要在我们的数据上把每年的样本外部分再跑一遍（评估模式，和 09_train.py 第 2 部分完全相同）。

做法：
  1. 安全加载 .pt：torch.load(weights_only=True)，只额外放行 numpy 数组的重建函数，不执行任意代码。
  2. 对每个 y = 1998..2021：载入 model_y.pt 的参数与标准化参数，按时间顺序跑 y 年的月块，
     成本跨年接上一年最后一天的权重（与 09_train.py 相同），1998-01-02 从空仓开始。
  3. 校验：逐年 Sharpe 与队友 summary.txt 对比。如果一致，说明队友的这份数据与我们本地数据相同。
  4. Beta：样本外日收益对市场超额收益做 OLS 回归的斜率（原文没写回归细节，自行设定：
     用 Fama-French 日度 Mkt-RF，带截距）。
  5. 原文 Table 2 口径：SR = μ / σ，μ、σ 为日收益均值 × 252、标准差 × √252（不减 rf）。

输出（attention factor/ 下）：
  11_oos_daily_full.csv   每日样本外收益（与 09 的 oos_daily.csv 列相同）
  11_oos_from_checkpoints.log                        校验结果、Beta、Table 2 口径的数字、市场组合对比
"""
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
from numpy._core.multiarray import _reconstruct
import numpy.dtypes as npd

HERE = os.path.dirname(os.path.abspath(__file__))
AF_DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(AF_DIR)
sys.path.insert(0, HERE)
from af_model import ANNUAL, AttentionFactorModel, HistBlocks, LevelNormalizer, run_blocks  # noqa: E402

TEAM = os.path.expanduser("~/Desktop/训练模型与summary")
RUNS = {"full": "完整数据"}   # 「快照数据」那次用的是另一份数据，在本地数据上无法重现，已不用
YEARS = range(1998, 2022)
LOG = os.path.join(AF_DIR, "11_oos_from_checkpoints.log")
open(LOG, "w").close()
torch.set_num_threads(max(1, os.cpu_count() or 1))

SAFE = [(_reconstruct, "numpy._core.multiarray._reconstruct"), (_reconstruct, "numpy.core.multiarray._reconstruct"),
        np.ndarray, np.dtype] + [getattr(npd, n) for n in dir(npd) if n.endswith("DType")]


def log(s=""):
    print(s, flush=True)
    with open(LOG, "a") as fh:
        fh.write(s + "\n")


def safe_load(path):
    with torch.serialization.safe_globals(SAFE):
        return torch.load(path, weights_only=True, map_location="cpu")


def sr_rf(r, rf):           # summary.txt 的口径：√252·mean(R−rf)/std(R)
    return (np.mean(r - rf) * ANNUAL) / (np.std(r) * np.sqrt(ANNUAL))


def table2(r):              # 原文 Table 2 口径：μ/σ，不减 rf
    mu, sd = np.mean(r) * ANNUAL, np.std(r, ddof=1) * np.sqrt(ANNUAL)
    return mu / sd, mu, sd


def beta(r, m):
    X = np.column_stack([np.ones_like(m), m])
    return np.linalg.lstsq(X, r, rcond=None)[0][1]


def read_summary(path):
    out = {}
    for line in open(path):
        m = re.match(r"^(\d{4}|全部)\s+(-?[\d.]+)\s+(-?[\d.]+)", line)
        if m:
            out[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return out


# ---------------- 市场收益 ----------------
ff = pd.read_sas(os.path.join(ROOT, "wrds/ff_daily.sas7bdat"))
ff["date"] = pd.to_datetime(ff["date"]).dt.strftime("%Y-%m-%d")
ff = ff.set_index("date")

hb = HistBlocks(AF_DIR)
results = {}
for key, folder in RUNS.items():
    log("=" * 78)
    log(f"{key}（{folder}）")
    log("=" * 78)
    w_last, p_last, frames = None, None, []
    for y in YEARS:
        ck = safe_load(os.path.join(TEAM, folder, "models", f"model_{y}.pt"))
        model = AttentionFactorModel(ck["K"])
        model.load_state_dict(ck["model"])
        model.eval()
        norm = LevelNormalizer.from_state_dict(ck["normalizer"])
        idx = [hb.months.index(f"{y}-{m:02d}") for m in range(1, 13) if f"{y}-{m:02d}" in hb.months]
        with torch.no_grad():
            blocks = [hb.block(b, norm) for b in idx]
            out = run_blocks(model, blocks, w_last, p_last)
        w_last, p_last = out["w_last"], out["permno_last"]
        dates = np.concatenate([b["dates"] for b in blocks]).astype("datetime64[D]").astype(str)
        frames.append(pd.DataFrame({"date": dates, "r_gross": out["r_gross"].numpy(), "cost": out["cost"].numpy(),
                                    "r_net": out["r_net"].numpy(), "turnover": out["turnover"].numpy(),
                                    "short": out["short"].numpy(), "rf": out["rf"].numpy()}))
        print(f"\r  {y} 完成", end="", flush=True)
    print()
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(os.path.join(AF_DIR, f"11_oos_daily_{key}.csv"), index=False)
    results[key] = df

    # ---- 校验：逐年 Sharpe（summary 口径）与队友 summary.txt 对比
    ref = read_summary(os.path.join(TEAM, folder, "summary.txt"))
    log(f"校验：逐年 Sharpe（summary 口径，减 rf）与队友 summary.txt 对比")
    log(f"{'年份':<6}{'本次毛SR':>10}{'队友毛SR':>10}{'本次净SR':>10}{'队友净SR':>10}")
    diffs = []
    for y, g in list(df.groupby(df["date"].str[:4])) + [("全部", df)]:
        a, b = sr_rf(g.r_gross.values, g.rf.values), sr_rf(g.r_net.values, g.rf.values)
        ra, rb = ref.get(y, (np.nan, np.nan))
        diffs += [abs(a - ra), abs(b - rb)]
        log(f"{y:<6}{a:>10.2f}{ra:>10.2f}{b:>10.2f}{rb:>10.2f}")
    log(f"最大绝对差 {np.nanmax(diffs):.3f}（summary 保留两位小数，≤ 0.005 即为完全一致）")
    log("")

# ---------------- Table 2 口径 + Beta ----------------
log("=" * 78)
log("原文 Table 2 口径（SR = μ/σ，不减 rf；μ、σ 为 %/年）与 Beta")
log("Beta：日收益对 Fama-French Mkt-RF 回归（带截距）的斜率；同时给出对 Mkt（= Mkt-RF + rf）的结果")
log("=" * 78)
for key, df in results.items():
    m = ff.reindex(df["date"])
    assert m["mktrf"].notna().all()
    sr, mu, sd = table2(df.r_gross.values)
    srn, mun, sdn = table2(df.r_net.values)
    b_g, b_n = beta(df.r_gross.values, m.mktrf.values), beta(df.r_net.values, m.mktrf.values)
    log(f"{key:<9} SR {sr:.2f}  μ {mu*100:.2f}  σ {sd*100:.2f}  |  SR_net {srn:.2f}  μ_net {mun*100:.2f}  "
        f"σ_net {sdn*100:.2f}  |  Beta(毛) {b_g:.3f}  Beta(净) {b_n:.3f}")

# ---------------- 市场组合：不同定义与原文对比 ----------------
log("")
log("=" * 78)
log("Market 行：不同定义（1998-2021，Table 2 口径）与原文 0.42 / 8.61 / 20.37 对比")
log("=" * 78)
dates = results["full"]["date"]
m = ff.reindex(dates)
p = pd.read_parquet(os.path.join(ROOT, "data/daily_panel.parquet"), columns=["PERMNO", "DATE", "me", "RET_ADJ", "in_universe"])
p["DATE"] = pd.to_datetime(p["DATE"])
p = p[p.DATE >= "1997-11-01"].sort_values(["PERMNO", "DATE"])
p["me_lag"] = p.groupby("PERMNO")["me"].shift(1)
q = p[(p.in_universe == 1) & (p.DATE >= "1998-01-01") & (p.DATE <= "2021-12-31")].dropna(subset=["me_lag", "RET_ADJ"])
q = q.assign(wr=q.RET_ADJ * q.me_lag)
g = q.groupby("DATE")
vw = (g.wr.sum() / g.me_lag.sum())
ew = g.RET_ADJ.mean()
vw.index, ew.index = vw.index.strftime("%Y-%m-%d"), ew.index.strftime("%Y-%m-%d")
vw, ew = vw.reindex(dates).values, ew.reindex(dates).values
rf = m.rf.values
cands = [("股票池 500 只，等权，总收益（之前表里用的）", ew),
         ("股票池 500 只，等权，超额收益", ew - rf),
         ("股票池 500 只，市值加权，总收益", vw),
         ("股票池 500 只，市值加权，超额收益", vw - rf),
         ("Fama-French Mkt-RF（全市场市值加权，超额）", m.mktrf.values),
         ("Fama-French Mkt（全市场市值加权，总收益）", m.mktrf.values + rf)]
for name, r in cands:
    s, mu, sd = table2(r)
    log(f"  {name:<36} SR {s:.2f}  μ {mu*100:6.2f}  σ {sd*100:6.2f}")
log("  原文 Market                                SR 0.42  μ   8.61  σ  20.37")

# ---------------- Beta：按原文定义 ----------------
# [ATT] Table 2 注释："Beta denotes the market beta. The equally weighted market portfolio is provided for reference."
# 因此 Market = 股票池 500 只等权组合（总收益，与表中其他行同口径），Beta = 对这个等权市场组合回归的斜率。
log("")
log("=" * 78)
log("Beta（原文定义：对等权市场组合回归，带截距）—— 表中使用这一列")
log("=" * 78)
for key, df in results.items():
    e = pd.Series(ew, index=dates).reindex(df["date"]).values
    log(f"{key:<9} Beta(毛) {beta(df.r_gross.values, e):.3f}  Beta(净) {beta(df.r_net.values, e):.3f}"
        )
s, mu, sd = table2(ew)
log(f"Market（股票池 500 只等权）SR {s:.2f}  μ {mu*100:.2f}  σ {sd*100:.2f}   原文 0.42 / 8.61 / 20.37")
