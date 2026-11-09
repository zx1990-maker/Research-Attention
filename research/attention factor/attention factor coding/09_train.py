"""
环节 ⑨：训练 + 样本外评估（Attention Factors for Statistical Arbitrage 复现）

原文：
  [ATT] 4.2 节：8 年滚动窗口训练，每年重新训练一次，样本外 1998-2021
  [ATT] 附录 B Table 4：d = 32，1 层 LongConv，dropout 0.1，30 个 epoch，lr = 0.003，
        weight decay = 0.05，λ_Var = 100，λ_squash = 0.001
  [19]  附录 D：训练窗口按时间切成连续、不重叠的 125 天 batch，依次优化；Adam 优化器

做法：
  对每个样本外年份 y（1998 ... 2021）：
    1. 训练窗口 = y-8 年 1 月 ~ y-1 年 12 月（96 个月块）；水平量标准化只用这个窗口拟合
    2. 每 6 个月块组成一个 batch（约 125 个交易日，见下面“自行设定”），按时间顺序依次优化，
       30 个 epoch。batch 第一天的成本接上一个 batch 最后一天的权重（不传梯度）
    3. 用训练好的模型，按时间顺序算 y 年每天的样本外收益（评估模式，成本精确计算，
       跨年接上一年最后一天的权重）
  每年存一次模型和样本外日收益；中断后重新运行同一条命令会从断点继续（已完成的年份跳过）。

自行设定（原文未说明）：
  - batch = 6 个月块（约 125 天）而不是严格 125 天：数据按月组织，月初换股票池，按月切更自然
  - 优化器 AdamW（weight decay 0.05 用解耦形式；[19] 用 Adam，[15] 代码用 AdamW）
  - 样本外第一天（1998-01-02）从空仓开始
  - 不在验证集上调 λ_Var、λ_ridge，直接用 Table 4 的值和 λ_ridge = 1e-5

输出（attention factor/train_K{K}/ 下）：
  train.log        全部过程（终端里看到的内容也都写在这里）
  oos_daily.csv    样本外每天：日期、毛收益、成本、净收益、换手、做空比例、rf
  model_{y}.pt     y 年使用的模型参数、标准化参数、最后一天的权重
  summary.txt      样本外表现汇总（每年 + 全样本），每年更新一次

用法（在 research 目录下）：
  caffeinate -ims python3 -u "attention factor/attention factor coding/09_train.py" --K 30
  参数：--K 因子个数（默认 30）；--start 1998 --end 2021 样本外年份；--epochs 30
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
AF_DIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from af_model import (ANNUAL, AttentionFactorModel, HistBlocks, af_loss, fit_normalizer,  # noqa: E402
                      run_blocks)

ap = argparse.ArgumentParser()
ap.add_argument("--K", type=int, default=30)
ap.add_argument("--start", type=int, default=1998)
ap.add_argument("--end", type=int, default=2021)
ap.add_argument("--epochs", type=int, default=30)          # [ATT] Table 4
ap.add_argument("--lr", type=float, default=0.003)         # [ATT] Table 4
ap.add_argument("--wd", type=float, default=0.05)          # [ATT] Table 4
ap.add_argument("--train-years", type=int, default=8)      # [ATT] 4.2 节
ap.add_argument("--batch-months", type=int, default=6)     # ≈ 125 天（[19] 附录 D）
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max-batches", type=int, default=0, help="只用于快速测试：每个 epoch 最多几个 batch")
ap.add_argument("--out", type=str, default="")
args = ap.parse_args()

OUT = args.out or os.path.join(AF_DIR, f"train_K{args.K}")
os.makedirs(OUT, exist_ok=True)
LOG_FILE = os.path.join(OUT, "train.log")
CSV_FILE = os.path.join(OUT, "oos_daily.csv")
torch.set_num_threads(max(1, os.cpu_count() or 1))


def log(msg=""):
    """同时打印到终端和写进 train.log。"""
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as fh:
        fh.write(line + "\n")


def fmt_t(s):
    s = int(s)
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def perf(r, rf=None):
    """年化 Sharpe、均值、波动（[19] 脚注 12 的算法）"""
    r = np.asarray(r, dtype=float)
    ex = r - (0 if rf is None else np.asarray(rf, dtype=float))
    mu, sd = ex.mean() * ANNUAL, r.std() * np.sqrt(ANNUAL)
    return mu / sd if sd > 0 else np.nan, r.mean() * ANNUAL, sd


def write_summary(df):
    lines = [f"Attention Factor 样本外表现  K = {args.K}  （{df['date'].min()} ~ {df['date'].max()}，"
             f"{len(df)} 个交易日）", "",
             "Sharpe = √252 × mean(R - rf) / std(R)；均值、波动为年化",
             "", f"{'年份':<6}{'毛Sharpe':>10}{'净Sharpe':>10}{'净均值':>10}{'净波动':>10}{'日换手':>9}{'做空':>8}"]
    groups = list(df.groupby(df["date"].str[:4])) + [("全部", df)]
    for y, g in groups:
        sg, _, _ = perf(g["r_gross"], g["rf"])
        sn, mn, vn = perf(g["r_net"], g["rf"])
        if y == "全部":
            lines.append("-" * 63)
        lines.append(f"{y:<6}{sg:>10.2f}{sn:>10.2f}{mn:>10.1%}{vn:>10.1%}{g['turnover'].mean():>9.3f}"
                     f"{g['short'].mean():>8.3f}")
    lines += ["", "原文摘要对照（1998-2021，24 年）：扣成本前 Sharpe 4 以上，扣成本后 2.3"]
    with open(os.path.join(OUT, "summary.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return lines


# ---------------------------------------------------------------------------
hb = HistBlocks(AF_DIR)
assert hb.rf is not None, "缺少 hist_rf.npy，请先运行 08_loss.py"
years = list(range(args.start, args.end + 1))
done = set()
if os.path.exists(CSV_FILE):
    done = set(pd.read_csv(CSV_FILE, dtype={"date": str})["date"].str[:4].astype(int))

log("=" * 70)
log(f"开始训练 K = {args.K}，样本外 {args.start}-{args.end}，每年 {args.epochs} 个 epoch，"
    f"lr {args.lr}，weight decay {args.wd}，输出目录 {OUT}")
if done:
    log(f"已完成的年份（跳过）：{sorted(done)}")

# 估算总步数，用来算预计剩余时间
todo = [y for y in years if y not in done]
n_batch = int(np.ceil(args.train_years * 12 / args.batch_months))
if args.max_batches:
    n_batch = min(n_batch, args.max_batches)
total_steps, steps_done, t_start = len(todo) * args.epochs * n_batch, 0, time.time()

w_last, p_last = None, None
for y in years:
    ckpt = os.path.join(OUT, f"model_{y}.pt")
    if y in done:
        if os.path.exists(ckpt):
            st = torch.load(ckpt, weights_only=False)
            w_last, p_last = st["w_last"], st["permno_last"]
        continue

    # ---- 1 训练窗口 ---------------------------------------------------------
    tr_months = [f"{yy}-{mm:02d}" for yy in range(y - args.train_years, y) for mm in range(1, 13)]
    tr_idx = [hb.months.index(m) for m in tr_months if m in hb.months]
    oos_idx = [hb.months.index(f"{y}-{mm:02d}") for mm in range(1, 13) if f"{y}-{mm:02d}" in hb.months]
    norm = fit_normalizer(AF_DIR, f"{y - args.train_years}-01-01", f"{y - 1}-12-31")
    batches = [tr_idx[i:i + args.batch_months] for i in range(0, len(tr_idx), args.batch_months)]
    if args.max_batches:
        batches = batches[:args.max_batches]

    torch.manual_seed(args.seed)
    model = AttentionFactorModel(args.K)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    log("-" * 70)
    log(f"{y} 年：训练窗口 {tr_months[0]} ~ {tr_months[-1]}（{len(tr_idx)} 个月，{len(batches)} 个 batch）")

    for ep in range(1, args.epochs + 1):
        model.train()
        t_ep = time.time()
        wl, pl = None, None
        stats = []
        for bi, bidx in enumerate(batches, 1):
            blocks = [hb.block(b, norm) for b in bidx]
            out = run_blocks(model, blocks, wl, pl)
            loss, sr, ev = af_loss(out["r_net"], out["rf"], out["eps"], out["R"], out["valid"], out["sid"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            wl, pl = out["w_last"].detach(), out["permno_last"]
            stats.append((loss.item(), sr.item(), ev.item(), out["turnover"][1:].mean().item()))
            steps_done += 1
            el = time.time() - t_start
            eta = el / steps_done * (total_steps - steps_done)
            done_pct = steps_done / total_steps
            bar = "█" * int(done_pct * 20) + "·" * (20 - int(done_pct * 20))
            print(f"\r  {y} epoch {ep:>2}/{args.epochs} batch {bi:>2}/{len(batches)}  "
                  f"loss {loss.item():+8.2f}  |  总进度 {bar} {done_pct:5.1%}  已用 {fmt_t(el)}  剩余约 {fmt_t(eta)}   ",
                  end="", flush=True)
        print("\r" + " " * 120 + "\r", end="", flush=True)
        s = np.array(stats).mean(0)
        log(f"  {y} epoch {ep:>2}/{args.epochs}  loss {s[0]:+8.2f}  训练Sharpe {s[1]:+6.2f}  "
            f"EV {s[2]:.3f}  日换手 {s[3]:.3f}  ({time.time() - t_ep:.0f}s)")

    # ---- 2 样本外 ---------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        oos_blocks = [hb.block(b, norm) for b in oos_idx]
        out = run_blocks(model, oos_blocks, w_last, p_last)
    w_last, p_last = out["w_last"], out["permno_last"]
    dates = np.concatenate([b["dates"] for b in oos_blocks]).astype(str)
    df_y = pd.DataFrame({"date": dates, "r_gross": out["r_gross"].numpy(), "cost": out["cost"].numpy(),
                         "r_net": out["r_net"].numpy(), "turnover": out["turnover"].numpy(),
                         "short": out["short"].numpy(), "rf": out["rf"].numpy()})
    df_y.to_csv(CSV_FILE, mode="a", header=not os.path.exists(CSV_FILE), index=False)
    torch.save({"year": y, "K": args.K, "model": model.state_dict(), "normalizer": norm.state_dict(),
                "w_last": w_last, "permno_last": p_last, "args": vars(args)}, ckpt)

    sg, _, _ = perf(df_y["r_gross"], df_y["rf"])
    sn, mn, _ = perf(df_y["r_net"], df_y["rf"])
    log(f"  >>> {y} 样本外：毛Sharpe {sg:+.2f}  净Sharpe {sn:+.2f}  净年化收益 {mn:+.1%}  "
        f"日换手 {df_y['turnover'].mean():.3f}")
    df_all = pd.read_csv(CSV_FILE, dtype={"date": str})
    sga, _, _ = perf(df_all["r_gross"], df_all["rf"])
    sna, _, _ = perf(df_all["r_net"], df_all["rf"])
    log(f"  >>> 累计 {df_all['date'].min()[:4]}-{y}：毛Sharpe {sga:+.2f}  净Sharpe {sna:+.2f}")
    write_summary(df_all)

log("=" * 70)
log("全部完成。样本外汇总（也在 summary.txt）：")
for line in write_summary(pd.read_csv(CSV_FILE, dtype={"date": str})):
    log(line)
