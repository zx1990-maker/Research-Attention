"""
环节 ⑧：损失函数 —— 数据（无风险利率）+ 检查脚本（Attention Factors for Statistical Arbitrage 复现）

原文：
  [ATT] 3.3 节目标函数：
        max_{ω^F, ω^port}  (R̄^port_net - R_f) / sqrt((1/T) Σ_t (R^port_{t,net} - R̄^port_net)²)      net Sharpe ratio
                           + λ_Var · (1/N) Σ_i (1 - Var(e_i) / Var(R_i))                           explained variance
        subject to ‖ω_t‖₁ = 1
        "The tradeoff between these two objectives is selected optimally on the validation data.
         Including the explained variance is necessary for identification, and empirically improves
         the performance."
  [ATT] 附录 B Table 4：λ_Var = 100
  [19]  附录 D："replacing the mean and variance by their annualized sample counterpart over a training set"
        脚注 12：μ = 252/T Σ R_t，σ = sqrt(252/T Σ (R_t - μ)²)

这个脚本做两件事：
  Part 1  数据：把 Fama-French 日无风险利率 rf 对齐到 04 的日期，存成 hist_rf.npy
  Part 2  检查：用 1998 年 1-6 月（约 125 个交易日，相当于 [19] 的一个 batch），参数随机初始化，
          验证 Sharpe、解释方差、损失、梯度，最后试跑 10 步 Adam 看损失能否下降

输入：wrds/ff_daily.sas7bdat，attention factor/ 下 hist_*.npy、X.npy、dates.npy、feature_cols.json
输出：attention factor/hist_rf.npy、attention factor/08_loss.log

用法（在 research 目录下）：
    python "attention factor/attention factor coding/08_loss.py"
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import pyreadstat
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
AF_DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(AF_DIR)
sys.path.insert(0, HERE)
from af_model import (ANNUAL, EV_MIN_OBS, LAMBDA_VAR, AttentionFactorModel, HistBlocks,  # noqa: E402
                      af_loss, explained_variance, fit_normalizer, run_blocks, sharpe_annualized)

FF_FILE = os.path.join(ROOT, "wrds", "ff_daily.sas7bdat")
LOG_PATH = os.path.join(AF_DIR, "08_loss.log")
K_CHECK = 8
SEED = 0
FIRST_TRAIN = ("1990-01-01", "1997-12-31")
MONTHS = ["1998-01", "1998-02", "1998-03", "1998-04", "1998-05", "1998-06"]
N_STEPS = 10
LR = 0.003            # [ATT] Table 4

t0 = time.time()
LOG = []


def log(msg=""):
    print(msg)
    LOG.append(str(msg))


checks = []


def add(item, ok_, val):
    checks.append((item, bool(ok_), val))


# ===========================================================================
# Part 1  无风险利率对齐到 04 的日期
# ===========================================================================
ff, _ = pyreadstat.read_sas7bdat(FF_FILE)
ff["date"] = pd.to_datetime(ff["date"])
hist_dates = pd.DatetimeIndex(np.load(os.path.join(AF_DIR, "hist_dates.npy")))
rf_s = ff.set_index("date")["rf"].reindex(hist_dates)
n_rf_missing = int(rf_s.isna().sum())
hist_rf = rf_s.fillna(0.0).values.astype("float32")
np.save(os.path.join(AF_DIR, "hist_rf.npy"), hist_rf)

# ===========================================================================
# Part 2  检查
# ===========================================================================
norm = fit_normalizer(AF_DIR, *FIRST_TRAIN)
hb = HistBlocks(AF_DIR)
blocks = [hb.block(hb.months.index(m), norm) for m in MONTHS]
dates = np.concatenate([b["dates"] for b in blocks])

torch.manual_seed(SEED)
model = AttentionFactorModel(K_CHECK)
model.eval()
with torch.no_grad():
    o = run_blocks(model, blocks)
    loss, sr, ev = af_loss(o["r_net"], o["rf"], o["eps"], o["R"], o["valid"], o["sid"])
T = len(o["r_net"])

# ---- 1 Sharpe：按原文公式手算（1/T 标准差、减 R_f），再乘 √252 --------------------------
rn = o["r_net"].double().numpy()
rfn = o["rf"].double().numpy()
sr_hand = np.sqrt(252) * (rn.mean() - rfn.mean()) / np.sqrt(((rn - rn.mean()) ** 2).mean())
sr_daily = (rn.mean() - rfn.mean()) / rn.std()
add("1 Sharpe = √252 × (R̄_net - R_f) / sqrt((1/T)Σ(R_net - R̄_net)²)（numpy 手算）",
    abs(sr_hand - sr.item()) < 1e-4 * max(1, abs(sr_hand)), f"本实现 {sr.item():+.5f}，手算 {sr_hand:+.5f}")

# ---- 2 rf 对齐 -------------------------------------------------------------------
rf_chk = ff.set_index("date")["rf"].reindex(pd.DatetimeIndex(dates)).values
add("2 无风险利率按日期对齐到 Fama-French 日度 rf", n_rf_missing == 0 and np.abs(rf_chk - rfn).max() < 1e-7,
    f"hist_dates {len(hist_dates)} 天缺失 {n_rf_missing} 天；本段 {T} 天 rf 均值 {rfn.mean() * 1e4:.3f} 个基点/天"
    f"（年化 {rfn.mean() * 252:.2%}）")

# ---- 3 解释方差：与 pandas 按 PERMNO 分组独立计算的结果比较 ------------------------------
m = o["valid"].numpy()
df = pd.DataFrame({"PERMNO": o["sid"].numpy()[m], "e": o["eps"].double().numpy()[m],
                   "r": o["R"].double().numpy()[m]})
g = df.groupby("PERMNO").agg(n=("e", "size"), ve=("e", lambda v: v.var(ddof=0)),
                             vr=("r", lambda v: v.var(ddof=0)))
g = g[(g["n"] >= EV_MIN_OBS) & (g["vr"] > 0)]
ev_pd = (1 - g["ve"] / g["vr"]).mean()
add("3 解释方差 = pandas 按 PERMNO 分组计算的 (1/N)Σ(1 - Var(e_i)/Var(R_i))", abs(ev_pd - ev.item()) < 1e-4,
    f"本实现 {ev.item():.6f}，pandas {ev_pd:.6f}，N = {len(g)} 只股票")

n_obs = df.groupby("PERMNO").size()
n_all6 = int((n_obs == T).sum())
n_drop = int((n_obs < EV_MIN_OBS).sum())

# ---- 4 边界情况：残差 = 收益（因子什么都没解释）-> 0；残差 = 0 -> 1 ---------------------------
ev0 = explained_variance(o["R"], o["R"], o["valid"], o["sid"]).item()
ev1 = explained_variance(torch.zeros_like(o["R"]), o["R"], o["valid"], o["sid"]).item()
add("4 边界：残差 = 收益时 EV = 0；残差 = 0 时 EV = 1", abs(ev0) < 1e-6 and abs(ev1 - 1) < 1e-6,
    f"EV(ε=R) = {ev0:.1e}，EV(ε=0) = {ev1:.6f}")

# ---- 5 损失 = -(Sharpe + λ_Var × EV) -------------------------------------------------
add("5 loss = -(Sharpe + 100 × EV)", abs(loss.item() + sr.item() + LAMBDA_VAR * ev.item()) < 1e-4,
    f"loss {loss.item():+.4f} = -({sr.item():+.4f} + {LAMBDA_VAR:g} × {ev.item():.4f})")


# ---- 6/7 梯度结构 --------------------------------------------------------------------
def grads_of(term_fn):
    torch.manual_seed(SEED)
    mdl = AttentionFactorModel(K_CHECK)
    mdl.eval()
    out = run_blocks(mdl, blocks)
    term_fn(out).backward()
    gW, gQ = mdl.emb.W_K.weight.grad, mdl.att.Q.grad
    gL = torch.cat([p.grad.reshape(-1) for p in mdl.net.parameters() if p.grad is not None]) \
        if any(p.grad is not None for p in mdl.net.parameters()) else torch.zeros(1)
    return {"W_K": gW.norm().item(), "Q": gQ.norm().item(), "LongConv": gL.norm().item(),
            "WQ": torch.cat([gW.reshape(-1), gQ.reshape(-1)])}


g_sr = grads_of(lambda out: sharpe_annualized(out["r_net"], out["rf"]))
g_ev = grads_of(lambda out: LAMBDA_VAR * explained_variance(out["eps"], out["R"], out["valid"], out["sid"]))
add("6 Sharpe 项的梯度传到 W_K、Q、LongConv", all(g_sr[k] > 0 for k in ["W_K", "Q", "LongConv"]),
    f"W_K {g_sr['W_K']:.1e}，Q {g_sr['Q']:.1e}，LongConv {g_sr['LongConv']:.1e}")
add("7 解释方差项只影响因子（W_K、Q），对 LongConv 梯度为 0", g_ev["W_K"] > 0 and g_ev["Q"] > 0
    and g_ev["LongConv"] == 0, f"W_K {g_ev['W_K']:.1e}，Q {g_ev['Q']:.1e}，LongConv {g_ev['LongConv']:.1e}")

# 年化与否对两项权衡的影响：比较两项对 (W_K, Q) 的梯度大小
ratio_ann = g_ev["WQ"].norm().item() / g_sr["WQ"].norm().item()
ratio_daily = ratio_ann * np.sqrt(ANNUAL)
cos_sr_ev = torch.nn.functional.cosine_similarity(g_ev["WQ"], g_sr["WQ"], dim=0).item()

# ---- 8 试跑：10 步 Adam，损失下降 -----------------------------------------------------
torch.manual_seed(SEED)
mdl = AttentionFactorModel(K_CHECK)
opt = torch.optim.Adam(mdl.parameters(), lr=LR)


def eval_loss(mdl_):
    mdl_.eval()
    with torch.no_grad():
        out_ = run_blocks(mdl_, blocks)
        return [v.item() for v in af_loss(out_["r_net"], out_["rf"], out_["eps"], out_["R"], out_["valid"],
                                          out_["sid"])] + [out_["turnover"][1:].mean().item()]


hist = [eval_loss(mdl)]
t_steps = time.time()
for step in range(N_STEPS):
    mdl.train()
    out = run_blocks(mdl, blocks)
    l_, _, _ = af_loss(out["r_net"], out["rf"], out["eps"], out["R"], out["valid"], out["sid"])
    opt.zero_grad()
    l_.backward()
    opt.step()
    hist.append(eval_loss(mdl))
t_step = (time.time() - t_steps) / N_STEPS
add(f"8 试跑 {N_STEPS} 步 Adam（lr = {LR}）：同一 batch 上损失下降", hist[-1][0] < hist[0][0],
    f"loss {hist[0][0]:+.3f} -> {hist[-1][0]:+.3f}")

# ===========================================================================
# 写 log
# ===========================================================================
n_pass = sum(c[1] for c in checks)
log("=" * 78)
log("环节 ⑧：损失函数 —— 检查")
log("=" * 78)
log("")
log("一、原文")
log("  [ATT] 3.3 节: \"Our objective function maximizes the net Sharpe ratio of the arbitrage portfolio and")
log("        the explained variance of the factors. The tradeoff between these two objectives is selected")
log("        optimally on the validation data. Including the explained variance is necessary for")
log("        identification, and empirically improves the performance.\"")
log("        max  (R̄^port_net - R_f) / sqrt((1/T) Σ_t (R^port_{t,net} - R̄^port_net)²)")
log("             + λ_Var · (1/N) Σ_i (1 - Var(e_i)/Var(R_i)),   subject to ‖ω_t‖₁ = 1")
log("        \"R_f is the risk-free rate. The learned parameters that determine ω^F and ω^port are the")
log("        query matrix Q, the embedding matrix W^K, and the LongConv model parameters θ.\"")
log("  [ATT] 附录 B Table 4: λ_Var = 100")
log("  [19] 附录 D: \"we estimate the parameters ... by replacing the mean and variance by their")
log("        annualized sample counterpart over a training set\"")
log("        脚注 12: μ = 252/T Σ R_t，σ = sqrt(252/T Σ (R_t - μ)²)")
log("")
log("二、实现（af_model.py）")
log("  1. sharpe_annualized(r_net, rf) = √252 × (mean(R_net) - mean(rf)) / std(R_net)")
log("     - 分子减 R_f、分母用 1/T：照 [ATT] 公式。")
log("     - 乘 √252：照 [19] 附录 D（[ATT] 公式本身没写年化，但它说训练方式沿用 [19]，")
log("       而且 λ_Var = 100 是相对年化 Sharpe 的大小定的，见下面第五部分）。")
log("     - R_f：Fama-French 日度 rf（与 data/coding/02 计算特征用的是同一个文件），")
log("       取这段时间的均值。R_t 用的是原始收益（不是超额收益），与 [ATT] 3 节 Notation 一致。")
log("  2. explained_variance(eps, R, valid, stock_id) = (1/N) Σ_i (1 - Var(ε_i)/Var(R_i))")
log("     - Var 是一个 batch 内每只股票的时间序列方差；ε 是环节 ④ 的样本外残差 ε_t = ω^ε_{t-1}R_t。")
log("     - 自行设定：按 PERMNO 汇总（batch 跨月，同一只股票在不同月块的位置不同）；")
log(f"       只计入 batch 内至少 {EV_MIN_OBS} 天有数据的股票；用 1/n 的方差。")
log("  3. af_loss = -(Sharpe + λ_Var × EV)：神经网络做最小化，所以取负号。")
log("  4. AttentionFactorModel：W_K、Q、LongConv 放在一个模块里，就是 [ATT] 说的全部可学习参数。")
log("  5. run_blocks(model, blocks)：按时间顺序跑若干个月块，拼成一段连续交易日；")
log("     成本按 PERMNO 对齐、精确计算（环节 ⑦），第一块从空仓开始。")
log("")
log("三、Part 1 数据")
log(f"  hist_rf.npy  ({len(hist_rf)},)  float32  与 hist_dates.npy 对齐的日无风险利率，缺失 {n_rf_missing} 天")
log("")
log(f"四、检查结果：{n_pass} / {len(checks)} 项通过")
log(f"  （模型 K = {K_CHECK}，参数随机初始化，λ_ridge = {model.proj.ridge:g}，{MONTHS[0]} - {MONTHS[-1]}，"
    f"{T} 个交易日 ≈ [19] 的一个 125 天 batch）")
log("")
for item, ok_, val in checks:
    log(f"  [{'通过' if ok_ else '失败'}] {item}")
    log(f"         {val}")
log("")
log("  说明：")
log(f"    检查 3：这 {T} 天里出现过的股票按 PERMNO 汇总；6 个月都在池子里的 {n_all6} 只（{T} 天数据），")
log(f"      不足 {EV_MIN_OBS} 天被排除的 {n_drop} 只。")
log("    检查 7：解释方差只取决于残差 ε = ω^ε R，而 ω^ε 只由 W_K、Q 决定，和 LongConv 无关，")
log("      所以这一项只推动因子去解释收益；Sharpe 项三组参数都推动。这正是原文说的")
log("      “解释方差项对识别（identification）是必要的”：没有它，因子只为 Sharpe 服务，可以变成任何组合。")
log("")
log("五、为什么 Sharpe 必须年化（截图里提到的问题）")
log(f"  随机初始化时这 {T} 天：日 Sharpe = {sr_daily:+.4f}，年化 Sharpe = {sr.item():+.3f}（差 √252 ≈ 15.9 倍）；")
log(f"  解释方差 EV = {ev.item():.4f}，λ_Var × EV = {LAMBDA_VAR * ev.item():.2f}")
log(f"  对 (W_K, Q) 的梯度大小之比  ‖∇(λ_Var·EV)‖ / ‖∇Sharpe‖：")
log(f"    Sharpe 年化：{ratio_ann:.2f}")
log(f"    Sharpe 不年化：{ratio_daily:.1f}（解释方差项相对放大 15.9 倍，几乎完全主导因子的学习）")
log(f"  两个梯度方向的余弦相似度 {cos_sr_ev:+.3f}（两个目标对因子的要求不同，需要权衡）")
log("  所以必须和 [19] 一样用年化 Sharpe，λ_Var = 100 才是原文的权衡。")
log("")
log(f"六、试跑 {N_STEPS} 步 Adam（lr = {LR}，无权重衰减；只验证能训练，不是正式训练，正式训练在环节 ⑨）")
log("  每步: 训练模式（dropout 打开）前向 6 个月块 + 反向 + 更新；每步之后在评估模式下重算：")
log("    步    loss       年化净Sharpe    EV       平均日换手")
for i, (l_, s_, e_, to_) in enumerate(hist):
    log(f"    {i:<4d} {l_:+9.3f}   {s_:+9.3f}      {e_:.4f}    {to_:.3f}")
log(f"  每步耗时 {t_step:.2f}s（{T} 天）。一个 8 年训练窗口约 2000 天 = 16 个 batch，")
log(f"  1 个 epoch 约 {t_step * 2000 / T:.0f}s，30 个 epoch 约 {t_step * 2000 / T * 30 / 60:.0f} 分钟（K = {K_CHECK}）。")
log("")
log("七、下一环节")
log("  环节 ⑨：训练循环（8 年滚动窗口、每年重训、30 个 epoch、125 天一个 batch、验证集选 λ_ridge）")
log("")
log(f"总耗时 {time.time() - t0:.0f}s")

with open(LOG_PATH, "w") as fh:
    fh.write("\n".join(LOG) + "\n")
