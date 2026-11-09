"""
环节 ⑦：交易成本和净收益 —— 检查脚本（Attention Factors for Statistical Arbitrage 复现）

原文：
  [ATT] 3.3 节 "Arbitrage Trading Objective"：
        "We measure transaction costs as in [19], which is common in this literature:
         cost(ω_t, ω_{t-1}) = 0.0005 × ‖ω_t - ω_{t-1}‖₁ + 0.0001 × ‖max(-ω_t, 0)‖₁"
        "The portfolio net return is then calculated as: R^port_{t,net} = R^port_t - cost(ω_t, ω_{t-1})"
  [19]  III.J 节：cost(w^R_{t-1}, w^R_{t-2}) = 0.0005‖w^R_{t-1} - w^R_{t-2}‖_{L1} + 0.0001‖min(w^R_{t-1}, 0)‖_{L1}
        "from the portfolio returns and use these net portfolio returns in the optimization problem"

这个脚本不训练，只验证环节 ⑦ 的计算（1998 年 1 月、2 月两个块，跨一次股票池调整，参数随机初始化）。

输入：attention factor/ 下 hist_*.npy、X.npy、dates.npy、feature_cols.json
输出：attention factor/07_trading_costs.log

用法（在 research 目录下）：
    python "attention factor/attention factor coding/07_trading_costs.py"
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
AF_DIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from af_model import (N_FEAT, TC_SHORT, TC_TRADE, AttentionFactorWeights, FeatureEmbedding,  # noqa: E402
                      HistBlocks, LongConvSignal, ResidualProjection, align_prev_weights,
                      block_costs, block_forward, fit_normalizer)

LOG_PATH = os.path.join(AF_DIR, "07_trading_costs.log")
K_CHECK = 8
D_EMB = 32
SEED = 0
FIRST_TRAIN = ("1990-01-01", "1997-12-31")
MONTHS = ["1998-01", "1998-02"]

t0 = time.time()
LOG = []


def log(msg=""):
    print(msg)
    LOG.append(str(msg))


checks = []


def add(item, ok_, val):
    checks.append((item, bool(ok_), val))


# ---------------------------------------------------------------------------
# 读数据、前向到环节 ⑥
# ---------------------------------------------------------------------------
norm = fit_normalizer(AF_DIR, *FIRST_TRAIN)
hb = HistBlocks(AF_DIR)
blks = [hb.block(hb.months.index(m), norm) for m in MONTHS]

torch.manual_seed(SEED)
emb, att, proj = FeatureEmbedding(N_FEAT, D_EMB), AttentionFactorWeights(K_CHECK, D_EMB), ResidualProjection()
net = LongConvSignal()
net.eval()
with torch.no_grad():
    outs = [block_forward(emb, att, proj, net, b["X"], b["R"], b["valid"]) for b in blks]
w1, w2 = outs[0]["w"].double(), outs[1]["w"].double()
p1, p2 = blks[0]["permno"], blks[1]["permno"]
c1, to1, sh1 = block_costs(w1, p1)                           # 1 月：从空仓开始
c2, to2, sh2 = block_costs(w2, p2, w_last=w1[-1], permno_last=p1)   # 2 月：接 1 月最后一天

# ---- 1 公式：逐天用 numpy 手算（1 月块内） ----------------------------------------
wn = w1.numpy()
hand = np.array([TC_TRADE * np.abs(wn[i] - (wn[i - 1] if i else 0)).sum() + TC_SHORT * np.maximum(-wn[i], 0).sum()
                 for i in range(len(wn))])
d1 = np.abs(hand - c1.numpy()).max()
add("1 cost = 0.0005‖ω_t-ω_{t-1}‖₁ + 0.0001‖max(-ω_t,0)‖₁（逐天手算）", d1 < 1e-15, f"最大差 {d1:.1e}")

# ---- 2 第一天从空仓开始：换手 = ‖ω‖₁ = 1 ----------------------------------------
add("2 从空仓开始的第一天：换手 = ‖ω‖₁ = 1", abs(to1[0].item() - 1) < 1e-6, f"换手 {to1[0].item():.6f}")

# ---- 3 月内：位置不变，按 PERMNO 对齐 = 原样 ----------------------------------------
al, ex = align_prev_weights(w1[3], p1, p1)
add("3 同一月内：按 PERMNO 对齐后与原权重相同，没有离开股票池的仓位",
    (al - w1[3]).abs().max() == 0 and ex.item() == 0, f"最大差 {(al - w1[3]).abs().max():.1e}，离开 {ex.item():.1e}")

# ---- 4 月初：按 PERMNO 对齐（与 pandas 外连接独立计算的换手比较） ------------------------
a = pd.DataFrame({"PERMNO": p1[p1 > 0], "w_prev": w1[-1].numpy()[p1 > 0]})
b = pd.DataFrame({"PERMNO": p2[p2 > 0], "w_cur": w2[0].numpy()[p2 > 0]})
mg = a.merge(b, on="PERMNO", how="outer").fillna(0.0)
to_pd = np.abs(mg["w_cur"] - mg["w_prev"]).sum()
add("4 月初换股票池：换手 = pandas 按 PERMNO 外连接后的 Σ|ω_t - ω_{t-1}|", abs(to_pd - to2[0].item()) < 1e-12,
    f"{blks[1]['dates'][0]}：本实现 {to2[0].item():.6f}，pandas {to_pd:.6f}")

set1, set2 = set(p1[p1 > 0]), set(p2[p2 > 0])
n_exit, n_enter, n_stay = len(set1 - set2), len(set2 - set1), len(set1 & set2)
_, exit_l1 = align_prev_weights(w1[-1], p1, p2)
pos_moved = int(sum(1 for k in range(len(p2)) if p2[k] > 0 and p1[k] != p2[k]))

# ---- 5 若直接按位置相减，结果是错的 ----------------------------------------------
to_wrong = (w2[0] - w1[-1]).abs().sum().item()
add("5 直接按“第几个位置”相减的换手与正确值不同（说明必须对齐）", abs(to_wrong - to2[0].item()) > 1e-3,
    f"按位置 {to_wrong:.4f}，按 PERMNO {to2[0].item():.4f}")

# ---- 6 做空成本：Σmax(-ω,0) = (1 - Σω)/2（因为 ‖ω‖₁ = 1） -----------------------------
sh_id = (1 - w1.sum(-1)) / 2
add("6 做空部分 Σmax(-ω,0) = (‖ω‖₁ - Σω)/2", (sh_id - sh1).abs().max() < 1e-6,
    f"最大差 {(sh_id - sh1).abs().max():.1e}；做空部分均值 {sh1.mean():.4f}")

# ---- 7 净收益 = 组合收益 - 成本 --------------------------------------------------
r_gross = torch.cat([outs[0]["r_port"], outs[1]["r_port"]]).double()
cost = torch.cat([c1, c2])
r_net = r_gross - cost
d7 = (r_gross - r_net - cost).abs().max().item()
add("7 净收益 = R^port - cost，天数一致（浮点舍入 < 1e-15 视为 0）", r_net.shape == r_gross.shape and d7 < 1e-15,
    f"{len(r_net)} 天，R^port - R_net - cost 最大 {d7:.1e}")

# ---- 8 梯度：成本对参数可导 -------------------------------------------------------
torch.manual_seed(SEED)
emb_g, att_g, net_g = FeatureEmbedding(N_FEAT, D_EMB), AttentionFactorWeights(K_CHECK, D_EMB), LongConvSignal()
og = [block_forward(emb_g, att_g, proj, net_g, b["X"], b["R"], b["valid"]) for b in blks]
cg1, _, _ = block_costs(og[0]["w"], p1)
cg2, _, _ = block_costs(og[1]["w"], p2, w_last=og[0]["w"][-1], permno_last=p1)
(cg1.sum() + cg2.sum()).backward()
gn = {"W_K": emb_g.W_K.weight.grad.norm().item(), "Q": att_g.Q.grad.norm().item(),
      "LongConv 卷积核": net_g.layers[0].kernel.grad.norm().item(), "解码层": net_g.decoder.weight.grad.norm().item()}
add("8 梯度从成本传回 W_K、Q、LongConv（训练时模型会学着降低换手）", all(v > 0 for v in gn.values()),
    "，".join(f"{k} {v:.1e}" for k, v in gn.items()))

# ---------------------------------------------------------------------------
# 全样本：每月股票池的进出
# ---------------------------------------------------------------------------
enter_cnt = []
for m in range(1, len(hb.months)):
    a_, b_ = hb.permno[m - 1], hb.permno[m]
    enter_cnt.append(len(set(b_[b_ > 0]) - set(a_[a_ > 0])))
enter_cnt = np.array(enter_cnt)

# 随机初始化下的成本量级
to_all = torch.cat([to1[1:], to2]).numpy()           # 去掉从空仓开始的第一天
c_all = cost[1:].numpy()
g_all = r_gross[1:].numpy()

# ===========================================================================
# 写 log
# ===========================================================================
n_pass = sum(c[1] for c in checks)
log("=" * 78)
log("环节 ⑦：交易成本和净收益 —— 检查")
log("=" * 78)
log("")
log("一、原文")
log("  [ATT] 3.3 节 \"Arbitrage Trading Objective\": \"We measure transaction costs as in [19], which is")
log("        common in this literature:")
log("        cost(ω_t, ω_{t-1}) = 0.0005 × ‖ω_t - ω_{t-1}‖₁ + 0.0001 × ‖max(-ω_t, 0)‖₁,")
log("        The first penalty represents a transaction cost of 5 basis points per transaction, whereas")
log("        the second one is a shorting cost of 1 basis point. The portfolio net return is then")
log("        calculated as: R^port_{t,net} = R^port_t - cost(ω_t, ω_{t-1})\"")
log("  [19] III.J 节: \"we subtract the market-friction costs cost(w^R_{t-1}, w^R_{t-2}) =")
log("        0.0005‖w^R_{t-1} - w^R_{t-2}‖_{L1} + 0.0001‖min(w^R_{t-1}, 0)‖_{L1} from the portfolio returns and")
log("        use these net portfolio returns in the optimization problem ... where w^R_{t-1} is the")
log("        strategy's allocation weight vector at time t-1\"")
log("")
log("二、实现（af_model.py）")
log("  1. trading_costs(w, w_prev, exit_l1): 公式照原文。TC_TRADE = 0.0005，TC_SHORT = 0.0001。")
log("  2. 时间约定（依据 [19] III.J 节）: t 日净收益 = ω_{t-1}ᵀR_t - cost(ω_{t-1}, ω_{t-2})，")
log("     即 t-1 日收盘调仓到 ω_{t-1} 的成本，从持有 ω_{t-1} 在 t 日赚的收益里扣。")
log("     [ATT] 写成 cost(ω_t, ω_{t-1})，只是下标整体平移，含义相同。")
log("  3. 按 PERMNO 对齐（align_prev_weights）:")
log("     - 前后两天都在池子里的股票：对应相减")
log("     - 今天新进池子的股票：昨天权重记为 0（新建仓位，算换手）")
log("     - 昨天在、今天不在池子里的股票：今天权重为 0，它的 |ω_{t-1}| 计入换手（清仓，算换手）")
log("     同一个月块内位置固定对应同一只股票，只有月初第一天需要对齐（block_costs）。")
log("  4. 第一天从空仓开始时换手 = 1（自行设定；样本外评估从 1998-01 第一天建仓）。")
log("  5. 原文与 [19] 都没有按前一天的收益调整 ω_{t-2}（价格变动造成的权重漂移），这里也不调整。")
log("  6. 训练时的处理（留到环节 ⑨）: [19] 附录 D 说训练时用“上一个 epoch 的权重”近似 ω_{t-2}，")
log("     原因是它的配置函数把上一期权重当输入，没法并行。[ATT] 的 ω_{t-1} 不依赖 ω_{t-2}，")
log("     一个 batch 内所有天的 ω 可以一次并行算出来，batch 内的成本可以精确计算；")
log("     只有 batch 第一天要用前一天的权重，那时再决定怎么取。")
log("")
log(f"三、检查结果：{n_pass} / {len(checks)} 项通过")
log(f"  （模型 K = {K_CHECK}，参数随机初始化，λ_ridge = {proj.ridge:g}，{MONTHS[0]} 与 {MONTHS[1]} 两个块，"
    f"共 {len(r_net)} 个交易日）")
log("")
for item, ok_, val in checks:
    log(f"  [{'通过' if ok_ else '失败'}] {item}")
    log(f"         {val}")
log("")
log(f"  {MONTHS[0]} -> {MONTHS[1]} 的股票池调整：留下 {n_stay} 只，离开 {n_exit} 只，新进 {n_enter} 只；")
log(f"    离开的股票在 1 月最后一天的 Σ|ω| = {exit_l1.item():.4f}（计入 2 月第一天的换手）")
log(f"    两个月第 k 个位置不是同一只股票的有 {pos_moved} 个（股票按 PERMNO 排序，一只进出就会让后面的位置错开），")
log(f"    这就是检查 5 按位置相减会出错的原因。")
log(f"  全样本 {hb.months[0]} - {hb.months[-1]}：每月新进股票池 {enter_cnt.min()}-{enter_cnt.max()} 只，"
    f"平均 {enter_cnt.mean():.1f} 只")
log("")
log("四、随机初始化下的成本量级（未训练，只看量级；不含从空仓开始的第一天）")
log(f"  每日换手 ‖ω_t - ω_{{t-1}}‖₁：均值 {to_all.mean():.3f}，范围 [{to_all.min():.3f}, {to_all.max():.3f}]"
    f"（上限是 2：全部卖掉再全部反向买入）")
log(f"  每日成本：均值 {c_all.mean() * 1e4:.2f} 个基点；年化成本 {c_all.mean() * 252:.1%}")
log(f"  每日组合收益（扣成本前）：均值 {g_all.mean() * 1e4:+.2f} 个基点，标准差 {g_all.std() * 1e4:.1f} 个基点")
log("  随机初始化的 LongConv 输出跟着残差窗口每天变化，换手很高；训练目标里扣了成本，")
log("  模型要学会少换手。[ATT] 第 1 节: \"The key issue in a two-step approach is that the factors cannot")
log("  adjust to reduce trading costs for arbitrage strategies. For example, PCA factors have high turnover")
log("  and large short positions, which diminish net performance.\"")
log("")
log("五、下一环节")
log("  环节 ⑧：损失函数 = -(年化净 Sharpe) - λ_Var × 解释方差，λ_Var = 100")
log("")
log(f"总耗时 {time.time() - t0:.0f}s")

with open(LOG_PATH, "w") as fh:
    fh.write("\n".join(LOG) + "\n")
