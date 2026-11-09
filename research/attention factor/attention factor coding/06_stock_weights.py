"""
环节 ⑥：映射回股票、归一化 —— 检查脚本（Attention Factors for Statistical Arbitrage 复现）

原文：
  [ATT] 3.3 节 "Arbitrage Trading"：
        R^port_t = ε_tᵀ ω^port_{t-1} = R_tᵀ (ω^ε_{t-1})ᵀ ω^port_{t-1} = R_tᵀ ω_{t-1}
        目标函数下方："subject to ‖ω_t‖₁ = 1"
  [19]  Guijarro-Ordonez, Pelger, Zanotti, Deep Learning Statistical Arbitrage（arXiv:2106.04028）
        公式 (3)/(5)：w^R_{t-1} = w^εᵀ_{t-1} Φ_{t-1} / ‖w^εᵀ_{t-1} Φ_{t-1}‖₁
        III.C 节："We use the standard normalization that the absolute values of the individual stock
        portfolio weights sums up to one, i.e. we use the normalization ‖ω^R_{t-1}‖₁ = 1"

这个脚本不训练，只验证环节 ⑥ 的计算（1998 年 1 月块，参数随机初始化）。

输入：attention factor/ 下 hist_*.npy、X.npy、dates.npy、feature_cols.json
输出：attention factor/06_stock_weights.log、attention factor/06_constraint_check.png

用法（在 research 目录下）：
    python "attention factor/attention factor coding/06_stock_weights.py"
"""

import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
AF_DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(AF_DIR)
sys.path.insert(0, HERE)
from af_model import (LOOKBACK, N_FEAT, AttentionFactorWeights, FeatureEmbedding,  # noqa: E402
                      HistBlocks, LongConvSignal, ResidualProjection, block_forward,
                      daily_residuals, fit_normalizer, portfolio_returns, residual_windows,
                      stock_weights)

LOG_PATH = os.path.join(AF_DIR, "06_stock_weights.log")
PDF_PATH = os.path.join(ROOT, "Attention Factors for Statistical Arbitrage.pdf")
K_CHECK = 8
D_EMB = 32
SEED = 0
FIRST_TRAIN = ("1990-01-01", "1997-12-31")
DEMO_MONTH = "1998-01"
L = LOOKBACK

t0 = time.time()
LOG = []


def log(msg=""):
    print(msg)
    LOG.append(str(msg))


checks = []


def add(item, ok_, val):
    checks.append((item, bool(ok_), val))


# ---------------------------------------------------------------------------
# 0 原文约束到底写的是 ω 还是 ω^port：直接读 PDF 里那个公式的字符
# ---------------------------------------------------------------------------
pdf_chars, pdf_note = None, "未安装 PyMuPDF，跳过"
try:
    import fitz
    doc = fitz.open(PDF_PATH)
    for page in doc:
        hits = page.search_for("subject to")
        if hits:
            r = hits[0]
            clip = fitz.Rect(r.x1, r.y0 - 4, r.x1 + 40, r.y1 + 4)        # "subject to" 后面的公式
            spans = [s for blk in page.get_text("rawdict", clip=clip)["blocks"]
                     for ln in blk.get("lines", []) for s in ln["spans"]]
            pdf_chars = [("".join(c["c"] for c in s["chars"]), round(s["size"], 1)) for s in spans]
            page.get_pixmap(clip=fitz.Rect(r.x0 - 10, r.y0 - 125, r.x0 + 260, r.y1 + 6), dpi=200).save(
                os.path.join(AF_DIR, "06_constraint_check.png"))
            pdf_note = f"第 {page.number + 1} 页"
            break
except ImportError:
    pass

# ---------------------------------------------------------------------------
# 读数据：1998 年 1 月块
# ---------------------------------------------------------------------------
norm = fit_normalizer(AF_DIR, *FIRST_TRAIN)
hb = HistBlocks(AF_DIR)
blk = hb.block(hb.months.index(DEMO_MONTH), norm)
Xb, Rb, Vb = blk["X"], blk["R"], blk["valid"]

torch.manual_seed(SEED)
emb, att, proj = FeatureEmbedding(N_FEAT, D_EMB), AttentionFactorWeights(K_CHECK, D_EMB), ResidualProjection()
net = LongConvSignal()
net.eval()
with torch.no_grad():
    out = block_forward(emb, att, proj, net, Xb, Rb, Vb)
w, w_port, w_eps, ok = out["w"], out["w_port"], out["w_eps"], out["ok"]
eps_t, R_t, valid_t = out["eps"], out["R"], out["valid"]
n_days, N = w.shape

# ---- 1 形状 ---------------------------------------------------------------
add("1 形状：ω^port (交易日, 500) -> ω (交易日, 500)", tuple(w.shape) == tuple(w_port.shape),
    f"{tuple(w_port.shape)} -> {tuple(w.shape)}")

# ---- 2 每天 ‖ω_t‖₁ = 1 -----------------------------------------------------
l1 = w.abs().sum(-1)
add("2 每天股票空间 ‖ω‖₁ = 1", (l1 - 1).abs().max() < 1e-5,
    f"{n_days} 天，‖ω‖₁ 范围 [{l1.min():.6f}, {l1.max():.6f}]")

# ---- 3 ω = (ω^ε)ᵀ ω^port / ‖·‖₁：逐只股票手算一天 -------------------------------
t_demo = 5
we = w_eps[t_demo].double().numpy()
wp = (w_port[t_demo] * ok[t_demo]).double().numpy()
w_hand = np.zeros(N)
for i in range(N):                    # 第 i 个残差组合（ω^ε 第 i 行）乘上它的下注 ω^port_i，累加
    w_hand += wp[i] * we[i]
w_hand /= np.abs(w_hand).sum()
d3 = np.abs(w_hand - w[t_demo].double().numpy()).max()
add("3 ω = Σ_i ω^port_i ×（第 i 个残差组合），再除以 L1（逐只手算）", d3 < 1e-6,
    f"{blk['dates'][t_demo]}，最大差 {d3:.1e}")

# ---- 4 两种写法的组合收益相同：ε_tᵀ ω^port / c = R_tᵀ ω ----------------------------
c = (w_eps.transpose(-1, -2) @ (w_port * ok).unsqueeze(-1)).squeeze(-1).abs().sum(-1)   # 归一化常数
r_res = (eps_t * w_port * ok).sum(-1) / c          # 残差空间算的收益
r_stk = portfolio_returns(w, R_t)                  # 股票空间算的收益
d4 = (r_res - r_stk).abs().max().item()
add("4 R^port_t：残差空间 ε_tᵀω^port 与股票空间 R_tᵀω 相同", d4 < 1e-7,
    f"最大差 {d4:.1e}（日收益量级 {r_stk.abs().mean():.1e}）")

# ---- 5 因子中性：λ_ridge = 0 时组合收益里的因子部分 ωᵀβᵀF_t 为 0 ----------------------------
# R_t = βᵀ_{t-1} F_t + ε_t（ε 的定义），所以 R^port_t = ωᵀβᵀF_t（因子部分）+ ωᵀε_t
def factor_share(lam, K=K_CHECK, dtype=torch.float64):
    torch.manual_seed(SEED)
    e_, a_, n_ = (FeatureEmbedding(N_FEAT, D_EMB).to(dtype), AttentionFactorWeights(K, D_EMB).to(dtype),
                  LongConvSignal().to(dtype))
    n_.eval()
    p_ = ResidualProjection(lam)
    with torch.no_grad():
        o_ = block_forward(e_, a_, p_, n_, Xb.to(dtype), Rb.to(dtype), Vb)
        wF_ = a_(e_(Xb[L:].to(dtype)), Vb[L:])
        bT_, _ = p_(wF_, Vb[L:])
        F_ = (wF_ @ o_["R"].unsqueeze(-1)).squeeze(-1)
        fac = (o_["w"] * (bT_ @ F_.unsqueeze(-1)).squeeze(-1)).sum(-1)
        return (fac.abs().mean() / o_["r_port"].abs().mean()).item()


share0 = factor_share(0.0)
share_tab = {(K, lam): factor_share(lam, K) for K in [8, 30] for lam in [0.0, 1e-6, 1e-5, 1e-4, 1e-3]}
add("5 因子中性：λ_ridge = 0（float64）时 R^port 中的因子部分 |ωᵀβᵀF| ≈ 0", share0 < 1e-9,
    f"|ωᵀβᵀF_t| / |R^port_t| = {share0:.1e}")

# ---- 6 不能交易的位置 -------------------------------------------------------------
n_bad = int((~ok).sum())
with torch.no_grad():
    w_port_bad = w_port.clone()
    w_port_bad[~ok] += 100.0          # 改动不可交易残差的 ω^port，结果不应变化
    w_chk = stock_weights(w_eps, w_port_bad, ok)
d6a = (w_chk - w).abs().max().item()
d6b = w[~valid_t].abs().max().item() if (~valid_t).any() else 0.0
add("6 窗口不完整的残差不下注；空位/当天无数据的股票权重为 0", d6a == 0 and d6b == 0,
    f"不完整窗口 {n_bad} 个 (股票,日)，改动它们的 ω^port 后 ω 变化 {d6a:.1e}；"
    f"空位 {int((~valid_t).sum())} 个，最大 |ω| = {d6b:.1e}")

# ---- 7 不用未来信息：改动 t 日收益 R_t，ω_{t-1} 不变 --------------------------------
j = 10                                   # 当月第 j 个交易日（块内第 L + j 天）
R2 = Rb.clone()
R2[L + j] += 0.05
with torch.no_grad():
    out2 = block_forward(emb, att, proj, net, Xb, R2, Vb)
d7a = (out2["w"][j] - w[j]).abs().max().item()
d7b = (out2["w"][j + 1] - w[j + 1]).abs().max().item()
add("7 改动 t 日收益，ω_{t-1} 不变、ω_t 改变（不用未来信息）", d7a == 0 and d7b > 0,
    f"ω_{{t-1}} 变化 {d7a:.1e}，ω_t 变化 {d7b:.1e}")

# ---- 8 梯度：从组合收益传回所有参数 ------------------------------------------------
torch.manual_seed(SEED)
emb_g, att_g, net_g = FeatureEmbedding(N_FEAT, D_EMB), AttentionFactorWeights(K_CHECK, D_EMB), LongConvSignal()
og = block_forward(emb_g, att_g, proj, net_g, Xb, Rb, Vb)
(og["r_port"].mean() / og["r_port"].std()).backward()
gn = {"W_K": emb_g.W_K.weight.grad.norm().item(), "Q": att_g.Q.grad.norm().item(),
      "LongConv 卷积核": net_g.layers[0].kernel.grad.norm().item(),
      "解码层": net_g.decoder.weight.grad.norm().item()}
add("8 梯度从组合收益传回 W_K、Q、LongConv", all(v > 0 for v in gn.values()),
    "，".join(f"{k} {v:.1e}" for k, v in gn.items()))

# ---------------------------------------------------------------------------
# 补充：如果只在残差空间把 ω^port 归一化（‖ω^port‖₁ = 1），股票空间 ‖ω‖₁ 是多少
# ---------------------------------------------------------------------------
with torch.no_grad():
    wp_eff = w_port * ok
    wp_n = wp_eff / wp_eff.abs().sum(-1, keepdim=True)
    w_alt = (w_eps.transpose(-1, -2) @ wp_n.unsqueeze(-1)).squeeze(-1)
    l1_alt = w_alt.abs().sum(-1)
    # 不同 K 下的情况
    alt_by_K = {}
    for K in [1, 8, 30, 100]:
        torch.manual_seed(SEED)
        e_, a_ = FeatureEmbedding(N_FEAT, D_EMB), AttentionFactorWeights(K, D_EMB)
        ee, we_ = daily_residuals(e_, a_, proj, Xb, Rb, Vb)
        xx, oo = residual_windows(ee, Vb)
        pp = net(xx) * oo
        pp = pp / pp.abs().sum(-1, keepdim=True)
        alt_by_K[K] = (we_[L:].transpose(-1, -2) @ pp.unsqueeze(-1)).squeeze(-1).abs().sum(-1)

# 权重的结构
wn = w.numpy()
long_l1 = np.clip(wn, 0, None).sum(1)
short_l1 = np.clip(-wn, 0, None).sum(1)
net_exp = wn.sum(1)
n_nz = (np.abs(wn) > 1e-8).sum(1)
r_np = r_stk.numpy()

# ===========================================================================
# 写 log
# ===========================================================================
n_pass = sum(c[1] for c in checks)
log("=" * 78)
log("环节 ⑥：映射回股票、归一化 —— 检查")
log("=" * 78)
log("")
log("一、原文")
log("  [ATT] 3.3 节 \"Arbitrage Trading\":")
log("        R^port_t = ε_tᵀ ω^port_{t-1} = R_tᵀ (ω^ε_{t-1})ᵀ ω^port_{t-1} = R_tᵀ ω_{t-1}")
log("        \"ω^ε_{t-1} is only a function of the characteristics X_{t-1}, while ω^port_{t-1} is only")
log("        a function of the time-series patterns in residuals\"")
log("        目标函数下方: \"subject to ‖ω_t‖₁ = 1\"")
log("  [19] 公式 (3)/(5): w^R_{t-1} = w^εᵀ_{t-1} Φ_{t-1} / ‖w^εᵀ_{t-1} Φ_{t-1}‖₁")
log("        III.C 节: \"Our strategies trade the residuals of all stocks, which are mapped back into")
log("        positions of the original stocks. We use the standard normalization that the absolute")
log("        values of the individual stock portfolio weights sums up to one, i.e. we use the")
log("        normalization ‖ω^R_{t-1}‖₁ = 1. This normalization implies a leverage constraint\"")
log("        （[19] 的 Φ_{t-1} 就是 [ATT] 的 ω^ε_{t-1}，w^ε 就是 ω^port）")
log("")
log("二、关于“约束里的 ω 有没有 port 上标”")
log(f"  直接读 PDF（{pdf_note}）\"subject to\" 后面公式的字符和字号：")
if pdf_chars:
    log("    " + "  ".join(f"'{ch}'({sz})" for ch, sz in pdf_chars if ch.strip()))
    log("  字符是 ∥ 𝜔 𝑡 ∥ 1 = 1：𝜔 后面只有下标 𝑡（字号 6.6），没有 \"port\"。")
log("  公式区域截图见 attention factor/06_constraint_check.png：")
log("  约束上方那一行是 R̄^port_net 的定义，\"port\" 是 R̄ 的上标，不是 ω 的。")
log("  结论：[ATT] 写的是股票空间的 ‖ω_t‖₁ = 1，和 [19] 公式 (3)/(5)、III.C 节一致。")
log("  因此是“先映射回股票，再在股票空间做 L1 归一化”，不是对 ω^port 归一化。")
log("")
log("三、实现（af_model.py）")
log("  stock_weights(w_eps, w_port, ok):")
log("    1. 30 天窗口不完整（ok = False）的残差：ω^port 置 0。依据 [19] III.B 节")
log("       \"At each day we only consider the stocks with no missing observations ... within the")
log("       rolling window\"；但这些股票仍参与因子构造（见 04 的 log），属自行设定。")
log("    2. ω = (ω^ε)ᵀ ω^port：ω^ε 第 i 行是第 i 个残差组合，ω 是各残差组合按下注加权的和。")
log("    3. ω ← ω / ‖ω‖₁。某天全部为 0 时不除（自行设定，实际数据中未出现）。")
log("  portfolio_returns(w, R): R^port_t = R_tᵀ ω_{t-1}")
log("  block_forward(...): 一个月块走完环节 ①-⑥，返回当月每天的 ε、ok、ω^port、ω、R^port 等，")
log("    后面的环节 ⑦⑧ 和训练直接用。")
log("  HistBlocks / fit_normalizer: 读取 04 的按月块、拟合水平量标准化（原先在每个脚本里重复写的读取代码）")
log("")
log(f"四、检查结果：{n_pass} / {len(checks)} 项通过")
log(f"  （模型 K = {K_CHECK}，参数随机初始化，λ_ridge = {proj.ridge:g}，演示月份 {DEMO_MONTH}，"
    f"{n_days} 个交易日）")
log("")
for item, ok_, val in checks:
    log(f"  [{'通过' if ok_ else '失败'}] {item}")
    log(f"         {val}")
log("")
log("  说明：")
log("    检查 5：R_t = βᵀF_t + ε_t，组合收益 = ωᵀβᵀF_t（因子部分）+ ωᵀε_t。λ_ridge = 0 时")
log("    ω^F βᵀ = I，ω^ε βᵀ = 0，任何 ω^port 映射回来的组合都完全不承担这 K 个因子的风险（套利组合）。")
log("    λ_ridge > 0 时只是近似中性。随机初始化下各因子几乎相同（见 03：有效股票数 ≈ 500，")
log("    ω^F ω^Fᵀ 接近奇异），ridge 起主导作用，因子部分占组合收益的比例：")
log("        λ_ridge   " + "  ".join(f"{lam:>7g}" for lam in [0.0, 1e-6, 1e-5, 1e-4, 1e-3]))
for K in [8, 30]:
    log(f"        K = {K:<4d}  " + "  ".join(f"{share_tab[(K, lam)]:7.1%}" for lam in [0.0, 1e-6, 1e-5, 1e-4, 1e-3]))
log("    训练后各因子会分开，ω^F ω^Fᵀ 不再接近奇异，ridge 的影响会变小；训练时在验证集上比较 λ_ridge（见 03）。")
log("    一开始这项检查写成“|βω| 远小于等权组合”，在 λ = 1e-5 下没通过（1.8 vs 0.13）：β 在 ω^Fω^Fᵀ 接近")
log("    奇异时数值很大，直接比 β 的大小没有意义，所以改成比较收益里的因子部分，并在 λ = 0 下检查精确性质。")
log("    检查 7 说明 ω_{t-1} 只用到 t-1 日及以前的信息（X_{t-1} 和 ε_{t-30..t-1}）。")
log("")
log("五、补充：如果只在残差空间归一化（‖ω^port‖₁ = 1），股票空间的 ‖ω‖₁ 会是多少")
log("  （随机初始化，1998-01 各交易日）")
for K, v in alt_by_K.items():
    log(f"    K = {K:<3d}  股票空间 ‖ω‖₁ 范围 [{v.min():.3f}, {v.max():.3f}]，均值 {v.mean():.3f}")
log("  不等于 1，而且随 K 和日期变化：每个残差组合本身是一个多空组合（自己 +1，减去因子暴露），")
log("  不同残差组合在同一只股票上的权重会相互抵消或叠加。")
log("  如果在残差空间归一化，实际杠杆每天都不一样，Sharpe 和交易成本都无法比较，所以必须在股票空间归一化。")
log("")
log("六、随机初始化下的权重长什么样（未训练，只看量级）")
log(f"  多头 Σmax(ω,0)：均值 {long_l1.mean():.3f}；空头 Σmax(-ω,0)：均值 {short_l1.mean():.3f}"
    f"（两者相加 = 1）")
log(f"  净敞口 Σω：范围 [{net_exp.min():+.3f}, {net_exp.max():+.3f}]（原文没有要求多空相等）")
log(f"  每天非零权重的股票数：{n_nz.min()}-{n_nz.max()}")
log(f"  单只股票最大 |ω|：{np.abs(wn).max():.4f}")
log(f"  组合日收益 R^port：均值 {r_np.mean():+.5f}，标准差 {r_np.std():.5f}")
log("")
log("七、下一环节")
log("  环节 ⑦：交易成本 cost = 0.0005‖ω_t - ω_{t-1}‖₁ + 0.0001‖max(-ω_t, 0)‖₁，按 PERMNO 对齐")
log("")
log(f"总耗时 {time.time() - t0:.0f}s")

with open(LOG_PATH, "w") as fh:
    fh.write("\n".join(LOG) + "\n")
