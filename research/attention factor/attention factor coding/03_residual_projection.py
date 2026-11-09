"""
环节 ③：载荷 β 与残差投影 ω^ε —— 检查脚本（Attention Factors for Statistical Arbitrage 复现）

原文：[ATT] 3.2 节 "Residual Portfolios" 与 "Attention Factors"
    β_{t-1}ᵀ = ω^Fᵀ (ω^F ω^Fᵀ + λ_ridge I_K)^{-1}
    ω^ε_{t-1} = I_N - β_{t-1}ᵀ ω^F_{t-1}
    ε_t = ω^ε_{t-1} R_t

这个脚本不训练，只验证环节 ③ 的计算是否正确：
  Part A  对每个 K、满员日和有空位的日，检查形状、与手算一致、残差的两种算法一致、
          空位处理、梯度能传回 Q 和 W_K
  Part B  验证公式的数学性质（λ = 0 时 β 是 ω^F 的伪逆："因子对自己的暴露 = 1"，
          ω^ε 是正交投影）
  Part C  λ_ridge 的取值：原文未给出，比较不同 λ 下的数值稳定性与偏差

输入：attention factor/ 下的 X.npy, R.npy, mask.npy, dates.npy, feature_cols.json
输出：attention factor/03_residual_projection.log

用法（在 research 目录下）：
    python "attention factor/attention factor coding/03_residual_projection.py"
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
AF_DIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from af_model import (N_FEAT, RIDGE_DEFAULT, AttentionFactorWeights,  # noqa: E402
                      FeatureEmbedding, LevelNormalizer, ResidualProjection)

LOG_PATH = os.path.join(AF_DIR, "03_residual_projection.log")
K_LIST = [1, 3, 5, 8, 15, 30, 100]
D_EMB = 32
FIRST_TRAIN = ("1990-01-01", "1997-12-31")
DEMO_DAY = "1998-01-02"
SEED = 0
RIDGE_GRID = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
Q_SCALES = [1, 30]       # 1 = 初始化状态；30 = 把 Q 放大 30 倍，模拟训练后权重集中的情形

t0 = time.time()
L = []


def log(msg=""):
    print(msg)
    L.append(str(msg))


X = np.load(os.path.join(AF_DIR, "X.npy"), mmap_mode="r")
R = np.load(os.path.join(AF_DIR, "R.npy"), mmap_mode="r")
mask = np.load(os.path.join(AF_DIR, "mask.npy"))
dts = pd.DatetimeIndex(np.load(os.path.join(AF_DIR, "dates.npy")))
feat_cols = json.load(open(os.path.join(AF_DIR, "feature_cols.json")))["feature_cols"]

tr = np.flatnonzero((dts >= FIRST_TRAIN[0]) & (dts <= FIRST_TRAIN[1]))
norm = LevelNormalizer(feat_cols).fit(np.asarray(X[tr[0]:tr[-1] + 1]))
n_stock = mask.sum(1)
d_full = int(np.flatnonzero(dts >= DEMO_DAY)[0])
d_pad = int(np.argmin(n_stock))
demo_days = {"满员日": d_full, "有空位的日": d_pad}


def t64(a):
    return torch.from_numpy(np.ascontiguousarray(a, dtype="float64"))


def build(K, dtype=torch.float64):
    torch.manual_seed(SEED)
    emb = FeatureEmbedding(N_FEAT, D_EMB).to(dtype)
    att = AttentionFactorWeights(K, D_EMB).to(dtype)
    return emb, att


# ===========================================================================
# Part A  正确性检查
# ===========================================================================
results = []
for K in K_LIST:
    emb, att = build(K)
    proj = ResidualProjection(RIDGE_DEFAULT)
    for day_name, d in demo_days.items():
        x, m, r = t64(norm.transform(X[d])), torch.from_numpy(mask[d]), t64(R[d])
        wF = att(emb(x), m)                              # (K, N)
        beta_T, w_eps = proj(wF, m)                      # (N, K), (N, N)

        def add(item, ok, val):
            results.append((K, day_name, item, bool(ok), val))

        add("1 形状", tuple(beta_T.shape) == (500, K) and tuple(w_eps.shape) == (500, 500),
            f"βᵀ {tuple(beta_T.shape)}，ω^ε {tuple(w_eps.shape)}")

        # 2 与手算一致：numpy 显式求逆
        wFn = wF.detach().numpy()
        bT_np = wFn.T @ np.linalg.inv(wFn @ wFn.T + RIDGE_DEFAULT * np.eye(K))
        we_np = (np.eye(500) - bT_np @ wFn) * np.outer(mask[d], mask[d])
        diff = max(np.abs(bT_np - beta_T.detach().numpy()).max(),
                   np.abs(we_np - w_eps.detach().numpy()).max())
        add("2 与手算公式一致", diff < 1e-8, f"最大差 {diff:.1e}")

        # 3 残差两种算法：ω^ε R  vs  R - βᵀ F（F = ω^F R）
        eps1 = (w_eps @ r).detach().numpy()
        F = wF @ r
        eps2 = ((r - beta_T @ F) * m).detach().numpy()
        diff3 = np.abs(eps1 - eps2).max()
        add("3 ω^εR = R - βᵀF", diff3 < 1e-10, f"最大差 {diff3:.1e}")

        # 4 空位：ω^ε 的空位行列全 0，空位残差 = 0
        pad = ~mask[d]
        we = w_eps.detach().numpy()
        pad_max = 0.0 if pad.sum() == 0 else max(np.abs(we[pad]).max(), np.abs(we[:, pad]).max(),
                                                   np.abs(eps1[pad]).max())
        add("4 空位行列与残差 = 0", pad_max == 0, f"空位 {pad.sum()} 个，最大值 {pad_max:.1e}")

        # 5 梯度：残差平方和对 Q、W_K 的梯度
        if day_name == "满员日":
            emb.zero_grad(), att.zero_grad()
            wF = att(emb(x), m)
            _, w_eps = proj(wF, m)
            ((w_eps @ r) ** 2).sum().backward()
            gq, gw = att.Q.grad.norm().item(), emb.W_K.weight.grad.norm().item()
            ok = gq > 0 and gw > 0 and np.isfinite(gq) and np.isfinite(gw)
            add("5 梯度传回 Q 和 W_K", ok, f"|∇Q| {gq:.2e}，|∇W_K| {gw:.2e}")

# float32 精度（训练时用 float32）：与 float64 结果的差
emb64, att64 = build(30)
emb32, att32 = build(30, torch.float32)
x, m = t64(norm.transform(X[d_full])), torch.from_numpy(mask[d_full])
_, we64 = ResidualProjection()(att64(emb64(x), m), m)
_, we32 = ResidualProjection()(att32(emb32(x.float()), m), m)
f32_diff = (we64 - we32.double()).abs().max().item()


# ===========================================================================
# Part B  数学性质（λ = 0）
# ===========================================================================
def props(K, q_scale, ridge, d=d_full):
    emb, att = build(K)
    with torch.no_grad():
        att.Q.mul_(q_scale)
        x, m, r = t64(norm.transform(X[d])), torch.from_numpy(mask[d]), t64(R[d])
        wF = att(emb(x), m)
        A0 = (wF @ wF.T).numpy()
        cond = np.linalg.cond(A0)
        if ridge == 0 and cond > 1e14:
            return dict(cond=cond, ok=False)
        beta_T, w_eps = ResidualProjection(ridge)(wF, m)
        wFn, bTn, we = wF.numpy(), beta_T.numpy(), w_eps.numpy()
        eps = we @ r.numpy()
        rn = r.numpy()
        return dict(
            cond=cond, ok=True,
            exposure=np.abs(wFn @ bTn - np.eye(K)).max(),     # ω^F βᵀ 与 I_K 的最大偏差
            factor_resid=np.abs(wFn @ we).max(),               # 因子组合自己的残差
            sym=np.abs(we - we.T).max(),                       # 对称
            idem=np.abs(we @ we - we).max(),                   # 幂等
            rank=np.linalg.matrix_rank(we, tol=1e-8),
            expl=1 - (eps ** 2).sum() / (rn[mask[d]] ** 2).sum(),   # 当日被因子解释的平方和比例
            eff_n=(1.0 / (wFn ** 2).sum(1)).mean(),
            beta_max=np.abs(bTn).max(),
        )


K_B = 8
pB = {s: props(K_B, s, 0.0) for s in Q_SCALES}

# ===========================================================================
# Part C  λ_ridge 敏感性
# ===========================================================================
pC = {(s, lam): props(K_B, s, lam) for s in Q_SCALES for lam in RIDGE_GRID}
pC30 = {lam: props(30, 30, lam) for lam in RIDGE_GRID}

# ===========================================================================
# 写 log
# ===========================================================================
n_pass = sum(r[3] for r in results)
log("=" * 78)
log("环节 ③：载荷 β 与残差投影 ω^ε —— 检查结果")
log("=" * 78)
log("")
log("一、原文")
log("  [ATT] 3.2 节 \"Residual Portfolios\"：因子是可交易组合 F_t = ω^F_{t-1} R_t，因此残差也是可交易组合")
log("        ε_t = R_t - β_{t-1}ᵀ F_t = (I_N - β_{t-1}ᵀ ω^F_{t-1}) R_t = ω^ε_{t-1} R_t")
log("  [ATT] 3.2 节 \"Attention Factors\"：\"Factor loadings β_{t-1} and weights ω^F_{t-1} are")
log("        mechanically related, as up to a rotation the loadings represent factor portfolio weights")
log("        (see, for example, [28]). We obtain the factor loadings as")
log("        β_{t-1}ᵀ = ω^Fᵀ (ω^F ω^Fᵀ + λ_ridge I_K)^{-1}, where we add a ridge penalty λ_ridge for")
log("        stability. Hence, the estimation of ω^F directly implies β_{t-1} and ω^ε_{t-1}.\"")
log("")
log("二、实现（af_model.py 中的 ResidualProjection，无可学习参数）")
log("  1. A = ω^F ω^Fᵀ + λ I_K（K × K）")
log("  2. βᵀ = ω^Fᵀ A^{-1}（N × K）。代码用 linalg.solve(A, ω^F) 再转置，与显式求逆结果相同、数值更稳定。")
log("  3. ω^ε = I_N - βᵀ ω^F（N × N）")
log("  自行设定（原文未说明）：")
log("  - 空位股票在 ω^ε 中的行和列置 0。原公式的 I_N 会在空位对角线上留下 1，")
log("    置 0 后空位残差恒为 0，也不影响其他股票。")
log(f"  - λ_ridge = {RIDGE_DEFAULT:g}。[ATT] 正文和附录 Table 4 都没有给出 λ_ridge；作者没有公开代码；")
log("    [28] Hao Ma (2021) 中也没有找到。唯一找到的取值来自第三方复现：")
log("    BSIC (2025) \"Attention Factors for Statistical Arbitrage\"（https://bsic.it/attention-factors/）：")
log("    \"The ridge term λ_ridge = 10⁻³ prevents the Gram matrix from being singular.\"")
log("    这不是原作者给出的值。Part C 显示 1e-3 对本数据的权重尺度偏大（因子对自己的暴露只剩")
log("    0.1-0.3），因此本代码改用 λ = 1e-5，理由见 Part C 结论。")
log("")
log("三、公式背后的数学：为什么说\"因子对自己的暴露 = 1\"")
log("  1. 因子模型：每只股票的收益 R_i = β_iᵀ F + ε_i，β_i 是股票 i 对 K 个因子的暴露。")
log("  2. 因子本身是股票组合：F = ω^F R。把第 k 个因子组合也当成一个\"资产\"，它对因子的")
log("     暴露是 ω^F_k βᵀ（ω^F 第 k 行乘 βᵀ）。合理的要求是：")
log("       第 k 个因子组合对第 k 个因子的暴露 = 1，对其他因子的暴露 = 0，即 ω^F βᵀ = I_K。")
log("     否则\"因子\"连自己都解释不了，残差里就会混进因子本身的波动。")
log("  3. ω^F βᵀ = I_K 有无穷多个解（K 个方程，N×K 个未知数），其中\"最小\"的那个是伪逆：")
log("       βᵀ = ω^Fᵀ (ω^F ω^Fᵀ)^{-1}")
log("     验证：ω^F βᵀ = ω^F ω^Fᵀ (ω^F ω^Fᵀ)^{-1} = I_K。这就是原文公式在 λ = 0 时的形式。")
log("  4. 由此可以推出：")
log("     - 因子组合自己的残差为 0：ω^F ω^ε = ω^F - (ω^F βᵀ) ω^F = ω^F - ω^F = 0")
log("     - P = βᵀ ω^F = ω^Fᵀ (ω^F ω^Fᵀ)^{-1} ω^F 是把收益投影到\"因子权重张成的空间\"的正交投影，")
log("       ω^ε = I - P 是投影到其正交补空间：对称、幂等（ω^ε ω^ε = ω^ε）、秩 = N - K。")
log("       直观地说：ω^ε 把每天的收益向量拆成\"因子能解释的部分\"和\"剩下的部分\"，只留下后者。")
log("  5. 为什么要加 λ：当 K 个因子的权重很像时（例如刚初始化时都接近等权），ω^F ω^Fᵀ")
log("     接近奇异（条件数极大），求逆会\"爆炸\"。加 λ I 相当于在分母上加一个小数，")
log("     代价是 ω^F βᵀ 不再严格等于 I_K，而是略小于 I_K（偏差大小见 Part C）。")
log("")
log("四、Part A 正确性检查")
log(f"  满员日 {dts[d_full].date()}（{n_stock[d_full]} 只），有空位的日 {dts[d_pad].date()}（{n_stock[d_pad]} 只）")
log(f"  λ = {RIDGE_DEFAULT:g}，W_K、Q 随机初始化（种子 {SEED}），用 float64 计算以便精确比较")
log(f"  结果：{n_pass} / {len(results)} 项通过")
log("")
log(f"  {'K':>4}  {'检查日':<8}  {'检查项':<20}  {'结果':<4}  数值")
for K, day_name, item, ok, val in results:
    log(f"  {K:>4}  {day_name:<8}  {item:<20}  {'通过' if ok else '失败':<4}  {val}")
log("")
log(f"  float32 与 float64 计算的 ω^ε 最大差（K = 30）: {f32_diff:.1e}（训练时用 float32，可以接受）")
log("")
log(f"五、Part B 数学性质验证（K = {K_B}，满员日，λ = 0）")
log("  Q 放大倍数 = 1：初始化状态，因子权重接近等权；= 30：模拟训练后权重集中到部分股票")
log(f"  {'Q 放大':>6}  {'因子有效股票数':>12}  {'条件数':>10}  {'|ω^Fβᵀ-I|':>10}  {'|ω^Fω^ε|':>10}  "
    f"{'对称偏差':>9}  {'幂等偏差':>9}  {'秩':>5}")
for s in Q_SCALES:
    p = pB[s]
    if not p["ok"]:
        log(f"  {s:>6}  {'':>12}  {p['cond']:>10.1e}  条件数过大，λ = 0 时无法稳定求逆")
        continue
    log(f"  {s:>6}  {p['eff_n']:>12.1f}  {p['cond']:>10.1e}  {p['exposure']:>10.1e}  "
        f"{p['factor_resid']:>10.1e}  {p['sym']:>9.1e}  {p['idem']:>9.1e}  {p['rank']:>5d}")
log(f"  预期：|ω^Fβᵀ-I| ≈ 0（因子对自己暴露 = 1），|ω^Fω^ε| ≈ 0（因子组合残差 = 0），")
log(f"        对称、幂等偏差 ≈ 0，秩 = N - K = {500 - K_B}")
log("  初始化时条件数很大，偏差会被放大（这正是需要 λ 的原因）；权重分散开后性质精确成立。")
log("")
log(f"六、Part C λ_ridge 的影响（满员日）")
for s in Q_SCALES:
    log(f"  K = {K_B}，Q 放大 {s} 倍（因子有效股票数 {pC[(s, RIDGE_GRID[-1])]['eff_n']:.1f}，"
        f"λ=0 时条件数 {pC[(s, 0.0)]['cond']:.1e}）")
    log(f"    {'λ':>8}  {'|ω^Fβᵀ-I|':>10}  {'最大|β|':>10}  {'当日因子解释比例':>14}")
    for lam in RIDGE_GRID:
        p = pC[(s, lam)]
        if not p["ok"]:
            log(f"    {lam:>8g}  {'无法求逆':>10}")
            continue
        log(f"    {lam:>8g}  {p['exposure']:>10.1e}  {p['beta_max']:>10.1f}  {p['expl']:>14.2%}")
log(f"  K = 30，Q 放大 30 倍（因子有效股票数 {pC30[RIDGE_GRID[-1]]['eff_n']:.1f}，"
    f"λ=0 时条件数 {pC30[0.0]['cond']:.1e}）")
log(f"    {'λ':>8}  {'|ω^Fβᵀ-I|':>10}  {'最大|β|':>10}  {'当日因子解释比例':>14}")
for lam in RIDGE_GRID:
    p = pC30[lam]
    if not p["ok"]:
        log(f"    {lam:>8g}  {'无法求逆':>10}")
        continue
    log(f"    {lam:>8g}  {p['exposure']:>10.1e}  {p['beta_max']:>10.1f}  {p['expl']:>14.2%}")
log("  说明：")
log("  - |ω^Fβᵀ-I| 越大，\"因子对自己暴露 = 1\"偏离得越多，残差里残留的因子成分越多。")
log("  - 最大|β| 很大说明求逆不稳定（初始化时因子几乎相同，λ 太小会让 β 爆炸）。")
log("  - 当日因子解释比例 = 1 - Σε² / ΣR²，K 个因子解释了当天收益平方和的多少。")
log("  - 结论：λ 的大小必须和 ω^Fω^Fᵀ 的量级比较。每个因子的权重加总为 1、分布在约 500 只股票上，")
log("    ω^Fω^Fᵀ 的元素只有约 1/有效股票数 ≈ 0.002-0.003，最小特征值更小。")
log("    因此 BSIC 的 λ = 1e-3 并不\"小\"：上表中它让 |ω^Fβᵀ-I| 达到 0.66-0.88，")
log("    即因子对自己的暴露只剩 0.1-0.3（而不是 1），当日因子解释比例也明显下降。")
log("    λ = 1e-6 ~ 1e-5 时偏差小得多，但初始化时 β 会变大（最大 |β| 14-74）。")
log("    BSIC 的 1e-3 可能对应不同的权重尺度；论文 Figure 4 中前 10 只股票占因子权重的 10%-23%，")
log("    说明训练后权重比这里的模拟集中得多，ω^Fω^Fᵀ 的特征值也会更大，λ 的影响随之变小。")
log("    本代码采用 λ = 1e-5 作为默认值：初始化时 β 仍然有限（最大 |β| 约 14），")
log("    权重分散开后因子对自己的暴露接近 1（K=8 时 |ω^Fβᵀ-I| = 0.03）。")
log("    训练时仍把 λ 作为超参数，在验证集上比较 {1e-6, 1e-5, 1e-4, 1e-3}。")
log("")
log("七、下一环节")
log("  环节 ④：LongConv 交易信号——用过去 30 天的残差 ε_{t-30..t-1}（用 ω^ε_{t-1} 计算）")
log("  生成残差空间的组合权重 ω^port_{t-1}")
log("")
log(f"总耗时 {time.time() - t0:.0f}s")

with open(LOG_PATH, "w") as fh:
    fh.write("\n".join(L) + "\n")

if n_pass != len(results):
    raise SystemExit(f"有 {len(results) - n_pass} 项检查未通过，见 {LOG_PATH}")
