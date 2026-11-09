"""
环节 ⑤：LongConv 交易信号 —— 检查脚本（Attention Factors for Statistical Arbitrage 复现）

原文：
  [ATT] 3.3 节：ω^port_{i,t-1} = LongConv_θ(ε_{i,(t-s,t-1)})；"our optimally tuned model has 32
        different convolutions"
  [ATT] 附录 A：y = K * u + D ⊙ u（FFT 计算），Squash 算子，几何衰减初始化
  [ATT] 附录 B Table 4：1 层，隐藏维度 32，dropout 0.1，λ_squash = 0.001
  [15]  Fu, Epstein, et al. (2023) Simple Hardware-Efficient Long Convolutions for Sequence Modeling，
        官方代码 https://github.com/HazyResearch/safari：
        standalone_long_convs.py（LongConv 层与 LongConvModel 整体结构）
        src/models/sequence/long_conv_kernel.py（"double_exp" 初始化、squash）

这个脚本不训练，只验证环节 ⑤ 的计算是否正确（用 1998 年 1 月的真实残差窗口，参数随机初始化）。

输入：attention factor/ 下 hist_*.npy、X.npy、dates.npy、feature_cols.json
输出：attention factor/05_longconv.log

用法（在 research 目录下）：
    python "attention factor/attention factor coding/05_longconv.py"
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
from af_model import (DROPOUT, LAMBDA_SQUASH, LOOKBACK, N_FEAT, AttentionFactorWeights,  # noqa: E402
                      FeatureEmbedding, LevelNormalizer, LongConvSignal, ResidualProjection,
                      daily_residuals, geom_decay_kernel, residual_windows)

LOG_PATH = os.path.join(AF_DIR, "05_longconv.log")
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


# ---------------------------------------------------------------------------
# 读数据：1998 年 1 月块，经过环节 ①-④ 得到累计残差窗口
# ---------------------------------------------------------------------------
hist_X = np.load(os.path.join(AF_DIR, "hist_X.npy"), mmap_mode="r")
hist_R = np.load(os.path.join(AF_DIR, "hist_R.npy"), mmap_mode="r")
hist_idx = np.load(os.path.join(AF_DIR, "hist_idx.npy"), mmap_mode="r")
hist_ptr = np.load(os.path.join(AF_DIR, "hist_ptr.npy"))
months = json.load(open(os.path.join(AF_DIR, "hist_months.json")))
feat_cols = json.load(open(os.path.join(AF_DIR, "feature_cols.json")))["feature_cols"]
X01 = np.load(os.path.join(AF_DIR, "X.npy"), mmap_mode="r")
d01 = pd.DatetimeIndex(np.load(os.path.join(AF_DIR, "dates.npy")))
tr = np.flatnonzero((d01 >= FIRST_TRAIN[0]) & (d01 <= FIRST_TRAIN[1]))
norm = LevelNormalizer(feat_cols).fit(np.asarray(X01[tr[0]:tr[-1] + 1]))

b = months.index(DEMO_MONTH)
idx = np.asarray(hist_idx[hist_ptr[b]:hist_ptr[b + 1]])
Xb = torch.from_numpy(norm.transform(np.asarray(hist_X)[idx]))
Rb = torch.from_numpy(np.asarray(hist_R)[idx])
Vb = torch.from_numpy(idx > 0)

torch.manual_seed(SEED)
emb, att, proj = FeatureEmbedding(N_FEAT, D_EMB), AttentionFactorWeights(K_CHECK, D_EMB), ResidualProjection()
with torch.no_grad():
    eps, _ = daily_residuals(emb, att, proj, Xb, Rb, Vb)
    x, ok = residual_windows(eps, Vb)                    # (交易日, 500, 30)

checks = []


def add(item, ok_, val):
    checks.append((item, bool(ok_), val))


torch.manual_seed(SEED)
net = LongConvSignal()
net.eval()                                              # 检查计算时关闭 dropout
layer = net.layers[0]

# ---- 1 形状 ---------------------------------------------------------------
with torch.no_grad():
    w_port = net(x)
add("1 形状：(交易日, 500, 30) -> (交易日, 500)", tuple(w_port.shape) == tuple(x.shape[:2]),
    f"{tuple(x.shape)} -> {tuple(w_port.shape)}")

# ---- 2 FFT 卷积 = 直接按定义求和 ----------------------------------------------
# (K̄ * u)[i] = Σ_{j ≤ i} u[j] K̄[i-j]，再加 D ⊙ u
torch.manual_seed(1)
u = torch.randn(3, 32, L, dtype=torch.float64)
lay64 = type(layer)(32, L).double()
with torch.no_grad():
    y_fft = lay64.conv(u).numpy()
    kb = lay64.squashed_kernel().numpy()
    Dv = lay64.D.numpy()
    un = u.numpy()
    y_dir = np.zeros_like(un)
    for i in range(L):
        for j in range(i + 1):
            y_dir[..., i] += un[..., j] * kb[:, i - j]
    y_dir += un * Dv[None, :, None]
d2 = np.abs(y_fft - y_dir).max()
add("2 FFT 卷积 = 按定义直接求和", d2 < 1e-10, f"最大差 {d2:.1e}")

# ---- 3 因果：改动第 j 天，只影响第 j 天及以后的卷积输出 ------------------------------
j = 12
u2 = u.clone()
u2[..., j] += 1.0
with torch.no_grad():
    dy = (lay64.conv(u2) - lay64.conv(u)).abs().amax(dim=(0, 1)).numpy()
add("3 卷积是因果的（只用之前的天；FFT 舍入误差 < 1e-12 视为 0）", dy[:j].max() < 1e-12 and dy[j:].min() > 1e-6,
    f"改动第 {j + 1} 天：之前各天变化最大 {dy[:j].max():.1e}，之后各天变化最小 {dy[j:].min():.1e}")

# ---- 4 Squash 算子 ------------------------------------------------------------
k_raw = layer.kernel.detach().numpy()
k_sq = layer.squashed_kernel().detach().numpy()
small = np.abs(k_raw) <= LAMBDA_SQUASH
expect = np.sign(k_raw) * np.maximum(np.abs(k_raw) - LAMBDA_SQUASH, 0)
add("4 Squash：|K| ≤ λ 的置 0，其余向 0 收缩 λ", np.abs(k_sq - expect).max() < 1e-7 and (k_sq[small] == 0).all(),
    f"λ = {LAMBDA_SQUASH}，置 0 的比例 {small.mean():.2%}，最大差 {np.abs(k_sq - expect).max():.1e}")

# ---- 5 几何衰减初始化 ------------------------------------------------------------
gen = torch.Generator().manual_seed(123)
Kg = geom_decay_kernel(32, L, generator=gen).numpy()
gen = torch.Generator().manual_seed(123)
xg = torch.randn(32, L, generator=gen).numpy()
tt, hh = np.arange(1, L + 1), np.arange(1, 33)
Kf = xg * np.exp(-(tt[None, :] / L) * (32 / 2) ** (hh[:, None] / 32))
add("5 初始化 = 附录 A 公式 x·exp(-(t/T)(d/2)^(h/d))", np.abs(Kg - Kf).max() < 1e-6,
    f"最大差 {np.abs(Kg - Kf).max():.1e}")
decay_slow = np.exp(-(L / L) * (32 / 2) ** (1 / 32))      # h = 1 在 t = T 处的衰减系数
decay_fast = np.exp(-(L / L) * (32 / 2) ** (32 / 32))     # h = d 在 t = T 处的衰减系数

# ---- 6 所有股票共用同一组参数：交换股票顺序，输出跟着交换 ---------------------------
perm = torch.randperm(x.shape[1], generator=torch.Generator().manual_seed(7))
with torch.no_grad():
    d6 = (net(x[:, perm]) - w_port[:, perm]).abs().max().item()
add("6 交换股票顺序，输出跟着交换（参数共享）", d6 < 1e-6, f"最大差 {d6:.1e}")

# ---- 7 dropout：训练模式随机、评估模式确定 ----------------------------------------
net.train()
with torch.no_grad():
    a1, a2 = net(x), net(x)
net.eval()
with torch.no_grad():
    e1, e2 = net(x), net(x)
add("7 dropout：训练模式两次输出不同、评估模式相同",
    (a1 - a2).abs().max() > 0 and (e1 - e2).abs().max() == 0,
    f"训练模式差 {(a1 - a2).abs().max():.1e}，评估模式差 {(e1 - e2).abs().max():.1e}")

# ---- 8 梯度：从 ω^port 一路传回 LongConv、Q、W_K ----------------------------------
torch.manual_seed(SEED)
emb_g, att_g = FeatureEmbedding(N_FEAT, D_EMB), AttentionFactorWeights(K_CHECK, D_EMB)
net_g = LongConvSignal()
e_g, _ = daily_residuals(emb_g, att_g, proj, Xb, Rb, Vb)
x_g, ok_g = residual_windows(e_g, Vb)
wp = net_g(x_g)
(wp[ok_g] ** 2).mean().backward()
grads = {"LongConv 卷积核 K": net_g.layers[0].kernel.grad, "LongConv D": net_g.layers[0].D.grad,
         "编码层": net_g.encoder.weight.grad, "解码层": net_g.decoder.weight.grad,
         "Q": att_g.Q.grad, "W_K": emb_g.W_K.weight.grad}
gn = {k: (v.norm().item() if v is not None else 0.0) for k, v in grads.items()}
add("8 梯度传到所有可学习参数", all(v > 0 for v in gn.values()),
    "，".join(f"{k} {v:.1e}" for k, v in gn.items()))

# ---- 参数量与耗时 ------------------------------------------------------------------
n_params = {name: sum(p.numel() for p in mod.parameters()) for name, mod in
            [("编码层 Linear(1,32)", net.encoder), ("LongConv 层", net.layers[0]),
             ("LayerNorm", net.norms[0]), ("解码层 Linear(32,1)", net.decoder)]}
timing = {}
for K in [8, 30, 100]:
    torch.manual_seed(SEED)
    e_, a_, n_ = FeatureEmbedding(N_FEAT, D_EMB), AttentionFactorWeights(K, D_EMB), LongConvSignal()
    t1 = time.time()
    for _ in range(3):
        ee, _ = daily_residuals(e_, a_, proj, Xb, Rb, Vb)
        xx, oo = residual_windows(ee, Vb)
        (n_(xx)[oo] ** 2).mean().backward()
    timing[K] = (time.time() - t1) / 3

with torch.no_grad():
    wp_ok = w_port[ok].numpy()

# ===========================================================================
# 写 log
# ===========================================================================
n_pass = sum(c[1] for c in checks)
log("=" * 78)
log("环节 ⑤：LongConv 交易信号 —— 检查结果")
log("=" * 78)
log("")
log("一、原文")
log("  [ATT] 3.3 节: \"ω^port_{i,t-1} = LongConv_θ(ε_{i,(t-s,t-1)}) ... We chose it because of its")
log("        linear scaling in the sequence length and simplicity ... our optimally tuned model has 32")
log("        different convolutions.\"")
log("  [ATT] 附录 A: y = K * u + D ⊙ u，u ∈ R^{N×d×T}，K ∈ R^{d×T}，D ∈ R^d；用 FFT 计算；")
log("        Squash: K̄ = sign(K) ⊙ max(|K| - λ_squash, 0)；")
log("        初始化: K^(h)_t = x exp(-(t/T)(d/2)^(h/d))，x ~ N(0,1)")
log("  [ATT] 附录 B Table 4: Hidden dim d = 32，Dropout 0.1，Nr layers 1，λ_squash = 0.001，")
log("        LongConv init = Geom Decay，Weight decay 0.05（训练时用）")
log("")
log("二、原文没写、按 [15] 官方代码补上的部分")
log("  [15] Fu, Epstein, et al. (2023)，官方代码 https://github.com/HazyResearch/safari")
log("  1. 输入层（1 个数 -> 32 个通道）: standalone_long_convs.py 的 LongConvModel:")
log("       self.encoder = nn.Linear(d_input, d_model)；这里 d_input = 1（每天的累计残差）")
log("  2. 一层 LongConv 内部（LongConv.forward）: 卷积 + D 跳跃连接 -> GELU -> Dropout ->")
log("       output_linear = Linear(d, 2d) + GLU")
log("  3. 残差块（LongConvModel.forward）: z = layer(x)；z = Dropout(z)；x = z + x；LayerNorm（post-norm）")
log("  4. 输出层（32 个通道 -> 1 个权重）: \"Pooling: average pooling over the sequence length\"")
log("       x = x.mean(dim=1)，再 self.decoder = nn.Linear(d_model, d_output)；这里 d_output = 1")
log("  5. dropout 位置: 上面第 2、3 步各一处（[15] 代码），比率用 [ATT] 的 0.1")
log("  6. 因果卷积: FFT 补零到 2T 后取前 T 个（[15] 代码 irfft(..., n=2L)[..., :L]），")
log("       第 i 天的卷积输出只用第 1..i 天")
log("  7. 每个隐藏维度一个卷积核（channels = 1，[15] long_conv.py 的默认值），共 32 个卷积核，")
log("       对应 [ATT] \"32 different convolutions\"")
log("  与 [15] 代码不同、按 [ATT] 原文的地方：")
log("  - 初始化随机数 x ~ N(0,1)（[15] 代码的 double_exp 用 0.02·N(0,1)），下标从 1 开始")
log(f"  - λ_squash = {LAMBDA_SQUASH}（[15] 代码默认 0.1）")
log("")
log("三、本环节在整个模型中的位置")
log("  输入: 环节 ④ 的累计残差窗口 (交易日, 500, 30)，每只股票的 30 天残差\"价格\"路径")
log("  每只股票的序列独立地经过同一个 LongConv（所有股票、所有交易日共用一组参数）")
log("  输出: ω^port (交易日, 500)，每只股票的残差组合上下多大的注（正 = 买入，负 = 卖出），")
log("        尚未归一化；环节 ⑥ 映射回股票并做 L1 归一化。")
log("  参数量: " + "，".join(f"{k} {v:,}" for k, v in n_params.items())
    + f"，共 {sum(n_params.values()):,}")
log("")
log(f"四、检查结果：{n_pass} / {len(checks)} 项通过")
log(f"  （数据: {DEMO_MONTH} 块，K = {K_CHECK}；所有参数随机初始化，未训练）")
log("")
for item, ok_, val in checks:
    log(f"  [{'通过' if ok_ else '失败'}] {item}")
    log(f"         {val}")
log("")
log("  补充：几何衰减初始化的含义")
log(f"    第 1 个卷积核在第 30 天的衰减系数 {decay_slow:.3f}（衰减慢，看长期），")
log(f"    第 32 个卷积核在第 30 天的衰减系数 {decay_fast:.2e}（衰减快，只看最近几天）。")
log(f"  补充：随机初始化下 ω^port 的分布（{DEMO_MONTH}，可交易的股票）："
    f"均值 {wp_ok.mean():+.4f}，标准差 {wp_ok.std():.4f}，范围 [{wp_ok.min():+.3f}, {wp_ok.max():+.3f}]")
log("")
log("五、计算耗时（一个月块，环节 ①-⑤ 前向 + 反向，float32，CPU）")
for K, sec in timing.items():
    log(f"  K = {K:>3}: {sec:.2f} 秒 / 月块；8 年训练窗口（96 个月块）约 {sec * 96:.0f} 秒 / epoch，"
        f"30 个 epoch 约 {sec * 96 * 30 / 60:.0f} 分钟")
log("")
log("六、下一环节")
log("  环节 ⑥：映射回股票 ω = (ω^ε)ᵀ ω^port，再在股票空间做 L1 归一化 ‖ω‖₁ = 1（[19] 公式 (3)(5)）")
log("")
log(f"总耗时 {time.time() - t0:.0f}s")

with open(LOG_PATH, "w") as fh:
    fh.write("\n".join(LOG) + "\n")

if n_pass != len(checks):
    raise SystemExit(f"有 {len(checks) - n_pass} 项检查未通过，见 {LOG_PATH}")
