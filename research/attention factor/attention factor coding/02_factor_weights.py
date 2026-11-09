"""
环节 ②：注意力因子权重 —— 检查脚本（Attention Factors for Statistical Arbitrage 复现）

原文：[ATT] 3.2 节，公式 (1)
    ω^F_{t-1} = Softmax(Q X̃ᵀ_{t-1} / √d),  Q ∈ R^{K×d}，Softmax 沿每一行（每个因子）在股票上做

这个脚本不训练，只验证环节 ② 的计算是否正确：
  用真实数据（1998 年第一个交易日的 500 只股票，以及一个股票不足 500 只的交易日）
  经过 ① 嵌入、② 注意力，得到 ω^F，逐项检查：
    1. 形状是 K × N
    2. 所有权重 ≥ 0
    3. 每一行加总 = 1
    4. 空位股票权重 = 0
    5. 与按公式手算（numpy）的结果一致
    6. 多天批量计算与逐天计算一致
    7. 梯度能传回 Q 和 W_K
  对论文用到的每个 K（1, 3, 5, 8, 15, 30, 100）都检查一遍。
  注意：W_K 和 Q 都是随机初始化、未训练的，ω^F 的具体数值没有经济含义。

输入：attention factor/ 下的 X.npy, R.npy, mask.npy, permno.npy, dates.npy, feature_cols.json
输出：attention factor/02_factor_weights.log

用法（在 research 目录下）：
    python "attention factor/attention factor coding/02_factor_weights.py"
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
from af_model import (N_FEAT, AttentionFactorWeights, FeatureEmbedding,  # noqa: E402
                      LevelNormalizer)

LOG_PATH = os.path.join(AF_DIR, "02_factor_weights.log")
K_LIST = [1, 3, 5, 8, 15, 30, 100]       # [ATT] Table 2 的因子个数
D_EMB = 32                               # [ATT] 附录 B Table 4: d_x = 32
FIRST_TRAIN = ("1990-01-01", "1997-12-31")
DEMO_DAY = "1998-01-02"
SEED = 0
TOL = 1e-5

t0 = time.time()
L = []


def log(msg=""):
    print(msg)
    L.append(str(msg))


# ---------------------------------------------------------------------------
# 读数据（X.npy 用内存映射，只读需要的几天）
# ---------------------------------------------------------------------------
X = np.load(os.path.join(AF_DIR, "X.npy"), mmap_mode="r")
R = np.load(os.path.join(AF_DIR, "R.npy"), mmap_mode="r")
mask = np.load(os.path.join(AF_DIR, "mask.npy"))
permno = np.load(os.path.join(AF_DIR, "permno.npy"), mmap_mode="r")
dts = pd.DatetimeIndex(np.load(os.path.join(AF_DIR, "dates.npy")))
feat_cols = json.load(open(os.path.join(AF_DIR, "feature_cols.json")))["feature_cols"]

tr = np.flatnonzero((dts >= FIRST_TRAIN[0]) & (dts <= FIRST_TRAIN[1]))
norm = LevelNormalizer(feat_cols).fit(np.asarray(X[tr[0]:tr[-1] + 1]))

d_full = int(np.flatnonzero(dts >= DEMO_DAY)[0])           # 500 只股票的一天
n_stock = mask.sum(1)
d_pad = int(np.argmin(n_stock))                             # 股票最少的一天（有空位）
demo_days = {"满员日": d_full, "有空位的日": d_pad}


def to_t(a):
    return torch.from_numpy(np.ascontiguousarray(a))


def np_softmax_rows(s, valid):
    """手算：只在有股票的位置上做 Softmax，空位权重为 0"""
    s = np.where(valid[None, :], s, -np.inf)
    s = s - s.max(axis=1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# 逐个 K 检查
# ---------------------------------------------------------------------------
results = []          # (K, 检查日, 检查项, 是否通过, 数值)
examples = {}

for K in K_LIST:
    torch.manual_seed(SEED)
    emb = FeatureEmbedding(N_FEAT, D_EMB)
    att = AttentionFactorWeights(K, D_EMB)

    for day_name, d in demo_days.items():
        x = to_t(norm.transform(X[d]))                     # (500, 79)
        m = to_t(mask[d])                                  # (500,)
        n_valid = int(m.sum())

        x_tilde = emb(x)                                   # (500, 32)
        w = att(x_tilde, m)                                # (K, 500)
        wn = w.detach().numpy()

        def add(item, ok, val):
            results.append((K, day_name, item, bool(ok), val))

        add("1 形状 = K × N", tuple(w.shape) == (K, 500), str(tuple(w.shape)))
        add("2 权重全部 ≥ 0", (wn >= 0).all(), f"最小值 {wn.min():.2e}")
        row_sum = wn.sum(1)
        add("3 每行加总 = 1", np.abs(row_sum - 1).max() < TOL,
            f"最大偏差 {np.abs(row_sum - 1).max():.1e}")
        pad_w = wn[:, ~mask[d]]
        add("4 空位权重 = 0", (pad_w.size == 0) or (np.abs(pad_w).max() == 0),
            f"空位 {500 - n_valid} 个" + (f"，最大权重 {np.abs(pad_w).max():.1e}" if pad_w.size else ""))

        # 5 按公式手算
        W_K = emb.W_K.weight.detach().numpy().T            # (79, 32)
        Q = att.Q.detach().numpy()                         # (K, 32)
        xt_np = norm.transform(X[d]) @ W_K
        s_np = Q @ xt_np.T / np.sqrt(D_EMB)
        w_np = np_softmax_rows(s_np, mask[d])
        diff = np.abs(w_np - wn).max()
        add("5 与手算公式一致", diff < TOL, f"最大差 {diff:.1e}")

        if day_name == "满员日":
            # 6 多天批量 vs 逐天
            idx = list(range(d, d + 5))
            xb = to_t(norm.transform(X[d:d + 5]))
            mb = to_t(mask[d:d + 5])
            wb = att(emb(xb), mb).detach().numpy()         # (5, K, 500)
            w_each = np.stack([att(emb(to_t(norm.transform(X[i]))), to_t(mask[i])).detach().numpy()
                               for i in idx])
            diff_b = np.abs(wb - w_each).max()
            add("6 批量 = 逐天", diff_b < TOL, f"形状 {wb.shape}，最大差 {diff_b:.1e}")

            # 7 梯度：用因子收益 F_t = ω^F R_t 构造一个标量，反向传播
            emb.zero_grad()
            att.zero_grad()
            F_t = att(emb(x), m) @ to_t(R[d])               # (K,)
            (F_t ** 2).sum().backward()
            gq = att.Q.grad.norm().item()
            gw = emb.W_K.weight.grad.norm().item()
            add("7 梯度传回 Q 和 W_K", gq > 0 and gw > 0, f"|∇Q| {gq:.2e}，|∇W_K| {gw:.2e}")

            # 记录一些描述统计（初始化状态下权重的分散程度）
            eff_n = 1.0 / (wn ** 2).sum(1)                  # 有效股票数
            examples[K] = dict(eff_n=eff_n, wmax=wn.max(1), F=F_t.detach().numpy(), w=wn)

# ---------------------------------------------------------------------------
# 写 log
# ---------------------------------------------------------------------------
n_pass = sum(r[3] for r in results)
log("=" * 78)
log("环节 ②：注意力因子权重 —— 检查结果")
log("=" * 78)
log("")
log("一、原文")
log("  [ATT] 3.2 节，公式 (1)：ω^F_{t-1} = Softmax(Q X̃ᵀ_{t-1} / √d)")
log("  \"Each asset is attended with a dot product to the query vector Q_k ∈ R^d for each factor,")
log("   with query matrix Q = (Q_1, ..., Q_K)ᵀ giving a factor weight matrix ω^F_{t-1} ...")
log("   where Softmax is applied along each row to ensure that the resulting factors are normalized.\"")
log("  d = 32（附录 B Table 4），K ∈ {1, 3, 5, 8, 15, 30, 100}（Table 2）")
log("")
log("二、实现（af_model.py 中的 AttentionFactorWeights）")
log("  1. 可学习参数 Q：K × 32 的矩阵，每一行是一个因子的查询向量。")
log("  2. 分数 = Q X̃ᵀ / √d，形状 K × N：第 k 行第 i 列 = 第 k 个因子与第 i 只股票的相似度。")
log("  3. 每一行在股票上做 Softmax -> ω^F（K × N），每一行全正、加总为 1。")
log("  自行设定（原文未说明）：")
log("  - 空位处理：某天股票不足 500 只时，空位分数设为 -inf，Softmax 后权重正好为 0。")
log("  - Q 的初始化：正态分布 N(0, 1/d)。")
log("")
log("三、检查设置")
log(f"  水平量标准化用第一个训练窗口 {FIRST_TRAIN[0]} ~ {FIRST_TRAIN[1]} 拟合")
log(f"  满员日：{dts[d_full].date()}（{n_stock[d_full]} 只股票）")
log(f"  有空位的日：{dts[d_pad].date()}（{n_stock[d_pad]} 只股票，{500 - n_stock[d_pad]} 个空位，"
    f"样本中股票最少的一天）")
log(f"  W_K、Q 随机初始化（种子 {SEED}），未训练，因此 ω^F 的数值没有经济含义，只检查计算是否正确。")
log("")
log(f"四、检查结果：{n_pass} / {len(results)} 项通过")
log("")
log(f"  {'K':>4}  {'检查日':<8}  {'检查项':<18}  {'结果':<4}  数值")
for K, day_name, item, ok, val in results:
    log(f"  {K:>4}  {day_name:<8}  {item:<18}  {'通过' if ok else '失败':<4}  {val}")
log("")
log("五、补充：初始化状态下因子权重的分散程度（满员日）")
log("  有效股票数 = 1 / Σ w²：如果某因子平均分给 500 只股票，有效股票数 = 500；")
log("  如果全部押在 1 只股票上，有效股票数 = 1。训练后权重会集中到某些行业，有效股票数会下降。")
log(f"  {'K':>4}  {'有效股票数(各因子平均)':>22}  {'最大单只权重':>12}")
for K in K_LIST:
    e = examples[K]
    log(f"  {K:>4}  {e['eff_n'].mean():>22.1f}  {e['wmax'].max():>12.4f}")
log("")
e8 = examples[8]
log("  例：K = 8 时，各因子在满员日的因子收益 F_t = ω^F R_t（%）：")
log("    " + "  ".join(f"F{k + 1}: {v * 100:+.3f}" for k, v in enumerate(e8["F"])))
log("    （每个因子都是只做多的组合，初始化时接近等权，所以 8 个因子收益都接近当天等权市场收益 "
    f"{np.asarray(R[d_full])[mask[d_full]].mean() * 100:+.3f}%）")
log("")
log("六、下一环节")
log("  环节 ③：载荷与残差投影")
log("    β_{t-1}ᵀ = ω^Fᵀ (ω^F ω^Fᵀ + λ_ridge I_K)^{-1}，  ω^ε = I_N - β_{t-1}ᵀ ω^F")
log("")
log(f"总耗时 {time.time() - t0:.0f}s")

with open(LOG_PATH, "w") as fh:
    fh.write("\n".join(L) + "\n")

if n_pass != len(results):
    raise SystemExit(f"有 {len(results) - n_pass} 项检查未通过，见 {LOG_PATH}")
