"""
Attention Factor 模型组件库（Attention Factors for Statistical Arbitrage 复现）

每完成一个环节就把对应的组件加到这里，后面的脚本统一 import：
    环节 ①  特征嵌入        LevelNormalizer, FeatureEmbedding        <- 已完成
    环节 ②  注意力因子权重   AttentionFactorWeights                   <- 已完成
    环节 ③  载荷与残差投影   ResidualProjection                       <- 已完成
    环节 ④  过去 30 天残差   daily_residuals, residual_windows        <- 已完成
    环节 ⑤  LongConv 交易信号 LongConvLayer, LongConvSignal           <- 已完成
    环节 ⑥  映射回股票、归一化   stock_weights, portfolio_returns, block_forward  <- 已完成
    环节 ⑦  交易成本和净收益     align_prev_weights, trading_costs, block_costs  <- 已完成
    环节 ⑧  损失函数             sharpe_annualized, explained_variance, af_loss,
                                 AttentionFactorModel, run_blocks           <- 已完成
    数据读取                     HistBlocks, fit_normalizer

引用的论文：
    [ATT] Epstein, Wang, Choi, Pelger (2025). Attention Factors for Statistical Arbitrage.
    [19]  Guijarro-Ordonez, Pelger, Zanotti. Deep Learning Statistical Arbitrage (arXiv:2106.04028).
    [15]  Fu, Epstein, et al. (2023). Simple Hardware-Efficient Long Convolutions for Sequence Modeling.
"""

import json

import numpy as np
import torch
import torch.nn as nn

N_CHARS = 39                         # 特征排序值列：X[..., 0:39]
N_FEAT = 79                          # 39 排序值 + 39 截面中位数 + 1 无风险利率
LEVEL_SLICE = slice(N_CHARS, N_FEAT)  # 水平量列：X[..., 39:79]

# 截面中位数里量级跨越好几个数量级、且恒为正的列：先取对数再标准化
LOG_LEVEL_COLS = ["med_AT", "med_LME", "med_Vol", "med_Variance", "med_Resid_Var",
                  "med_Spread", "med_STD_W1", "med_LTurnover"]


# ===========================================================================
# 环节 ①  特征嵌入
# ===========================================================================
class LevelNormalizer:
    """
    对 40 个"水平量"列（39 个截面中位数 + rf）做标准化；39 个排序值列保持不变。

    原因：排序值已经在 [-0.5, 0.5]，但中位数是原始量纲（例如 med_AT ≈ 1 万，
    med_Spread ≈ 0.0007），直接乘 W_K 会让大数值的列主导嵌入。
    做法：部分列先取对数，然后用"训练窗口内"的均值和标准差做 z-score，
    样本外只用训练窗口的统计量，不会用到未来信息。
    [ATT] 原文没有说明这一步，属于自行设定。
    """

    def __init__(self, feat_cols):
        self.feat_cols = list(feat_cols)
        level_cols = self.feat_cols[LEVEL_SLICE]
        self.log_mask = np.array([c in LOG_LEVEL_COLS for c in level_cols])
        self.mean = None
        self.std = None

    def _transform_levels(self, lv):
        lv = lv.astype("float64").copy()
        lv[..., self.log_mask] = np.log(np.maximum(lv[..., self.log_mask], 1e-12))
        return lv

    def fit(self, X_days):
        """X_days: (D, N, 79)，只传训练窗口的数据。水平量在同一天所有股票上相同，取每天第 1 只即可。"""
        lv = self._transform_levels(X_days[:, 0, LEVEL_SLICE])
        self.mean = lv.mean(axis=0)
        self.std = lv.std(axis=0)
        self.std[self.std < 1e-12] = 1.0
        return self

    def transform(self, X):
        """X: (..., N, 79) -> 同形状 float32，水平量列被标准化。"""
        X = np.array(X, dtype="float32", copy=True)
        lv = self._transform_levels(X[..., LEVEL_SLICE])
        X[..., LEVEL_SLICE] = ((lv - self.mean) / self.std).astype("float32")
        return X

    def state_dict(self):
        return {"feat_cols": self.feat_cols, "mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_state_dict(cls, sd):
        obj = cls(sd["feat_cols"])
        obj.mean, obj.std = np.array(sd["mean"]), np.array(sd["std"])
        return obj


class FeatureEmbedding(nn.Module):
    """
    [ATT] 3.2 节 "Attention Factors"：
        X̃_t = X_t W_K,   W_K ∈ R^{M×d}
    M = 79（[ATT] 4.1 节），d = 32（[ATT] 附录 B Table 4: "d_x = 32, Model attention dim"）。

    实现为不带偏置的线性层（原文公式没有偏置项）。
    nn.Linear 的 weight 形状是 (d, M) = W_Kᵀ，所以 forward 计算的就是 X @ W_K。
    初始化用 PyTorch 默认（Kaiming uniform），原文没有说明，属于自行设定。

    输入  X: (..., N, M)   N 只股票，每只 M 个特征
    输出  X̃: (..., N, d)
    """

    def __init__(self, n_features=N_FEAT, d=32):
        super().__init__()
        self.W_K = nn.Linear(n_features, d, bias=False)

    def forward(self, X):
        return self.W_K(X)


def load_feature_cols(parquet_path):
    """从 features.parquet 的元数据读出 79 个特征列名（顺序固定）。"""
    import pyarrow.parquet as pq
    meta = pq.read_schema(parquet_path).metadata
    return json.loads(meta[b"feature_cols"])


# ===========================================================================
# 环节 ②  注意力因子权重
# ===========================================================================
class AttentionFactorWeights(nn.Module):
    """
    [ATT] 3.2 节，公式 (1)：
        ω^F_{t-1} = Softmax(Q X̃ᵀ_{t-1} / √d),   Q = (Q_1, ..., Q_K)ᵀ ∈ R^{K×d}
    "where Softmax is applied along each row to ensure that the resulting factors are normalized"
    即：对每个因子（每一行），在所有股票上做 Softmax，得到 K × N 的因子权重矩阵，
    每一行全为正、加总为 1（每个因子是一个只做多的股票组合）。

    自行设定（原文未说明）：
      - 空位处理：某天股票不足 N 只时，空位的分数设为 -inf，Softmax 后权重正好为 0，
        权重只在真实股票之间分配。
      - Q 的初始化：正态分布 N(0, 1/d)，使初始点积的量级与 √d 缩放相匹配。

    输入  X̃: (..., N, d)   环节 ① 的输出
          mask: (..., N)   bool，True 表示该位置有股票（可为 None）
    输出  ω^F: (..., K, N)
    """

    def __init__(self, n_factors, d=32):
        super().__init__()
        self.K, self.d = n_factors, d
        self.Q = nn.Parameter(torch.randn(n_factors, d) / d ** 0.5)

    def scores(self, X_tilde):
        """Q X̃ᵀ / √d，形状 (..., K, N)"""
        return torch.matmul(self.Q, X_tilde.transpose(-1, -2)) / self.d ** 0.5

    def forward(self, X_tilde, mask=None):
        s = self.scores(X_tilde)
        if mask is not None:
            s = s.masked_fill(~mask.unsqueeze(-2), float("-inf"))
        return torch.softmax(s, dim=-1)


# ===========================================================================
# 环节 ③  载荷与残差投影
# ===========================================================================
RIDGE_DEFAULT = 1e-5   # 原文未给出；BSIC (2025) 用 1e-3，但对本数据偏大，改用 1e-5，见 03 的 log


class ResidualProjection(nn.Module):
    """
    [ATT] 3.2 节 "Residual Portfolios" 与 "Attention Factors" 最后一个公式：
        β_{t-1}ᵀ = ω^Fᵀ (ω^F ω^Fᵀ + λ_ridge I_K)^{-1}          (N × K)
        ω^ε_{t-1} = I_N - β_{t-1}ᵀ ω^F_{t-1}                    (N × N)
    残差 ε_t = ω^ε_{t-1} R_t = R_t - β_{t-1}ᵀ F_t，其中 F_t = ω^F_{t-1} R_t。
    没有可学习参数：β 和 ω^ε 完全由 ω^F 决定。

    实现细节：
      - 用 linalg.solve 代替显式求逆（数值更稳定，结果相同）：
        βᵀ = ω^Fᵀ A^{-1} = (A^{-1} ω^F)ᵀ，A = ω^F ω^Fᵀ + λ I 对称。
      - 自行设定：空位股票在 ω^ε 中对应的行和列置 0，使空位的残差恒为 0、
        也不会影响其他股票的残差（原公式中 I_N 会给空位留下对角线上的 1）。

    输入  ω^F: (..., K, N)，mask: (..., N) 或 None
    输出  βᵀ: (..., N, K)，ω^ε: (..., N, N)
    """

    def __init__(self, ridge=RIDGE_DEFAULT):
        super().__init__()
        self.ridge = ridge

    def forward(self, wF, mask=None):
        K, N = wF.shape[-2], wF.shape[-1]
        eye_K = torch.eye(K, dtype=wF.dtype, device=wF.device)
        A = wF @ wF.transpose(-1, -2) + self.ridge * eye_K          # (..., K, K)
        beta_T = torch.linalg.solve(A, wF).transpose(-1, -2)          # (..., N, K)
        eye_N = torch.eye(N, dtype=wF.dtype, device=wF.device)
        w_eps = eye_N - beta_T @ wF                                   # (..., N, N)
        if mask is not None:
            mf = mask.to(wF.dtype)
            w_eps = w_eps * mf.unsqueeze(-1) * mf.unsqueeze(-2)
        return beta_T, w_eps


# ===========================================================================
# 环节 ④  过去 30 天残差
# ===========================================================================
LOOKBACK = 30   # [ATT] 4.2 节: "All sequence models use a look-back window of the past 30 daily residual returns"


def daily_residuals(emb, att, proj, X, R, valid):
    """
    每天用当天自己的残差投影算当天的残差（[19] 公式 (1) 与 III.B 节的"daily out-of-sample residuals"）：
        ε_s = ω^ε_{s-1} R_s，  ω^ε_{s-1} 由第 s 行的特征 X_{s-1} 经环节 ①②③ 得到

    输入  X: (D, N, 79) 已标准化的特征（第 s 行是 X_{s-1}）
          R: (D, N) 第 s 天的收益 R_s
          valid: (D, N) bool，该位置当天是否有数据
    输出  eps: (D, N) 每天的残差，无数据的位置为 0
          w_eps: (D, N, N) 每天的残差投影矩阵（环节 ⑥ 把组合权重映射回股票时要用当天的）
    """
    wF = att(emb(X), valid)                       # (D, K, N)
    _, w_eps = proj(wF, valid)                    # (D, N, N)
    eps = (w_eps @ R.unsqueeze(-1)).squeeze(-1)   # (D, N)
    return eps, w_eps


def residual_windows(eps, valid, L=LOOKBACK):
    """
    把每日残差整理成每个交易日 t 的回看窗口，并累计成"残差价格"（[19] 的 Int(·)）：
        x_{t} = (ε_{t-L}, ε_{t-L}+ε_{t-L+1}, ..., Σ_{l=1..L} ε_{t-L-1+l})
    只用 t-1 及以前的残差，不含 t 日。

    输入  eps, valid: (D, N)
    输出  x: (D-L, N, L)  第 i 个对应交易日 t = L + i
          ok: (D-L, N) bool，窗口内 L 天全部有数据（[19]：窗口内有缺失的股票当天不交易）
    """
    win = eps.unfold(0, L, 1)[:-1]                       # (D-L, N, L)，第 i 个窗口 = 第 i..i+L-1 天
    v = valid.to(eps.dtype).unfold(0, L, 1)[:-1]
    return win.cumsum(-1), v.sum(-1) == L


# ===========================================================================
# 环节 ⑤  LongConv 交易信号
# ===========================================================================
LAMBDA_SQUASH = 1e-3   # [ATT] 附录 B Table 4: λ_squash = 0.001
DROPOUT = 0.1          # [ATT] 附录 B Table 4: Dropout = 0.1


def geom_decay_kernel(d, T, generator=None):
    """
    [ATT] 附录 A "Initialization"：
        K^(h)_t = x · exp( -(t/T) · (d/2)^(h/d) ),  x ~ N(0, 1),  1 ≤ t ≤ T, 1 ≤ h ≤ d
    不同的 h 衰减速度不同：h 小的衰减慢（看长期），h 大的衰减快（看短期）。
    [15] 代码（safari 仓库 long_conv_kernel.py 的 "double_exp"）用的是 x ~ 0.02·N(0,1)，
    下标从 0 开始；这里按 [ATT] 原文的写法（x ~ N(0,1)，下标从 1 开始）。
    """
    t = torch.arange(1, T + 1, dtype=torch.float32)
    h = torch.arange(1, d + 1, dtype=torch.float32)
    decay = torch.exp(-(t[None, :] / T) * (d / 2.0) ** (h[:, None] / d))     # (d, T)
    x = torch.randn(d, T, generator=generator)
    return x * decay


class LongConvLayer(nn.Module):
    """
    一层 LongConv，结构照 [15] 官方代码（HazyResearch/safari 的 standalone_long_convs.py 中的 LongConv）：
        y = K̄ * u + D ⊙ u                      [ATT] 附录 A
        K̄ = sign(K) ⊙ max(|K| - λ_squash, 0)   [ATT] 附录 A 的 Squash 算子
        卷积用 FFT 计算（补零到 2T，取前 T 个 -> 因果卷积：第 i 天只用第 1..i 天）
        之后：GELU -> dropout -> Linear(d, 2d) + GLU（[15] 代码中的 output_linear）
    输入/输出: (B, d, T)
    """

    def __init__(self, d=32, T=LOOKBACK, lam_squash=LAMBDA_SQUASH, dropout=DROPOUT):
        super().__init__()
        self.d, self.T, self.lam = d, T, lam_squash
        self.kernel = nn.Parameter(geom_decay_kernel(d, T))      # K ∈ R^{d×T}
        self.D = nn.Parameter(torch.randn(d))                    # 跳跃连接 D ∈ R^d
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.output_linear = nn.Sequential(nn.Linear(d, 2 * d), nn.GLU(dim=-1))

    def squashed_kernel(self):
        k = self.kernel
        return torch.sign(k) * torch.relu(torch.abs(k) - self.lam)

    def conv(self, u):
        """K̄ * u + D ⊙ u，u: (B, d, T)"""
        T = u.shape[-1]
        k = self.squashed_kernel()
        k_f = torch.fft.rfft(k, n=2 * T)
        u_f = torch.fft.rfft(u, n=2 * T)
        y = torch.fft.irfft(u_f * k_f, n=2 * T)[..., :T]
        return y + u * self.D[:, None]

    def forward(self, u):
        y = self.dropout(self.activation(self.conv(u)))
        y = self.output_linear(y.transpose(-1, -2)).transpose(-1, -2)
        return y


class LongConvSignal(nn.Module):
    """
    [ATT] 3.3 节：ω^port_{i,t-1} = LongConv_θ(ε_{i,(t-s,t-1)})
    整体结构照 [15] 官方代码 standalone_long_convs.py 中的 LongConvModel：
        编码:   Linear(1, d)          每天的累计残差（1 个数）-> d = 32 个通道
        n 层:   z = LongConvLayer(x) -> Dropout -> x = x + z -> LayerNorm   （残差块，post-norm）
        汇总:   在 T = 30 天上取平均
        解码:   Linear(d, 1)          -> 每只股票一个数 ω^port_i
    [ATT] Table 4：1 层、隐藏维度 32、dropout 0.1、λ_squash = 0.001。
    同一组参数用于所有股票、所有交易日（每只股票的残差序列独立地经过同一个 LongConv）。

    输入  x: (..., L) 累计残差窗口（环节 ④ 的 residual_windows 输出）
    输出  ω^port: (...)
    """

    def __init__(self, d=32, n_layers=1, T=LOOKBACK, lam_squash=LAMBDA_SQUASH, dropout=DROPOUT):
        super().__init__()
        self.encoder = nn.Linear(1, d)
        self.layers = nn.ModuleList([LongConvLayer(d, T, lam_squash, dropout) for _ in range(n_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        self.decoder = nn.Linear(d, 1)

    def forward(self, x):
        shape = x.shape[:-1]
        h = self.encoder(x.reshape(-1, x.shape[-1], 1))          # (B, T, d)
        h = h.transpose(-1, -2)                                   # (B, d, T)
        for layer, drop, norm in zip(self.layers, self.dropouts, self.norms):
            h = h + drop(layer(h))
            h = norm(h.transpose(-1, -2)).transpose(-1, -2)
        h = h.mean(dim=-1)                                        # (B, d)
        return self.decoder(h).reshape(shape)


# ===========================================================================
# 环节 ⑥  映射回股票、归一化
# ===========================================================================
def stock_weights(w_eps, w_port, ok=None, tiny=1e-12):
    """
    [ATT] 3.3 节：R^port_t = ε_tᵀ ω^port_{t-1} = R_tᵀ (ω^ε_{t-1})ᵀ ω^port_{t-1} = R_tᵀ ω_{t-1}
                  目标函数下方 "subject to ‖ω_t‖₁ = 1"（ω 没有 port 上标，是股票空间的权重）
    [19] 公式 (3)/(5)：w^R_{t-1} = w^εᵀ_{t-1} Φ_{t-1} / ‖w^εᵀ_{t-1} Φ_{t-1}‖₁
         III.C 节："mapped back into positions of the original stocks ... the absolute values of the
         individual stock portfolio weights sums up to one, i.e. ‖w^R_{t-1}‖₁ = 1"
    即：先把残差组合上的下注 ω^port 换算成股票权重 ω = (ω^ε)ᵀ ω^port，再在股票空间做 L1 归一化。

    自行设定：
      - ok = False（30 天窗口不完整，[19] 规定当天不交易）的残差，ω^port 置 0。
      - ‖ω‖₁ 为 0（当天没有任何可交易残差）时不除，权重保持全 0。

    输入  w_eps: (..., N, N) ω^ε_{t-1}（第 i 行 = 第 i 个残差组合在 N 只股票上的权重）
          w_port: (..., N)   ω^port_{t-1}，环节 ⑤ 的输出
          ok: (..., N) bool 或 None
    输出  ω_{t-1}: (..., N)，每天 ‖ω‖₁ = 1
    """
    if ok is not None:
        w_port = w_port * ok.to(w_port.dtype)
    w = (w_eps.transpose(-1, -2) @ w_port.unsqueeze(-1)).squeeze(-1)
    l1 = w.abs().sum(-1, keepdim=True)
    return w / torch.where(l1 > tiny, l1, torch.ones_like(l1))


def portfolio_returns(w, R):
    """R^port_t = R_tᵀ ω_{t-1}。w, R: (..., N) -> (...)"""
    return (w * R).sum(-1)


def block_forward(emb, att, proj, net, X, R, valid, L=LOOKBACK):
    """
    一个月块的完整前向（环节 ①-⑥）。块 = 往前 L 个交易日 + 当月交易日（见 04）。
    返回当月每个交易日 t 的：
        eps    ε_t = ω^ε_{t-1} R_t                (n, N)
        R      R_t                                  (n, N)
        valid  t 日该位置有数据                     (n, N)
        ok     30 天窗口完整、可交易                (n, N)
        w_port ω^port_{t-1}                         (n, N)
        w      ω_{t-1}，‖ω‖₁ = 1                    (n, N)
        r_port R^port_t = R_tᵀ ω_{t-1}              (n,)
        w_eps  ω^ε_{t-1}                            (n, N, N)
    """
    eps, w_eps = daily_residuals(emb, att, proj, X, R, valid)
    x, ok = residual_windows(eps, valid, L)
    w_port = net(x)
    w = stock_weights(w_eps[L:], w_port, ok)
    return dict(eps=eps[L:], R=R[L:], valid=valid[L:], ok=ok, w_port=w_port, w=w,
                r_port=portfolio_returns(w, R[L:]), w_eps=w_eps[L:])


# ===========================================================================
# 环节 ⑦  交易成本和净收益
# ===========================================================================
TC_TRADE = 0.0005   # [ATT] 3.3 节 / [19] III.J 节：每笔交易 5 个基点
TC_SHORT = 0.0001   # [ATT] 3.3 节 / [19] III.J 节：做空 1 个基点


def align_prev_weights(w_prev, permno_prev, permno_cur):
    """
    把前一天的权重按 PERMNO 对到今天的位置（月初换股票池时，前后两天的第 k 个位置不是同一只股票）。
        aligned[k] = 前一天同一只股票的权重；今天新进池子的股票 = 0
        exit_l1    = 前一天持有、今天已不在池子里的股票的 Σ|ω|（这些仓位要清掉，也是交易）
    permno 为 0 的是空位，不参与匹配。

    输入  w_prev: (N_prev,) 张量；permno_prev: (N_prev,)；permno_cur: (N_cur,)（numpy）
    输出  aligned: (N_cur,) 张量，exit_l1: 标量张量
    """
    pp, pc = np.asarray(permno_prev), np.asarray(permno_cur)
    order = np.argsort(pp)
    pos = np.clip(np.searchsorted(pp[order], pc), 0, len(pp) - 1)
    hit = (pp[order][pos] == pc) & (pc > 0)
    src = torch.from_numpy(order[pos])
    aligned = torch.where(torch.from_numpy(hit), w_prev[src], torch.zeros((), dtype=w_prev.dtype))
    exiting = torch.from_numpy((pp > 0) & ~np.isin(pp, pc[pc > 0]))
    exit_l1 = (w_prev.abs() * exiting.to(w_prev.dtype)).sum()
    return aligned, exit_l1


def trading_costs(w, w_prev, exit_l1=0.0):
    """
    [ATT] 3.3 节 "Arbitrage Trading Objective"（"We measure transaction costs as in [19]"）：
        cost(ω_t, ω_{t-1}) = 0.0005 ‖ω_t - ω_{t-1}‖₁ + 0.0001 ‖max(-ω_t, 0)‖₁
    [19] III.J 节写法：cost(w_{t-1}, w_{t-2}) = 0.0005‖w_{t-1} - w_{t-2}‖ + 0.0001‖min(w_{t-1}, 0)‖，
        从 t 日组合收益 w_{t-1}ᵀR_t 中扣除。两者相同。
    w, w_prev 已按 PERMNO 对齐，形状 (..., N)；exit_l1 是已离开股票池的仓位，计入换手。
    返回 cost, turnover = ‖ω_t - ω_{t-1}‖₁, short = ‖max(-ω_t, 0)‖₁
    """
    turnover = (w - w_prev).abs().sum(-1) + exit_l1
    short = torch.relu(-w).sum(-1)
    return TC_TRADE * turnover + TC_SHORT * short, turnover, short


def block_costs(w, permno, w_last=None, permno_last=None):
    """
    一个月块内每个交易日的成本（时间约定照 [19]：t 日净收益 = ω_{t-1}ᵀR_t - cost(ω_{t-1}, ω_{t-2})）。
        当月第 1 天：和上一块最后一天的权重比（按 PERMNO 对齐，离开股票池的仓位计入换手）
        当月其余天：同一块内位置对应同一只股票，直接相减
        w_last 为 None：从空仓开始（第 1 天换手 = ‖ω‖₁ = 1）
    输入  w: (n, N) 当月每天的 ω（block_forward 的输出），permno: (N,)
    输出  cost, turnover, short: (n,)
    """
    if w_last is None:
        prev0, exit0 = torch.zeros_like(w[0]), torch.zeros((), dtype=w.dtype)
    else:
        prev0, exit0 = align_prev_weights(w_last, permno_last, permno)
    w_prev = torch.cat([prev0.unsqueeze(0), w[:-1]], 0)
    exit_vec = torch.zeros(w.shape[0], dtype=w.dtype)
    exit_vec[0] = exit0
    return trading_costs(w, w_prev, exit_vec)


# ===========================================================================
# 环节 ⑧  损失函数
# ===========================================================================
LAMBDA_VAR = 100.0   # [ATT] 附录 B Table 4: λ_Var = 100
ANNUAL = 252         # [19] 脚注 12：年化用 252 个交易日
EV_MIN_OBS = 10      # 自行设定：一只股票在 batch 内至少 10 天数据才计入解释方差


def sharpe_annualized(r_net, rf):
    """
    [ATT] 3.3 节目标函数的第一项（net Sharpe ratio）：
        (R̄^port_net - R_f) / sqrt( (1/T) Σ_t (R^port_{t,net} - R̄^port_net)² )
        分母按原文用 1/T（总体标准差）。R_f 取同一段时间日无风险利率的均值。
    [19] 附录 D："replacing the mean and variance by their annualized sample counterpart over a
        training set"；脚注 12：μ = 252/T Σ R_t，σ = sqrt(252/T Σ (R_t - μ)²)
        所以年化 Sharpe = √252 × 日均值 / 日标准差。
    输入  r_net, rf: (T,)
    """
    mu = r_net.mean() - rf.mean()
    sd = r_net.std(unbiased=False)
    return ANNUAL ** 0.5 * mu / sd


def explained_variance(eps, R, valid, stock_id, min_obs=EV_MIN_OBS):
    """
    [ATT] 3.3 节目标函数的第二项（explained variance）：
        (1/N) Σ_i ( 1 - Var(e_i) / Var(R_i) )
    Var 是每只股票在这段时间（一个 batch）里的时间序列方差，e_i 是它的残差 ε_{i,t}。

    自行设定（原文未说明）：
      - 按 PERMNO（stock_id）汇总：一个 batch 跨好几个月，同一只股票在不同月块里的位置不同。
      - 只用当天有数据（valid）的 (股票, 日)；N = batch 内至少 min_obs 天数据、且 Var(R_i) > 0 的股票数。
      - 方差用总体方差（1/n），与 Sharpe 的分母一致；比值对 1/n 还是 1/(n-1) 不敏感。
    输入  eps, R: (T, N)，valid: (T, N) bool，stock_id: (T, N) int64
    输出  标量
    """
    m = valid.reshape(-1)
    e, r = eps.reshape(-1)[m], R.reshape(-1)[m]
    sid = torch.as_tensor(stock_id).reshape(-1)[m]
    _, inv = torch.unique(sid, return_inverse=True)
    U = int(inv.max()) + 1
    z = lambda v: torch.zeros(U, dtype=e.dtype).index_add(0, inv, v)   # noqa: E731
    n = z(torch.ones_like(e))
    var_e = z(e * e) / n - (z(e) / n) ** 2
    var_r = z(r * r) / n - (z(r) / n) ** 2
    keep = (n >= min_obs) & (var_r > 0)
    return (1 - var_e[keep] / var_r[keep]).mean()


def af_loss(r_net, rf, eps, R, valid, stock_id, lam_var=LAMBDA_VAR):
    """
    [ATT] 3.3 节：max  net Sharpe + λ_Var × explained variance
    神经网络习惯做最小化，所以 loss = -(Sharpe + λ_Var × EV)。
    返回 loss, sharpe, ev
    """
    sr = sharpe_annualized(r_net, rf)
    ev = explained_variance(eps, R, valid, stock_id)
    return -(sr + lam_var * ev), sr, ev


class AttentionFactorModel(nn.Module):
    """把环节 ①-⑤ 的组件放在一起：W_K、Q、LongConv 是全部可学习参数（[ATT] 3.3 节最后一句）。"""

    def __init__(self, K, d=32, ridge=RIDGE_DEFAULT, n_layers=1):
        super().__init__()
        self.emb = FeatureEmbedding(N_FEAT, d)
        self.att = AttentionFactorWeights(K, d)
        self.proj = ResidualProjection(ridge)
        self.net = LongConvSignal(d=d, n_layers=n_layers)

    def forward_block(self, blk):
        return block_forward(self.emb, self.att, self.proj, self.net, blk["X"], blk["R"], blk["valid"])


def run_blocks(model, blocks, w_last=None, permno_last=None):
    """
    依次跑若干个连续的月块（环节 ①-⑦），拼成一段连续交易日，供 af_loss 使用。
    成本按时间顺序精确计算：每块第 1 天接上一块最后一天（按 PERMNO 对齐）；
    w_last 为 None 时第一块从空仓开始。
    返回 dict：r_gross, cost, turnover, short, r_net, rf, eps, R, valid, sid（股票 PERMNO），
              w_last, permno_last（最后一天的权重，用来接下一段）
    """
    keys = ["r_gross", "cost", "turnover", "short", "rf", "eps", "R", "valid", "sid"]
    acc = {k: [] for k in keys}
    for blk in blocks:
        o = model.forward_block(blk)
        c, to, sh = block_costs(o["w"], blk["permno"], w_last, permno_last)
        n = o["w"].shape[0]
        acc["r_gross"].append(o["r_port"])
        acc["cost"].append(c)
        acc["turnover"].append(to)
        acc["short"].append(sh)
        acc["rf"].append(blk["rf"])
        acc["eps"].append(o["eps"])
        acc["R"].append(o["R"])
        acc["valid"].append(o["valid"])
        acc["sid"].append(torch.from_numpy(np.broadcast_to(blk["permno"], (n, len(blk["permno"]))).copy()))
        w_last, permno_last = o["w"][-1], blk["permno"]
    out = {k: torch.cat(v) for k, v in acc.items()}
    out["r_net"] = out["r_gross"] - out["cost"]
    out["w_last"], out["permno_last"] = w_last, permno_last
    return out


# ===========================================================================
# 数据读取（04 生成的按月块）
# ===========================================================================
def fit_normalizer(af_dir, start, end):
    """用 [start, end] 内的交易日拟合 LevelNormalizer（01 的 X.npy）。"""
    import os
    import pandas as pd
    feat_cols = json.load(open(os.path.join(af_dir, "feature_cols.json")))["feature_cols"]
    X01 = np.load(os.path.join(af_dir, "X.npy"), mmap_mode="r")
    d01 = pd.DatetimeIndex(np.load(os.path.join(af_dir, "dates.npy")))
    tr = np.flatnonzero((d01 >= start) & (d01 <= end))
    return LevelNormalizer(feat_cols).fit(np.asarray(X01[tr[0]:tr[-1] + 1]))


class HistBlocks:
    """读取 04 输出的按月块。block(b, norm) 返回第 b 个月块的张量。"""

    def __init__(self, af_dir):
        import os
        j = lambda f: os.path.join(af_dir, f)   # noqa: E731
        self.X = np.load(j("hist_X.npy"), mmap_mode="r")
        self.R = np.load(j("hist_R.npy"), mmap_mode="r")
        self.idx = np.load(j("hist_idx.npy"), mmap_mode="r")
        self.day = np.load(j("hist_day.npy"))
        self.ptr = np.load(j("hist_ptr.npy"))
        self.permno = np.load(j("hist_permno.npy"))
        self.dates = np.load(j("hist_dates.npy"))
        self.months = json.load(open(j("hist_months.json")))
        self.rf = np.load(j("hist_rf.npy")) if os.path.exists(j("hist_rf.npy")) else None   # 08 生成

    def block(self, b, norm, L=LOOKBACK):
        idx = np.asarray(self.idx[self.ptr[b]:self.ptr[b + 1]])
        days = self.day[self.ptr[b]:self.ptr[b + 1]]
        rf = None if self.rf is None else torch.from_numpy(self.rf[days][L:])
        return dict(X=torch.from_numpy(norm.transform(self.X[idx])),
                    R=torch.from_numpy(np.asarray(self.R[idx])),
                    valid=torch.from_numpy(idx > 0),
                    permno=self.permno[b],                                   # (N,) 当月股票，空位 0
                    dates=self.dates[days][L:],                                # 当月交易日
                    rf=rf,                                                     # 当月交易日的日无风险利率
                    month=self.months[b])
