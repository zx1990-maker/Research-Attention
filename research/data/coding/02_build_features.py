"""
第 2 步：构建 79 维特征（Attention Factors for Statistical Arbitrage 复现）

79 = 39 个特征的截面排序分位数 + 这 39 个特征的截面中位数 + 无风险利率

特征定义依据：
    [ATT] Epstein, Wang, Choi, Pelger (2025) Attention Factors for Statistical Arbitrage，
          Table 1 与 4.1 节
    [CPZ] Chen, Pelger, Zhu (2024) Deep Learning in Asset Pricing，arXiv:1904.00745v3，
          正文 A.1 节（第 24-26 页）与 Appendix B Table A.VII（第 65-68 页）
    每个特征的计算方法、原文出处、哪些是自行设定的，见 data/02_build_features.log

输入：
    data/daily_panel.parquet          第 1 步输出的日度面板
    wrds/comp_ccm_annual.sas7bdat     Compustat 年报（已带 CCM 链接）
    wrds/ff_daily.sas7bdat            Fama-French 日度因子与无风险利率

输出：
    data/features.parquet   一行 = (股票, 交易日 t)，包含两类行：
        (a) t 日在前 500 股票池中的股票（in_universe = 1）
        (b) 下个月将进入股票池的股票，在进入前 PRE_WINDOW 个交易日内的行（in_universe = 0）。
            用途：算过去 30 天残差时，每天要用当天的特征算当天的 ω^ε，新进股票在进池前
            那些天也需要特征（见 attention factor/04 的说明）。
        DATE, PERMNO       主键
        RET_ADJ            t 日收益（预测目标 R_t）
        in_universe        t 日是否在股票池中
        <39 个特征名>      只用 t-1 日及以前的信息，相对于 t 日股票池 500 只的排序分位数，[-0.5, 0.5]
                           （股票池外的行按"插入股票池后的位置"计算，超出范围的取 ±0.5）
        med_<特征名>       该特征当天在股票池内的截面中位数（原始量纲）
        rf                 t-1 日无风险利率
      => 每一行就是 (X_{t-1}, R_t)，后续模型直接用，不用再错位。
      parquet 元数据里存了特征分组（feature_groups），供 Table 3 消融实验使用。
    data/02_build_features.log  特征说明文档 + 运行结果

用法（在 research 目录下）：
    python data/coding/02_build_features.py

结构：
    Part 0  读数据，转成宽表（行 = 交易日，列 = 股票），建立月份索引
    Part 1  日度特征 4 个（Ret_D1, Ret_W1, STD_W1, Vol）—— 每天更新
    Part 2  月度特征 13 个（过去收益与交易摩擦）—— 每月末计算，下个月使用
    Part 3  年度特征 22 个（Compustat 年报）—— Fama-French 6 月惯例
    Part 4  缺失值填补 + 截面排序 + 截面中位数 + rf -> 79 维
    Part 5  数据检查
    Part 6  保存 + 写 log
"""

import json
import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyreadstat

HERE = os.path.dirname(os.path.abspath(__file__))      # research/data/coding
ROOT = os.path.dirname(os.path.dirname(HERE))          # research
WRDS = os.path.join(ROOT, "wrds")
DATA = os.path.join(ROOT, "data")

PANEL_FILE = os.path.join(DATA, "daily_panel.parquet")
CCM_FILE = os.path.join(WRDS, "comp_ccm_annual.sas7bdat")
FF_FILE = os.path.join(WRDS, "ff_daily.sas7bdat")
OUT_FILE = os.path.join(DATA, "features.parquet")
LOG_PATH = os.path.join(DATA, "02_build_features.log")

# 论文 Table 1 的 6 组 39 个特征
FEATURE_GROUPS = {
    "past_returns": ["r2_1", "r12_2", "r12_7", "r36_13", "ST_Rev", "Ret_D1", "Ret_W1", "STD_W1"],
    "trading_frictions": ["AT", "LME", "LTurnover", "Rel2High", "Resid_Var", "Spread", "SUV",
                          "Variance", "Vol", "Beta"],
    "value": ["A2ME", "BEME", "C", "CF", "CF2P", "Q", "Lev", "E2P"],
    "investment": ["Investment", "NOA", "DPI2A"],
    "profitability": ["PROF", "CTO", "FC2Y", "OP", "PM", "RNA", "D2A"],
    "intangibles": ["OA", "OL", "PCM"],
}
CHARS = [c for g in FEATURE_GROUPS.values() for c in g]
assert len(CHARS) == 39

PRE_WINDOW = 60   # 新进股票：保留进入股票池前 60 个交易日的特征（回看窗口 30 天，留余量）

# ---------------------------------------------------------------------------
# 特征说明（写入 log）
#   freq: 更新频率；calc: 本代码的计算方法；src: 原文出处与原文定义；
#   self: 原文没有给出、由本代码自行设定的部分（None 表示完全按原文）
# ---------------------------------------------------------------------------
CPZ = "[CPZ] Table A.VII"
ATT = "[ATT] Table 1 / 4.1 节"
DOC = {
    # ---------------- 过去收益 ----------------
    "r2_1": dict(
        freq="每月",
        calc="上个自然月的收益（日收益 RET_ADJ 在月内复利，月内至少 10 个交易日有收益）。",
        src=f'{CPZ}: "Short-term momentum. Lagged one-month return"（Jegadeesh & Titman 1993）',
        self="原文 r2_1 与 ST_Rev 的定义字面相同（都是上个月收益），因此两列数值相同；"
             "月内最少 10 个交易日为自行设定。"),
    "r12_2": dict(
        freq="每月",
        calc="预测月为 t 时，t-12 到 t-2 月（共 11 个月）的累计收益，至少 8 个月有数据。",
        src=f'{CPZ}: "Momentum"，Fama & French (1996) 的 t-12 到 t-2 月累计收益',
        self="原文依赖 CRSP 缺失代码 -99 的筛选规则无法照搬，改为至少 8/11 个月有数据。"),
    "r12_7": dict(
        freq="每月",
        calc="t-12 到 t-7 月（共 6 个月）的累计收益，至少 5 个月有数据。",
        src=f'{CPZ}: "Cumulative return from 12 months before the return prediction to seven '
            f'months before"（Novy-Marx 2012）',
        self="最少月数 5 为自行设定。"),
    "r36_13": dict(
        freq="每月",
        calc="t-36 到 t-13 月（共 24 个月）的累计收益，至少 18 个月有数据。",
        src=f'{CPZ}: "Cumulative return from 36 months before the return prediction to 13 '
            f'months before"（De Bondt & Thaler 1985）',
        self="最少月数 18 为自行设定。"),
    "ST_Rev": dict(
        freq="每月",
        calc="上个自然月的收益（同 r2_1）。",
        src=f'{CPZ}: "Short-term reversal. Prior month return"（Jegadeesh & Titman 1993）',
        self=None),
    "Ret_D1": dict(
        freq="每日",
        calc="t-1 日的收益 RET_ADJ。",
        src=f'{ATT}: 只有名称 "Daily return"，以及 4.1 节 "we also include the previous day '
            f'and week return and volatility"',
        self="原文未给出具体定义（CPZ 中没有此特征），按名称自行定义。"),
    "Ret_W1": dict(
        freq="每日",
        calc="t-5 到 t-1 共 5 个交易日的累计收益，至少 4 天有数据。",
        src=f'{ATT}: 只有名称 "Weekly return"',
        self="原文未给出具体定义（CPZ 中没有此特征），一周 = 5 个交易日，自行定义。"),
    "STD_W1": dict(
        freq="每日",
        calc="t-5 到 t-1 共 5 个交易日日收益的标准差，至少 4 天有数据。",
        src=f'{ATT}: 只有名称 "Weekly volatility"',
        self="原文未给出具体定义（CPZ 中没有此特征），自行定义。"),
    # ---------------- 交易摩擦 ----------------
    "AT": dict(
        freq="每年",
        calc="总资产 AT（百万美元）。",
        src=f'{CPZ}: "Total Assets (AT)"（Gandhi & Lustig 2015）',
        self=None),
    "LME": dict(
        freq="每月",
        calc="上月最后一个交易日的市值 |PRC| × SHROUT（百万美元）。",
        src=f'{CPZ}: "Total market capitalization at the end of the previous month defined as '
            f'price times shares outstanding"（Fama & French 1992）',
        self=None),
    "LTurnover": dict(
        freq="每月",
        calc="上个月日成交量 VOL 之和 ÷ 上月末流通股数 SHROUT，月内至少 10 天有成交量。",
        src=f'{CPZ}: "Turnover is last month\'s volume (VOL) over shares outstanding (SHROUT)"'
            f'（Datar et al. 1998）',
        self="月内最少 10 天为自行设定；未对 NASDAQ 成交量做双重计数调整（原文未提）。"),
    "Rel2High": dict(
        freq="每月",
        calc="上月末复权收盘价 ÷ 截至上月末过去 252 个交易日的最高复权收盘价（至少 120 天）。"
             "复权价 = |PRC| / CFACPR。",
        src=f'{CPZ}: "The ratio of stock price at the end of the previous calendar month and the '
            f'highest daily price in the past year"（George & Hwang 2004）',
        self="数据中没有日内最高价，"
             "用日收盘价的最高值代替；最少 120 天为自行设定。"),
    "Resid_Var": dict(
        freq="每月",
        calc="过去两个自然月（上月和上上月）的日超额收益对 FF3（mktrf, smb, hml）回归，"
             "残差方差 = RSS / (n - 4)，至少 20 个交易日。",
        src=f'{CPZ}: "Variance of the residuals from a regression of excess returns in the past '
            f'two months on the Fama and French three-factor model"（Ang et al. 2004）',
        self="最少 20 天为自行设定。"),
    "Spread": dict(
        freq="每月",
        calc="上个月每日报价价差的平均值，日价差 = (ASK - BID) / 中间价（CRSP 字段 quoted_spread），"
             "月内至少 5 天有报价。",
        src=f'{CPZ}: "The average daily bid-ask spread in the previous month"（Chung & Zhang 2014）',
        self="月内最少 5 天为自行设定。注意 1985-1992 年 NYSE 股票在 CRSP 中基本没有日度报价，"
             "这些缺失由截面中位数填补。"),
    "SUV": dict(
        freq="每月",
        calc="用上月之前 3 个自然月的日数据回归 VOL = b0 + b1·max(r,0) + b2·max(-r,0)，"
             "得到残差标准差 σ（至少 40 天）；上个月每天计算 (实际 VOL - 预测 VOL) / σ，"
             "取月内平均（至少 10 天）。",
        src=f'{CPZ}: "Difference between actual volume and predicted volume in the previous month. '
            f'Predicted volume comes from a regression of daily volume on a constant and the '
            f'absolute values of positive and negative returns. Unexplained volume is standardized '
            f'by the standard deviation of the residuals from the regression"（Garfinkel 2009）',
        self="原文没有给出回归的估计窗口，也没有说明上月内的日值如何汇总；"
             "这里自行设定：估计窗口为前 3 个月，上月内取日均值。"),
    "Variance": dict(
        freq="每月",
        calc="过去两个自然月（上月和上上月）日收益的方差，至少 20 个交易日。",
        src=f'{CPZ}: "Variance of daily returns in the past two months"（Ang et al. 2004）',
        self="最少 20 天为自行设定。"),
    "Vol": dict(
        freq="每日",
        calc="t-5 到 t-1 共 5 个交易日的成交额之和 Σ VOL × |PRC|（百万美元），至少 4 天。",
        src=f'{ATT}: 只有名称 "Weekly trading volume"',
        self="原文未给出具体定义（CPZ 中没有此特征）。自行定义为成交额（而不是成交股数），"
             "因为股数在不同股票间不可比。"),
    "Beta": dict(
        freq="每月",
        calc="Frazzini-Pedersen beta：β = ρ · σ_i / σ_m。σ 为过去 252 个交易日日度对数超额收益"
             "的标准差（至少 120 天）；ρ 为过去 1260 个交易日（5 年）重叠 3 日对数超额收益"
             "的相关系数（至少 750 个）。市场超额收益用 FF 的 mktrf。取上月末的值。",
        src=f'{CPZ}: "Product of correlations between the excess return of stock i and the market '
            f'excess return and the ratio of volatilities. We calculate volatilities from the '
            f'standard deviations of daily log excess returns over a one-year horizon requiring at '
            f'least 120 observations. We estimate correlations using overlapping three-day log '
            f'excess returns over a five-year period requiring at least 750 non-missing '
            f'observations."（Frazzini & Pedersen 2014）',
        self="Frazzini-Pedersen 原文还会把 β 向 1 收缩，CPZ 未提，这里不做。"),
    # ---------------- 价值 ----------------
    "A2ME": dict(
        freq="每年",
        calc="AT ÷ 上年 12 月末市值。",
        src=f'{CPZ}: "Total assets (AT) over market capitalization (PRC*SHROUT) as of December '
            f't-1"（Bhandari 1988）',
        self=None),
    "BEME": dict(
        freq="每年",
        calc="BE ÷ 上年 12 月末市值。BE = SH + TXDITC - PS；SH = SEQ，缺失则 CEQ + PS，"
             "再缺失则 AT - LT；PS 依次取 PSTKRV、PSTKL、PSTK。",
        src=f'{CPZ}: BEME 条目（Fama & French 1992），BE 定义同上，"The market value of equity '
            f'(PRC*SHROUT) is as of December t-1"',
        self="BE ≤ 0 视为缺失（Fama-French 惯例，CPZ 未写）；TXDITC、PS 缺失记 0。"),
    "C": dict(
        freq="每年",
        calc="CHE ÷ AT。",
        src=f'{CPZ}: "Ratio of cash and short-term investments (CHE) to total assets (AT)"'
            f'（Palazzo 2012）',
        self=None),
    "CF": dict(
        freq="每年",
        calc="(NI + DP - WCAPCH - CAPX) ÷ BE。",
        src=f'{CPZ}: "net income (NI), depreciation and amortization (DP), less change in working '
            f'capital (WCAPCH), and capital expenditure (CAPX) over the book-value of equity"'
            f'（Hou et al. 2011）',
        self="本数据中 WCAPCH 有 98.6% 缺失，缺失时用资产负债表计算的营运资本变动代替："
             "Δ(ACT - CHE) - Δ(LCT - DLC - TXP)。"),
    "CF2P": dict(
        freq="每年",
        calc="(IB + DP + TXDB) ÷ 上年 12 月末市值。",
        src=f'{CPZ}: "Cashflow over market capitalization (PRC*SHROUT) as of December t-1. '
            f'Cashflow is defined as income before extraordinary items (IB) plus depreciation and '
            f'amortization (DP) plus deferred taxes (TXDB)"（Desai et al. 2004）',
        self="TXDB 缺失记 0。"),
    "Q": dict(
        freq="每年",
        calc="(AT + 上年 12 月末市值 - CEQ - TXDB) ÷ AT。",
        src=f'{CPZ}: "Tobin\'s Q is total assets (AT), the market value of equity (SHROUT times '
            f'PRC) minus ... (CEQ), minus deferred taxes (TXDB) scaled by total assets (AT)"'
            f'（Kaldor 1966）',
        self="原文没有写市值取哪个时点，这里与其他价值类特征一致，用上年 12 月末；TXDB 缺失记 0。"),
    "Lev": dict(
        freq="每年",
        calc="(DLTT + DLC) ÷ (DLTT + DLC + SEQ)。",
        src=f'{CPZ}: "Ratio of long-term debt (DLTT) and debt in current liabilities (DLC) to the '
            f'sum of long-term debt, debt in current liabilities, and stockholders\' equity (SEQ)"'
            f'（Lewellen 2015）',
        self="DLTT、DLC 缺失记 0。"),
    "E2P": dict(
        freq="每年",
        calc="IB ÷ 上年 12 月末市值。",
        src=f'{CPZ}: "total earnings before extraordinary items for the last fiscal year end in '
            f't-1. P (actually ME) is price times shares outstanding at the end of December of '
            f't-1"（Basu 1983）',
        self=None),
    # ---------------- 投资 ----------------
    "Investment": dict(
        freq="每年",
        calc="AT(本财年) ÷ AT(上财年) - 1，两个财年必须相邻。",
        src=f'{CPZ}: "Change in total assets (AT) from the fiscal year ending in year t-2 to the '
            f'fiscal year ending in t-1, divided by t-2 total assets"（Cooper et al. 2008）',
        self=None),
    "NOA": dict(
        freq="每年",
        calc="(经营资产 - 经营负债) ÷ 上财年 AT；经营资产 = AT - CHE - IVAO；"
             "经营负债 = AT - DLC - DLTT - MIB - PSTK - CEQ。",
        src=f'{CPZ}: NOA 条目，定义同上（Hirshleifer et al. 2004）',
        self="IVAO、DLC、DLTT、MIB、PSTK 缺失记 0。"),
    "DPI2A": dict(
        freq="每年",
        calc="(ΔPPEGT + ΔINVT) ÷ 上财年 AT。",
        src=f'{CPZ}: "Changes in property, plants, and equipment (PPEGT) and inventory (INVT) over '
            f'lagged total assets"（Lyandres et al. 2008）',
        self=None),
    # ---------------- 盈利 ----------------
    "PROF": dict(
        freq="每年",
        calc="GP ÷ BE（GP 缺失时用 REVT - COGS）。",
        src=f'{CPZ}: "Gross profitability (GP) divided by the book value of equity"'
            f'（Ball et al. 2015）',
        self="GP 缺失时用 REVT - COGS 代替。"),
    "CTO": dict(
        freq="每年",
        calc="SALE ÷ 上财年 AT。",
        src=f'{CPZ}: "Ratio of net sales (SALE) to lagged total assets (AT)"'
            f'（Haugen & Baker 1996）',
        self=None),
    "FC2Y": dict(
        freq="每年",
        calc="(XSGA + XRD + XAD) ÷ SALE。",
        src=f'{CPZ}: "Ratio of selling, general, and administrative expenses (XSGA), research and '
            f'development expenses (XRD), and advertising expenses (XAD) to net sales (SALE)"'
            f'（D\'Acunto et al. 2017）',
        self="XRD、XAD 缺失记 0（缺失率分别为 54%、68%）。"),
    "OP": dict(
        freq="每年",
        calc="(REVT - COGS - TIE - XSGA) ÷ BE。",
        src=f'{CPZ}: "Annual revenues (REVT) minus cost of goods sold (COGS), interest expense '
            f'(TIE), and selling, general, and administrative expenses (XSGA) divided by book '
            f'equity"（Fama & French 2015）',
        self="TIE 在本数据中 92% 缺失（主要只有金融类公司有值），XSGA 缺失 25%，缺失均记 0。"),
    "PM": dict(
        freq="每年",
        calc="OIADP ÷ SALE。",
        src=f'{CPZ}: "Operating income after depreciation (OIADP) over net sales (SALE)"'
            f'（Soliman 2008）',
        self=None),
    "RNA": dict(
        freq="每年",
        calc="OIADP ÷ 上财年净经营资产（经营资产 - 经营负债，定义同 NOA）。",
        src=f'{CPZ}: "Ratio of operating income after depreciation (OIADP) to lagged net operating '
            f'assets"（Soliman 2008）',
        self="上财年净经营资产 ≤ 0 时视为缺失。"),
    "D2A": dict(
        freq="每年",
        calc="DP ÷ AT。",
        src=f'{CPZ}: "Ratio of depreciation and amortization (DP) to total assets (AT)"'
            f'（Gorodnichenko & Weber 2016）',
        self=None),
    # ---------------- 无形资产 ----------------
    "OA": dict(
        freq="每年",
        calc="[Δ(ACT - CHE) - Δ(LCT - DLC - TXP) - DP] ÷ 上财年 AT。",
        src=f'{CPZ}: "Changes in non-cash working capital minus depreciation (DP) scaled by lagged '
            f'total assets"，非现金营运资本的定义见 AC 条目（Sloan 1996）',
        self="CPZ 的 AC 条目把各项都写成 \"minus\"，有歧义；这里按 Sloan (1996) 的标准公式，"
             "DLC 和 TXP 从流动负债中扣除。TXP 缺失记 0。"),
    "OL": dict(
        freq="每年",
        calc="(COGS + XSGA) ÷ AT。",
        src=f'{CPZ}: "Sum of cost of goods sold (COGS) and selling, general, and administrative '
            f'expenses (XSGA) over total assets (AT)"（Novy-Marx 2011）',
        self="XSGA 缺失记 0。"),
    "PCM": dict(
        freq="每年",
        calc="(SALE - COGS) ÷ SALE。",
        src=f'{CPZ}: "Difference between net sales (SALE) and costs of goods sold (COGS) divided by '
            f'net sales (SALE)"（Bustamante & Donangelo 2017）',
        self=None),
}
assert set(DOC) == set(CHARS)

t0 = time.time()


def tick():
    return f"[{time.time() - t0:.0f}s]"


def to_datetime(s):
    """pyreadstat 有时把 SAS 日期读成"自 1960-01-01 起的天数"，统一转成 datetime。"""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_datetime("1960-01-01") + pd.to_timedelta(s, unit="D")
    return pd.to_datetime(s, errors="coerce")   # linkenddt=9999-12-29（仍有效）-> NaT


def fin(a):
    """inf -> NaN"""
    a = np.asarray(a, dtype="float64")
    return np.where(np.isfinite(a), a, np.nan)


# ===========================================================================
# Part 0  读数据，转成宽表，建立月份索引
# ===========================================================================
print("Part 0  读数据")
cols = ["PERMNO", "DATE", "RET_ADJ", "PRC", "CFACPR", "VOL", "SHROUT", "me",
        "quoted_spread", "in_universe"]
p = pd.read_parquet(PANEL_FILE, columns=cols).set_index(["DATE", "PERMNO"])
W = {c: p[c].unstack() for c in cols[2:]}
del p
dates = W["RET_ADJ"].index
permnos = W["RET_ADJ"].columns
T, S = len(dates), len(permnos)
for c in W:
    W[c] = W[c].reindex(index=dates, columns=permnos)

R = W["RET_ADJ"].astype("float64")
U_raw = W.pop("in_universe")
E = U_raw.notna().values                      # 该 (日, 股票) 在日度面板中有记录
U = U_raw.fillna(0).values.astype(bool)       # 在股票池中
del U_raw

ff, _ = pyreadstat.read_sas7bdat(FF_FILE)
ff["date"] = to_datetime(ff["date"])
ff = ff.set_index("date").reindex(dates)
assert ff["mktrf"].notna().all()
rf = ff["rf"].values

# 月份索引：starts[m] = 第 m 个月第一个交易日的位置；mend[m] = 最后一个交易日的位置
per = dates.to_period("M")
mcode = pd.factorize(per)[0]
starts = np.flatnonzero(np.r_[True, mcode[1:] != mcode[:-1]])
mend = np.r_[starts[1:] - 1, T - 1]
months = per[starts]
M = len(starts)

# 输出的行：(a) 股票池内；(b) 第 m 月股票池中的股票，在第 m 月开始前 PRE_WINDOW 个交易日内的行
U_month = np.logical_or.reduceat(U, starts, axis=0)          # (M, S) 第 m 月在股票池中
NEED = U.copy()
for m_ in range(M):
    s0 = max(starts[m_] - PRE_WINDOW, 0)
    NEED[s0:starts[m_], U_month[m_]] = True
NEED &= E
ti, si = np.nonzero(NEED)
N_ROWS = len(ti)
IS_U = U[ti, si]
row_m = mcode[ti]                  # 每个输出行所在月份
print(f"  {T} 个交易日 × {S} 只股票，{M} 个月，输出行数 {N_ROWS:,}"
      f"（股票池内 {IS_U.sum():,}，进池前 {(~IS_U).sum():,}）{tick()}")


def msum(a):
    """日度宽表 (T×S) -> 每月合计 (M×S)，NaN 当 0。"""
    return np.add.reduceat(np.nan_to_num(np.asarray(a, dtype="float64")), starts, axis=0)


def prev(a, k=1):
    """月度数组向后移 k 个月（第 m 行变成第 m-k 个月的值）。"""
    out = np.full_like(a, np.nan)
    out[k:] = a[:-k]
    return out


raw = {}   # 特征名 -> 长度 N_ROWS 的向量（只含 t-1 及以前的信息）


def store_daily(name, A):
    """日度特征：按股票向前填充（论文：缺失用最近观测值）-> 滞后一天 -> 取股票池内的行。"""
    A = pd.DataFrame(fin(A), index=dates, columns=permnos).ffill().shift(1)
    raw[name] = A.values[ti, si].astype("float32")


def store_monthly(name, Mo):
    """月度特征：第 m 月末计算的值用于第 m+1 月每一天；缺失先用该股票最近一个月的值。"""
    Mo = pd.DataFrame(fin(Mo)).ffill().values
    v = np.full(N_ROWS, np.nan)
    ok = row_m >= 1
    v[ok] = Mo[row_m[ok] - 1, si[ok]]
    raw[name] = v.astype("float32")


def at_month_end(A):
    """日度宽表 -> 每月最后一个交易日的值（先按股票向前填充）(M×S)。"""
    return pd.DataFrame(np.asarray(A, dtype="float64")).ffill().values[mend]


# ===========================================================================
# Part 1  日度特征（4 个）：每天更新
# ===========================================================================
print("Part 1  日度特征")
Rv = R.values
LR = np.log1p(Rv)
PRC = W.pop("PRC").values.astype("float64")
VOL = W.pop("VOL").values.astype("float64")

store_daily("Ret_D1", Rv)
store_daily("Ret_W1", np.expm1(pd.DataFrame(LR).rolling(5, min_periods=4).sum().values))
store_daily("STD_W1", pd.DataFrame(Rv).rolling(5, min_periods=4).std().values)
store_daily("Vol", pd.DataFrame(VOL * PRC / 1e6).rolling(5, min_periods=4).sum().values)
print(f"  完成 {tick()}")

# ===========================================================================
# Part 2  月度特征（13 个）：每月末计算，下个月使用
# ===========================================================================
print("Part 2  月度特征")
validR = np.isfinite(Rv)

# ---- 2.1 过去收益 ----------------------------------------------------------
nR = msum(validR)
mret = np.where(nR >= 10, np.expm1(msum(LR)), np.nan)        # 自然月收益
store_monthly("ST_Rev", mret)
store_monthly("r2_1", mret)
LM = pd.DataFrame(np.log1p(mret))
store_monthly("r12_2", np.expm1(LM.shift(1).rolling(11, min_periods=8).sum().values))
store_monthly("r12_7", np.expm1(LM.shift(6).rolling(6, min_periods=5).sum().values))
store_monthly("r36_13", np.expm1(LM.shift(12).rolling(24, min_periods=18).sum().values))
del LM
print(f"  过去收益完成 {tick()}")

# ---- 2.2 规模、换手、价差、价格位置 ------------------------------------------
ME = W.pop("me").values.astype("float64")
ME_m = at_month_end(ME) / 1000.0                     # 月末市值，百万美元
store_monthly("LME", ME_m)
del ME

SHROUT = W.pop("SHROUT").values.astype("float64")
nV = msum(np.isfinite(VOL))
store_monthly("LTurnover", np.where(nV >= 10, msum(VOL) / (at_month_end(SHROUT) * 1000), np.nan))
del SHROUT

QS = W.pop("quoted_spread").values.astype("float64")
nQ = msum(np.isfinite(QS))
store_monthly("Spread", np.where(nQ >= 5, msum(QS) / np.maximum(nQ, 1), np.nan))
del QS

CF = W.pop("CFACPR").values.astype("float64")
Padj = PRC / np.where(CF > 0, CF, np.nan)
hi = pd.DataFrame(Padj).rolling(252, min_periods=120).max().values
store_monthly("Rel2High", at_month_end(Padj) / at_month_end(hi))
del CF, Padj, hi
print(f"  LME/LTurnover/Spread/Rel2High 完成 {tick()}")

# ---- 2.3 Variance：过去两个自然月日收益方差 ----------------------------------
R0 = np.nan_to_num(Rv)
n1, s1, ss1 = nR, msum(R0), msum(R0 ** 2)
n2, s2, ss2 = n1 + prev(n1), s1 + prev(s1), ss1 + prev(ss1)
with np.errstate(divide="ignore", invalid="ignore"):
    store_monthly("Variance", np.where(n2 >= 20, (ss2 - s2 ** 2 / n2) / (n2 - 1), np.nan))

# ---- 2.4 Resid_Var：过去两个自然月 FF3 回归残差方差 ---------------------------
Y = Rv - rf[:, None]                                  # 日超额收益
vY = np.isfinite(Y).astype("float64")
Y0 = np.nan_to_num(Y)
F = np.column_stack([np.ones(T), ff["mktrf"], ff["smb"], ff["hml"]])
G = np.zeros((M, S, 4, 4))
h = np.zeros((M, S, 4))
for k in range(4):
    h[..., k] = msum(Y0 * F[:, [k]])
    for l in range(k, 4):
        G[..., k, l] = G[..., l, k] = msum(vY * (F[:, k] * F[:, l])[:, None])
yy = msum(Y0 ** 2)
G2, h2, yy2 = G + np.nan_to_num(prev(G)), h + np.nan_to_num(prev(h)), yy + np.nan_to_num(prev(yy))
G2[0], h2[0], yy2[0] = np.nan, np.nan, np.nan        # 第一个月没有上一个月
nn = G2[..., 0, 0]
ok = nn >= 20
b = np.linalg.solve(G2[ok], h2[ok][..., None])[..., 0]
rv = np.full((M, S), np.nan)
rv[ok] = (yy2[ok] - (b * h2[ok]).sum(-1)) / (nn[ok] - 4)
store_monthly("Resid_Var", rv)
del G, h, G2, h2, yy, yy2, b, rv, Y0, vY
print(f"  Variance/Resid_Var 完成 {tick()}")

# ---- 2.5 Beta：Frazzini-Pedersen ---------------------------------------------
le = np.log1p(Rv) - np.log1p(rf)[:, None]             # 日度对数超额收益
lm = np.log1p(ff["mktrf"].values + rf) - np.log1p(rf)
sig_i = pd.DataFrame(le).rolling(252, min_periods=120).std().values
sig_m = pd.Series(lm).rolling(252, min_periods=120).std().values
le3 = pd.DataFrame(le).rolling(3, min_periods=3).sum()    # 重叠 3 日收益
lm3 = pd.Series(lm).rolling(3, min_periods=3).sum()
v3 = le3.notna()
X3 = v3.mul(lm3.fillna(0), axis=0)
Y3 = le3.fillna(0)
W5 = 1260


def rs(A):
    return A.rolling(W5, min_periods=1).sum()


n5 = rs(v3.astype("float64"))
Sx, Sy = rs(X3), rs(Y3)
cxy = rs(X3 * Y3) - Sx * Sy / n5
cxx = rs(X3 ** 2) - Sx ** 2 / n5
cyy = rs(Y3 ** 2) - Sy ** 2 / n5
rho = (cxy / np.sqrt(cxx * cyy)).where(n5 >= 750).values
beta = rho * sig_i / sig_m[:, None]
store_monthly("Beta", at_month_end(fin(beta)))
del le, sig_i, le3, v3, X3, Y3, n5, Sx, Sy, cxy, cxx, cyy, rho, beta
print(f"  Beta 完成 {tick()}")

# ---- 2.6 SUV：标准化异常成交量 -----------------------------------------------
x1 = np.where(validR, np.maximum(Rv, 0), 0.0)
x2 = np.where(validR, np.maximum(-Rv, 0), 0.0)
ok_d = (np.isfinite(VOL) & validR).astype("float64")
y = np.where(ok_d > 0, VOL, 0.0)
x1, x2 = x1 * ok_d, x2 * ok_d
mon = {"n": msum(ok_d), "a": msum(x1), "b": msum(x2), "c": msum(x1 ** 2), "d": msum(x2 ** 2),
       "p": msum(y), "q": msum(x1 * y), "r": msum(x2 * y), "yy": msum(y ** 2)}
est = {k: pd.DataFrame(v).rolling(3, min_periods=3).sum().shift(1).values for k, v in mon.items()}
n, a, b_, c, d = est["n"], est["a"], est["b"], est["c"], est["d"]
pp, q, r_, yyv = est["p"], est["q"], est["r"], est["yy"]
with np.errstate(divide="ignore", invalid="ignore"):
    # 回归 y = b0 + b1·x1 + b2·x2（x1·x2 恒为 0，可以解析求解）
    b0 = (pp - a * q / c - b_ * r_ / d) / (n - a ** 2 / c - b_ ** 2 / d)
    b1 = (q - a * b0) / c
    b2 = (r_ - b_ * b0) / d
    rss = yyv - (b0 * pp + b1 * q + b2 * r_)
    sig = np.sqrt(np.where(rss > 0, rss / (n - 3), np.nan))
    nm = mon["n"]
    resid_mean = (mon["p"] - b0 * nm - b1 * mon["a"] - b2 * mon["b"]) / nm
    suv = resid_mean / sig
suv = np.where((n >= 40) & (nm >= 10) & (c > 0) & (d > 0), suv, np.nan)
store_monthly("SUV", suv)
del x1, x2, ok_d, y, mon, est, b0, b1, b2, rss, sig, suv
print(f"  SUV 完成 {tick()}")

RET_T = Rv[ti, si].astype("float32")                 # t 日收益（目标）
RF_LAG = pd.Series(rf).shift(1).values[ti].astype("float32")
del R, Rv, LR, R0, PRC, VOL, W

# ===========================================================================
# Part 3  年度特征（22 个）：Fama-French 6 月惯例
#   财年截止日在 y-1 年的年报，从 y 年 7 月 1 日起使用（到下一份年报生效为止）；
#   与市值相关的比率用 y-1 年 12 月末的市值。
# ===========================================================================
print("Part 3  年度特征")
c, _ = pyreadstat.read_sas7bdat(CCM_FILE)
for col in ["datadate", "linkdt", "linkenddt"]:
    c[col] = to_datetime(c[col])
c = c[(c["datadate"] >= c["linkdt"]) & (c["linkenddt"].isna() | (c["datadate"] <= c["linkenddt"]))]
c["PERMNO"] = c["permno"].astype("int64")
c = c[c["PERMNO"].isin(set(permnos))].copy()
c["dyear"] = c["datadate"].dt.year
c = c.sort_values(["PERMNO", "datadate"])
c = c.drop_duplicates(["PERMNO", "fyear"], keep="last")
c = c.drop_duplicates(["PERMNO", "dyear"], keep="last")   # 同一日历年两份年报（改财年）取后一份
n_acct = len(c)

z = lambda s: s.fillna(0)

# 账面权益 BE（Fama-French 定义）
ps = c["pstkrv"].fillna(c["pstkl"]).fillna(c["pstk"])
sh = c["seq"].fillna(c["ceq"] + z(ps)).fillna(c["at"] - c["lt"])
c["BE"] = (sh + z(c["txditc"]) - z(ps)).where(lambda s: s > 0)
op_assets = c["at"] - c["che"] - z(c["ivao"])
op_liab = c["at"] - z(c["dlc"]) - z(c["dltt"]) - z(c["mib"]) - z(c["pstk"]) - c["ceq"]
c["noa_lvl"] = op_assets - op_liab

# 上一财年的值（只有相邻财年才算）
lagcols = ["at", "act", "che", "lct", "dlc", "txp", "ppegt", "invt", "noa_lvl"]
g = c.groupby("PERMNO")
consecutive = (c["fyear"] - g["fyear"].shift(1)) == 1
for v in lagcols:
    c[v + "_l"] = g[v].shift(1).where(consecutive)


def d_(v):
    return c[v] - c[v + "_l"]


at_ = c["at"].where(c["at"] > 0)
at_l = c["at_l"].where(c["at_l"] > 0)
sale = c["sale"].where(c["sale"] > 0)
dwc = (d_("act") - d_("che")) - (d_("lct") - d_("dlc") - (z(c["txp"]) - z(c["txp_l"])))

# 上年 12 月末市值（datadate 所在日历年的 12 月）
dec_pos = {mo.year: i for i, mo in enumerate(months) if mo.month == 12}
mi_dec = c["dyear"].map(dec_pos)
col_i = permnos.get_indexer(c["PERMNO"])
okd = mi_dec.notna().values
ME_dec = np.full(len(c), np.nan)
ME_dec[okd] = ME_m[mi_dec[okd].astype(int).values, col_i[okd]]
ME_dec = np.where(ME_dec > 0, ME_dec, np.nan)

A = pd.DataFrame({"PERMNO": c["PERMNO"].values})
A["eff"] = pd.to_datetime((c["dyear"] + 1).astype(str) + "-07-01").values
A["AT"] = at_.values
A["A2ME"] = c["at"].values / ME_dec
A["BEME"] = c["BE"].values / ME_dec
A["C"] = (c["che"] / at_).values
A["CF"] = ((c["ni"] + c["dp"] - c["wcapch"].fillna(dwc) - c["capx"]) / c["BE"]).values
A["CF2P"] = (c["ib"] + c["dp"] + z(c["txdb"])).values / ME_dec
A["Q"] = ((c["at"] - c["ceq"] - z(c["txdb"])).values + ME_dec) / at_.values
A["Lev"] = ((z(c["dltt"]) + z(c["dlc"])) / (z(c["dltt"]) + z(c["dlc"]) + c["seq"])).values
A["E2P"] = c["ib"].values / ME_dec
A["Investment"] = (c["at"] / at_l - 1).values
A["NOA"] = (c["noa_lvl"] / at_l).values
A["DPI2A"] = ((d_("ppegt") + d_("invt")) / at_l).values
A["PROF"] = (c["gp"].fillna(c["revt"] - c["cogs"]) / c["BE"]).values
A["CTO"] = (c["sale"] / at_l).values
A["FC2Y"] = ((c["xsga"] + z(c["xrd"]) + z(c["xad"])) / sale).values
A["OP"] = ((c["revt"] - c["cogs"] - z(c["tie"]) - z(c["xsga"])) / c["BE"]).values
A["PM"] = (c["oiadp"] / sale).values
A["RNA"] = (c["oiadp"] / c["noa_lvl_l"].where(c["noa_lvl_l"] > 0)).values
A["D2A"] = (c["dp"] / at_).values
A["OA"] = ((dwc - c["dp"]) / at_l).values
A["OL"] = ((c["cogs"] + z(c["xsga"])) / at_).values
A["PCM"] = ((c["sale"] - c["cogs"]) / sale).values
acct = [k for k in A.columns if k not in ("PERMNO", "eff")]
assert len(acct) == 22
A[acct] = A[acct].replace([np.inf, -np.inf], np.nan)
A = A.sort_values("eff")
del c

L = pd.DataFrame({"DATE": dates.values[ti], "PERMNO": permnos.values[si]})
m = pd.merge_asof(L, A, left_on="DATE", right_on="eff", by="PERMNO", direction="backward")
assert len(m) == N_ROWS and (m["PERMNO"].values == L["PERMNO"].values).all()
acct_match = m["eff"].notna().values
for v in acct:
    raw[v] = m[v].values.astype("float32")
del m, A, L
assert set(raw) == set(CHARS), set(CHARS) ^ set(raw)
print(f"  完成 {tick()}")

# ===========================================================================
# Part 4  缺失值填补 + 截面排序 + 截面中位数 + rf -> 79 维
# ===========================================================================
print("Part 4  填补、排序 -> 79 维")
DATE = dates.values[ti]
PERMNO = permnos.values[si]
year = pd.DatetimeIndex(DATE).year.values
sel90 = (year >= 1990) & IS_U                         # 统计只看股票池内的行
Xu90 = pd.DataFrame({k: raw[k][sel90] for k in CHARS})
coverage = Xu90.notna().mean()                        # 填中位数之前的覆盖率（1990+）
q_tbl = Xu90.quantile([0.01, 0.5, 0.99]).T
del Xu90

# 截面中位数：每天只用股票池内的股票（T × 39 的小表）
day_u = ti[IS_U]
MED = pd.DataFrame({k: raw[k][IS_U] for k in CHARS}).groupby(day_u).median()
MED = MED.reindex(range(T)).values.astype("float32")    # (T, 39)，第 d 行 = 第 d 天的中位数

# 截面排序：相对于当天股票池内的分布
#   股票池内的股票：平均秩 r（并列取平均），q = (r-1)/(n-1) - 0.5，与原来的做法完全一致
#   股票池外的股票：把它"插入"当天股票池，q = (#小于 + #等于/2) / n - 0.5，超出范围自然为 ±0.5
Xr = np.zeros((N_ROWS, len(CHARS)), dtype="float32")
n_u = np.bincount(day_u, minlength=T)
for j, k in enumerate(CHARS):
    v = raw.pop(k).astype("float64")
    v = np.where(np.isnan(v), MED[ti, j], v)            # 缺失 -> 当天股票池中位数
    ok = ~np.isnan(v)                                   # 当天整列缺失时仍为 NaN
    g = np.full(N_ROWS, -1, dtype="int64")
    g[ok] = np.unique(v[ok], return_inverse=True)[1]    # 全局"值的名次"，保持大小顺序
    G = int(g.max()) + 2
    key = ti.astype("int64") * G + g
    ukey = np.sort(key[IS_U & ok])
    day_start = np.searchsorted(ukey, np.arange(T, dtype="int64") * G, side="left")
    less = np.searchsorted(ukey, key, side="left") - day_start[ti]
    leq = np.searchsorted(ukey, key, side="right") - day_start[ti]
    eq = leq - less
    n = n_u[ti].astype("float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        q_in = (less + (eq - 1) / 2.0) / (n - 1) - 0.5
        q_out = (less + eq / 2.0) / n - 0.5
    q = np.where(IS_U, q_in, q_out)
    q = np.where(ok & np.isfinite(q), q, 0.0)           # 整天缺失的特征记 0
    Xr[:, j] = q.astype("float32")
del raw
Xr = pd.DataFrame(Xr, columns=CHARS)
print(f"  完成 {tick()}")

# ===========================================================================
# Part 5  数据检查
# ===========================================================================
print("Part 5  数据检查")
# 泄露检查：特征（t-1 信息）与 t 日收益的截面秩相关（IC）。
# 如果误用了 t 日信息，Ret_D1 的 IC 会接近 1；正常应该很小。
yr = pd.Series(RET_T).where(IS_U).groupby(DATE).rank(pct=True).values - 0.5
ok_y = np.isfinite(yr) & sel90
Xs, ys, ds = Xr[ok_y], yr[ok_y], DATE[ok_y]
num = Xs.mul(ys, axis=0).groupby(ds).mean()
den = np.sqrt((Xs ** 2).groupby(ds).mean().mul(pd.Series(ys ** 2).groupby(ds).mean().values, axis=0))
ic = (num / den).mean()
del Xs, num, den
d90 = np.flatnonzero(dates.year >= 1990)
med_missing_90 = pd.Series(np.isnan(MED[d90]).any(0), index=["med_" + k for k in CHARS])
n_per_day = pd.Series(DATE[IS_U]).value_counts()
# 股票池外的行：排序值是否落在 [-0.5, 0.5]，以及取到边界的比例
out_rows = ~IS_U
out_at_edge = (np.abs(Xr.values[out_rows]) >= 0.5 - 1e-6).mean() if out_rows.any() else 0.0
out_range_ok = bool((np.abs(Xr.values) <= 0.5 + 1e-6).all())

# ===========================================================================
# Part 6  保存 + 写 log
# ===========================================================================
feat_cols = CHARS + ["med_" + k for k in CHARS] + ["rf"]
assert len(feat_cols) == 79
CHUNK = 1_000_000
writer = None
for a in range(0, N_ROWS, CHUNK):
    b = min(a + CHUNK, N_ROWS)
    part = pd.concat([
        pd.DataFrame({"DATE": DATE[a:b], "PERMNO": PERMNO[a:b], "RET_ADJ": RET_T[a:b],
                      "in_universe": IS_U[a:b].astype("int8")}),
        Xr.iloc[a:b].reset_index(drop=True),
        pd.DataFrame(MED[ti[a:b]], columns=["med_" + k for k in CHARS]),
        pd.DataFrame({"rf": RF_LAG[a:b]})], axis=1)
    tbl = pa.Table.from_pandas(part, preserve_index=False)
    if writer is None:
        meta = dict(tbl.schema.metadata or {})
        meta[b"feature_groups"] = json.dumps(FEATURE_GROUPS).encode()
        meta[b"feature_cols"] = json.dumps(feat_cols).encode()
        schema = tbl.schema.with_metadata(meta)
        writer = pq.ParquetWriter(OUT_FILE, schema, compression="zstd")
    writer.write_table(tbl.cast(schema), row_group_size=500_000)
writer.close()
n_out_cols = 4 + len(feat_cols)
size_mb = os.path.getsize(OUT_FILE) / 1e6

GROUP_CN = {"past_returns": "过去收益", "trading_frictions": "交易摩擦", "value": "价值",
            "investment": "投资", "profitability": "盈利", "intangibles": "无形资产"}
L_ = []
w = L_.append
w("=" * 78)
w("第 2 步：79 维特征说明（Attention Factors for Statistical Arbitrage 复现）")
w("=" * 78)
w("")
w("一、参考文献")
w("  [ATT] Epstein, Wang, Choi, Pelger (2025). Attention Factors for Statistical Arbitrage.")
w("        ICAIF '25, arXiv:2510.11616。Table 1 列出 39 个特征；4.1 节说明数据处理，")
w("        并写明特征构造细节见 [CPZ] 的 Internet Appendix。")
w("  [CPZ] Chen, Pelger, Zhu (2024). Deep Learning in Asset Pricing. Management Science 70(2)。")
w("        使用 arXiv:1904.00745v3 版本：正文 A.1 节（第 24-26 页）与 Appendix B 的")
w("        Table A.VII（第 65-68 页，46 个特征的逐条定义）。")
w("  对应关系：[ATT] 的 39 个 = [CPZ] 的 46 个去掉 AC, ATO, D2P, IdioVol, LT_Rev, MktBeta,")
w("        NI, ROA, ROE, S2P, SGA2S 共 11 个，再加上 Ret_D1, Ret_W1, STD_W1, Vol 共 4 个。")
w("")
w("二、通用规则")
w("  1. 股票池：每月按上月末市值取前 500 只普通股。")
w("     出处 [ATT] 4.1 节: \"we consider only the 500 largest stocks based on the previous")
w("     month market capitalization\"。")
w("  2. 更新频率与时点：")
w("     出处 [CPZ] A.1 节第 24 页: \"Yearly updated variables are updated at the end of each")
w("     June following the Fama-French convention, while monthly changing variables are")
w("     updated at the end of each month for the use in the next month.\"")
w("     本代码实现：")
w("       - 年度特征（22 个会计类）：财年截止日在 y-1 年的年报，从 y 年 7 月 1 日起使用，")
w("         直到下一份年报生效；与市值相关的比率用 y-1 年 12 月末的市值。")
w("       - 月度特征（13 个）：第 m 月最后一个交易日计算，用于第 m+1 月的每一天。")
w("       - 日度特征（4 个，[ATT] 新增）：每天更新，只用 t-1 日及以前的数据。")
w("         [ATT] 没有写这 4 个特征的更新频率，按名称（daily/weekly）自行设定为每天更新。")
w("  3. 时间对齐：输出文件中日期为 t 的一行，特征只用到 t-1 日及以前的信息，")
w("     收益 RET_ADJ 是 t 日的，即每行就是 (X_{t-1}, R_t)。")
w("  4. 缺失值：先用该股票最近一次的观测值，没有则用当天截面中位数。")
w("     出处 [ATT] 4.1 节: \"Missing values in the characteristics are imputed with last")
w("     observed values if available and by the cross-sectional median otherwise.\"")
w("     （[CPZ] 的做法是直接删掉有缺失的股票，这里按 [ATT]。）")
w("  5. 标准化：每天在股票池内做截面排序，转成分位数后线性映射到 [-0.5, 0.5]。")
w("     出处 [ATT] 4.1 节: \"Firm characteristics are normalized to rank quantiles\"；")
w("     [CPZ] 第 26 页: \"we rank them cross-sectionally and convert them into quantiles\"。")
w("     映射区间 [-0.5, 0.5] 原文未写明，为自行设定。")
w("  6. 79 维 = 39 个排序值 + 39 个截面中位数 + 无风险利率。")
w("     出处 [ATT] 4.1-4.2 节: \"we also include the cross-sectional median of the")
w("     characteristics and the risk free rate, which results in the dimension of X_{t-1} of 79\"。")
w("     中位数保留原始量纲。[CPZ] 对中位数序列做了平稳化变换（如对数差分），[ATT] 没有提，")
w("     这里不做变换，留到建模阶段统一标准化。无风险利率用 FF 日度 rf（t-1 日）。")
w("  7. 收益：CRSP 日收益并入退市收益（第 1 步），市场与 FF3 因子用 ff_daily。")
w(f"  8. 进池前的行（in_universe = 0）：第 m 月股票池中的股票，在第 m 月开始前 {PRE_WINDOW} 个交易日内")
w("     的行也输出。特征的计算方法完全相同；排序值相对于当天股票池 500 只的分布计算")
w("     （把该股票\"插入\"当天股票池后的分位位置），截面中位数也用当天股票池的。")
w("     用途：环节④ 每天用当天的 ω^ε 计算当天的残差（[19] 的做法），新进股票在进池前")
w("     那些天也需要特征。这一类行原文未涉及，属于实现上的需要。")
w("")
w("三、39 个特征逐条说明")
w("  （覆盖率 = 1990 年以后股票池内、用最近观测值填补后、用截面中位数填补前的非缺失比例）")
i = 0
for gname, feats in FEATURE_GROUPS.items():
    w("")
    w(f"  ---- {GROUP_CN[gname]}（{gname}，{len(feats)} 个）" + "-" * 40)
    for f in feats:
        i += 1
        dd = DOC[f]
        w("")
        w(f"  [{i:2d}] {f}    更新频率: {dd['freq']}    覆盖率: {coverage[f]:.1%}")
        w(f"       计算方法: {dd['calc']}")
        w(f"       原文出处: {dd['src']}")
        if dd["self"] is None:
            w("       自行设定: 无，完全按原文")
        else:
            w(f"       自行设定: {dd['self']}")
w("")
w("四、运行结果")
w(f"  输出文件: {OUT_FILE}（{size_mb:.0f} MB）")
w(f"  行数: {N_ROWS:,}（股票池内 {IS_U.sum():,}，进池前 {(~IS_U).sum():,}），"
  f"列数: {n_out_cols} = DATE, PERMNO, RET_ADJ, in_universe + 79 维特征")
w(f"  进池前的行：排序值全部在 [-0.5, 0.5] 内: {'是' if out_range_ok else '否'}；"
  f"取到边界 ±0.5 的比例 {out_at_edge:.1%}（这些股票当时多为市值较小的股票）")
w(f"  日期: {pd.Timestamp(DATE.min()).date()} -> {pd.Timestamp(DATE.max()).date()}；股票池内"
  f"每天股票数 {n_per_day.min()}-{n_per_day.max()}")
w(f"  Compustat 年报（链接有效、属于样本股票）: {n_acct:,} 条；"
  f"股票池行中匹配到年报的比例（1990+）: {acct_match[sel90].mean():.1%}")
w(f"  1990 年以后截面中位数列是否有缺失: "
  f"{', '.join(med_missing_90[med_missing_90].index) or '无'}")
w(f"  1985-1989 年部分特征整天缺失（回看期不足、Compustat 从 1987 财年才开始），")
w(f"  这些天的排序值记为 0；模型从 1990 年开始使用，不受影响。")
w(f"  泄露检查（特征与 t 日收益的平均截面秩相关，1990+）: 最大绝对值 {ic.abs().max():.4f}"
  f"（{ic.abs().idxmax()}）。若误用了 t 日信息，Ret_D1 会接近 1；")
w(f"    Ret_D1 = {ic['Ret_D1']:.4f}, ST_Rev = {ic['ST_Rev']:.4f}, r12_2 = {ic['r12_2']:.4f}")
w("")
w("  各特征原始值分位数（1990+，填补前）:")
for line in q_tbl.to_string(float_format=lambda v: f"{v:.4g}").splitlines():
    w("    " + line)
w("")
w(f"  总耗时 {tick()}")

with open(LOG_PATH, "w") as fh:
    fh.write("\n".join(L_) + "\n")
print("\n".join(L_[-60:]))
