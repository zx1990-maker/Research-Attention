"""
第 1 步：构建日度面板数据（Attention Factors for Statistical Arbitrage 复现）

输入（wrds/ 文件夹）：
    crsp_daily.sas7bdat          CRSP 日度行情
    monthly_top500_prior.sas7bdat 每月前 500 名单（按上月末市值排名）
    crsp_daily_delist.sas7bdat   退市收益

输出（data/ 文件夹）：
    daily_panel.parquet  所有"曾经进入过前 500"的股票在 1985-2021 年的全部日度数据，
                         并带 in_universe 标记（当天是否在当月的前 500 股票池中）
    universe_monthly.parquet  每月前 500 名单（PERMNO, eval_month）
    01_build_daily_panel.log  运行日志与数据检查结果

用法（在 research 目录下）：
    python data/coding/01_build_daily_panel.py
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyreadstat

HERE = os.path.dirname(os.path.abspath(__file__))      # research/data/coding
ROOT = os.path.dirname(os.path.dirname(HERE))          # research
WRDS = os.path.join(ROOT, "wrds")
OUT = os.path.join(ROOT, "data")
os.makedirs(OUT, exist_ok=True)

DAILY_FILE = os.path.join(WRDS, "crsp_daily.sas7bdat")
DAILY_RAW_FILE = os.path.join(WRDS, "crsp_daily_raw.sas7bdat")
UNIVERSE_FILE = os.path.join(WRDS, "monthly_top500_prior.sas7bdat")
DELIST_FILE = os.path.join(WRDS, "crsp_daily_delist.sas7bdat")

DAILY_COLS = ["PERMNO", "DATE", "SICCD", "EXCHCD", "RET", "RETX", "abs_prc",
              "SHROUT", "me", "VOL", "BID", "ASK", "quoted_spread", "CFACPR"]
CHUNKSIZE = 2_000_000

# Shumway (1997)：业绩原因退市（DLSTCD 500, 520-584）且 DLRET 缺失时，用 -30% 代替
SHUMWAY_CODES = {500} | set(range(520, 585))
SHUMWAY_RET = -0.30

LOG_PATH = os.path.join(OUT, "01_build_daily_panel.log")
_log_f = open(LOG_PATH, "w")


def log(msg=""):
    print(msg)
    _log_f.write(str(msg) + "\n")
    _log_f.flush()


def to_datetime(s):
    """pyreadstat 有时把 SAS 日期读成"自 1960-01-01 起的天数"，统一转成 datetime。"""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_datetime("1960-01-01") + pd.to_timedelta(s, unit="D")
    return pd.to_datetime(s)


t0 = time.time()

# ---------------------------------------------------------------------------
# 1. 股票池：每月前 500 名单
# ---------------------------------------------------------------------------
log("== 1. 读取每月前 500 名单")
uni, _ = pyreadstat.read_sas7bdat(UNIVERSE_FILE, usecols=["PERMNO", "eval_month", "in_top500"])
uni["eval_month"] = to_datetime(uni["eval_month"])
uni = uni[uni["in_top500"] == 1][["PERMNO", "eval_month"]].copy()
uni["PERMNO"] = uni["PERMNO"].astype("int64")
uni["ym"] = uni["eval_month"].dt.to_period("M")
uni = uni.drop_duplicates(["PERMNO", "ym"])
uni[["PERMNO", "eval_month"]].to_parquet(os.path.join(OUT, "universe_monthly.parquet"), index=False)

keep_permnos = set(uni["PERMNO"].unique())
log(f"月份范围: {uni['eval_month'].min().date()} -> {uni['eval_month'].max().date()}, "
    f"共 {uni['ym'].nunique()} 个月")
log(f"曾进入前 500 的股票数: {len(keep_permnos):,}")
log(f"每月股票数 min/max: {uni.groupby('ym').size().min()} / {uni.groupby('ym').size().max()}")

# ---------------------------------------------------------------------------
# 2. 分块读取日度数据，只保留曾进入前 500 的股票
# ---------------------------------------------------------------------------
_, meta = pyreadstat.read_sas7bdat(DAILY_FILE, metadataonly=True)
_, meta_raw = pyreadstat.read_sas7bdat(DAILY_RAW_FILE, metadataonly=True)
log(f"\n== 2. 分块读取 crsp_daily（{meta.number_rows:,} 行；"
    f"crsp_daily_raw 为 {meta_raw.number_rows:,} 行）")

parts = []
n_read = 0
reader = pyreadstat.read_file_in_chunks(pyreadstat.read_sas7bdat, DAILY_FILE,
                                        chunksize=CHUNKSIZE, usecols=DAILY_COLS)
for chunk, _ in reader:
    n_read += len(chunk)
    chunk = chunk[chunk["PERMNO"].isin(keep_permnos)].copy()
    chunk["DATE"] = to_datetime(chunk["DATE"])
    parts.append(chunk)
    log(f"  已读 {n_read:,} / {meta.number_rows:,} 行，累计保留 "
        f"{sum(len(p) for p in parts):,} 行 [{time.time() - t0:.0f}s]")

df = pd.concat(parts, ignore_index=True)
del parts
df["PERMNO"] = df["PERMNO"].astype("int64")
df = df.rename(columns={"abs_prc": "PRC"})

n_before = len(df)
df = df.sort_values(["PERMNO", "DATE"]).drop_duplicates(["PERMNO", "DATE"], keep="last")
log(f"去除重复 (PERMNO, DATE): {n_before - len(df):,} 行")
log(f"保留 {len(df):,} 行, 日期 {df['DATE'].min().date()} -> {df['DATE'].max().date()}")

# ---------------------------------------------------------------------------
# 3. 并入退市收益
# ---------------------------------------------------------------------------
log("\n== 3. 并入退市收益")
dl, _ = pyreadstat.read_sas7bdat(DELIST_FILE)
dl["DLSTDT"] = to_datetime(dl["DLSTDT"])
dl["PERMNO"] = dl["PERMNO"].astype("int64")
dl = dl[dl["PERMNO"].isin(keep_permnos)].copy()

n_missing_dlret = dl["DLRET"].isna().sum()
shumway_mask = dl["DLRET"].isna() & dl["DLSTCD"].isin(SHUMWAY_CODES)
dl.loc[shumway_mask, "DLRET"] = SHUMWAY_RET
dl = dl.dropna(subset=["DLRET"])
log(f"相关股票的退市记录 {len(dl) + (n_missing_dlret - shumway_mask.sum()):,} 条，"
    f"DLRET 缺失 {n_missing_dlret:,} 条，其中 {shumway_mask.sum():,} 条按 Shumway 规则填 -30%，"
    f"其余丢弃")

# 退市日如果不是交易日（常见：晚于最后一个交易日），归到该股票最后一个交易日
last_day = df.groupby("PERMNO")["DATE"].max().rename("last_date")
dl = dl.join(last_day, on="PERMNO")
dl = dl.dropna(subset=["last_date"])
on_trading_day = dl.merge(df[["PERMNO", "DATE"]], left_on=["PERMNO", "DLSTDT"],
                          right_on=["PERMNO", "DATE"], how="left")["DATE"].notna().values
dl["merge_date"] = np.where(on_trading_day, dl["DLSTDT"], dl["last_date"])
# 退市日早于最后交易日却不是交易日的极少数情况，也归到最后交易日
dl["merge_date"] = pd.to_datetime(dl["merge_date"])
dl = dl[["PERMNO", "merge_date", "DLRET"]].rename(columns={"merge_date": "DATE"})
dl = dl.drop_duplicates(["PERMNO", "DATE"])

df = df.merge(dl, on=["PERMNO", "DATE"], how="left")
has_dl = df["DLRET"].notna()
df["RET_ADJ"] = df["RET"]
both = has_dl & df["RET"].notna()
only_dl = has_dl & df["RET"].isna()
df.loc[both, "RET_ADJ"] = (1 + df.loc[both, "RET"]) * (1 + df.loc[both, "DLRET"]) - 1
df.loc[only_dl, "RET_ADJ"] = df.loc[only_dl, "DLRET"]
log(f"并入退市收益: {has_dl.sum():,} 行（与当日 RET 合成 {both.sum():,} 行，"
    f"RET 缺失直接用 DLRET {only_dl.sum():,} 行）")

# ---------------------------------------------------------------------------
# 4. 股票池标记：DATE 在 M 月，且该股票在 eval_month = M 的前 500 中
# ---------------------------------------------------------------------------
log("\n== 4. 打股票池标记")
df["ym"] = df["DATE"].dt.to_period("M")
uni["in_universe"] = np.int8(1)
df = df.merge(uni[["PERMNO", "ym", "in_universe"]], on=["PERMNO", "ym"], how="left")
df["in_universe"] = df["in_universe"].fillna(0).astype("int8")
df = df.drop(columns="ym")

# ---------------------------------------------------------------------------
# 5. 数据检查
# ---------------------------------------------------------------------------
log("\n== 5. 数据检查")
u = df[df["in_universe"] == 1]
per_day = u.groupby("DATE").agg(n=("PERMNO", "size"), n_ret=("RET_ADJ", lambda s: s.notna().sum()))
log("股票池内每个交易日的股票数（有收益的）:")
log(per_day["n_ret"].describe().round(1).to_string())
low_days = per_day[per_day["n_ret"] < 480]
log(f"有收益股票数 < 480 的交易日: {len(low_days)} 天")
if len(low_days):
    log(low_days.head(10).to_string())

log(f"\n股票池内 RET_ADJ 缺失比例: {u['RET_ADJ'].isna().mean():.4%}")
log("股票池内 RET_ADJ 分位数:")
log(u["RET_ADJ"].quantile([0, .001, .01, .5, .99, .999, 1]).round(4).to_string())
extreme = u[(u["RET_ADJ"] > 1.0) | (u["RET_ADJ"] < -0.6)]
log(f"单日收益 > +100% 或 < -60% 的记录: {len(extreme)} 条")
if len(extreme):
    log(extreme[["PERMNO", "DATE", "RET", "DLRET", "RET_ADJ", "PRC"]].head(15).to_string())

idx, _ = pyreadstat.read_sas7bdat(os.path.join(WRDS, "crsp_daily_index.sas7bdat"))
idx["DATE"] = to_datetime(idx["DATE"])
ew = u.groupby("DATE")["RET_ADJ"].mean().rename("ew_top500")
cmp_ = idx.set_index("DATE").join(ew, how="inner")
log(f"\n前 500 等权日收益 vs CRSP 全市场等权 (ewretd) 相关系数: "
    f"{cmp_['ew_top500'].corr(cmp_['ewretd']):.4f}")
log(f"前 500 等权日收益 vs CRSP 市值加权 (vwretd) 相关系数: "
    f"{cmp_['ew_top500'].corr(cmp_['vwretd']):.4f}")
log(f"年化平均收益: 前 500 等权 {cmp_['ew_top500'].mean() * 252:.2%}, "
    f"ewretd {cmp_['ewretd'].mean() * 252:.2%}, vwretd {cmp_['vwretd'].mean() * 252:.2%}")

# ---------------------------------------------------------------------------
# 6. 保存
# ---------------------------------------------------------------------------
df = df.sort_values(["DATE", "PERMNO"]).reset_index(drop=True)
out_path = os.path.join(OUT, "daily_panel.parquet")
pq.write_table(pa.Table.from_pandas(df, preserve_index=False), out_path,
               compression="zstd", row_group_size=500_000)
log(f"\n== 6. 已保存 {out_path}")
log(f"共 {len(df):,} 行, {df['PERMNO'].nunique():,} 只股票, 列: {list(df.columns)}")
log(f"文件大小: {os.path.getsize(out_path) / 1e6:.0f} MB, 总耗时 {time.time() - t0:.0f}s")
_log_f.close()
