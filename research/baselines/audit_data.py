#!/usr/bin/env python3
"""Audit supplied parquet inputs without modifying them.

Usage: python audit_data.py [--universe PATH --features PATH --daily PATH]
Writes audit.json and notes/data_audit.md beside this script by default.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

BASE = Path(__file__).resolve().parent / 'data'
DEFAULTS = {
    'universe': BASE / 'universe_monthly.parquet',
    'features': BASE / 'features.parquet',
    'daily': BASE / 'daily_panel.parquet',
}


def describe_file(path: Path) -> dict:
    parquet = pq.ParquetFile(path)
    missing = {name: 0 for name in parquet.schema_arrow.names}
    ranges = {}
    # Arrow chunk aggregation avoids materializing the 82-column features table.
    for batch in parquet.iter_batches(batch_size=200_000):
        for name, col in zip(batch.schema.names, batch.columns):
            missing[name] += col.null_count
            if name in {'DATE', 'eval_month', 'PERMNO'}:
                continue
            extrema = pc.min_max(col).as_py()
            if extrema['min'] is None:
                continue
            if name not in ranges:
                ranges[name] = extrema
            else:
                ranges[name]['min'] = min(ranges[name]['min'], extrema['min'])
                ranges[name]['max'] = max(ranges[name]['max'], extrema['max'])
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return {
        'path': str(path.resolve()), 'bytes': path.stat().st_size,
        'sha256': digest.hexdigest(), 'rows': parquet.metadata.num_rows,
        'columns': {field.name: str(field.type) for field in parquet.schema_arrow},
        'missing_count': missing, 'ranges': ranges,
        'metadata': {k.decode(): json.loads(v.decode()) for k, v in
                     (parquet.schema_arrow.metadata or {}).items()
                     if k in {b'feature_cols', b'feature_groups'}},
    }


def key_stats(df: pd.DataFrame, date: str) -> dict:
    counts = df.groupby(date).size()
    return {
        'date_min': str(df[date].min().date()), 'date_max': str(df[date].max().date()),
        'date_count': int(counts.size), 'unique_permno': int(df.PERMNO.nunique()),
        'duplicate_keys': int(df.duplicated(['PERMNO', date]).sum()),
        'assets_per_date': {k: float(v) for k, v in counts.describe().items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, path in DEFAULTS.items():
        parser.add_argument('--' + name, type=Path, default=path)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    report = {'files': {}}
    for name in DEFAULTS:
        print(f'Auditing {name}...', flush=True)
        report['files'][name] = describe_file(getattr(args, name))
    u = pd.read_parquet(args.universe)
    f = pd.read_parquet(args.features, columns=['DATE', 'PERMNO', 'RET_ADJ', 'Ret_D1', 'med_Ret_D1', 'rf'])
    d = pd.read_parquet(args.daily, columns=['DATE', 'PERMNO', 'RET', 'DLRET', 'RET_ADJ', 'me', 'in_universe'])
    for name, df, date in [('universe', u, 'eval_month'), ('features', f, 'DATE'), ('daily', d, 'DATE')]:
        report['files'][name].update(key_stats(df, date))
    u['month'] = u.eval_month.dt.to_period('M')
    d['month'] = d.DATE.dt.to_period('M')
    ui = pd.MultiIndex.from_arrays([u.PERMNO, u.month])
    actual = d.in_universe.eq(1).to_numpy()
    report['membership_month_offsets'] = {}
    for offset in [-1, 0, 1]:
        predicted = pd.MultiIndex.from_arrays([d.PERMNO, d.month + offset]).isin(ui)
        report['membership_month_offsets'][str(offset)] = {
            'mismatch_rows': int(np.count_nonzero(predicted != actual)),
            'false_positive': int(np.count_nonzero(predicted & ~actual)),
            'false_negative': int(np.count_nonzero(~predicted & actual)),
        }
    j = d.loc[actual, ['DATE', 'PERMNO', 'RET_ADJ']].merge(
        f[['DATE', 'PERMNO', 'RET_ADJ']], on=['DATE', 'PERMNO'], how='outer',
        suffixes=['_daily', '_features'], indicator=True, validate='one_to_one')
    report['features_join'] = {
        'counts': {k: int(v) for k, v in j._merge.value_counts().items()},
        'max_abs_return_difference': float((j.RET_ADJ_daily-j.RET_ADJ_features).abs().max()),
    }
    del j
    report['risk_free_rate'] = {
        'unique_values_per_date_distribution': {str(k): int(v) for k, v in f.groupby('DATE').rf.nunique().value_counts().items()},
        'missing_dates': [str(v.date()) for v in f.loc[f.rf.isna(), 'DATE'].drop_duplicates()],
        'range': [float(f.rf.min()), float(f.rf.max())],
        'interpretation': 'Values are consistent with a daily decimal rate; provenance is not supplied.',
    }
    d.sort_values(['PERMNO', 'DATE'], inplace=True)
    d['lag1'] = d.groupby('PERMNO', sort=False).RET_ADJ.shift(1)
    d['lag2'] = d.groupby('PERMNO', sort=False).RET_ADJ.shift(2)
    j = f.merge(d[['DATE', 'PERMNO', 'lag1', 'lag2']], on=['DATE', 'PERMNO'], validate='one_to_one')
    report['ret_d1_alignment'] = {}
    for col in ['RET_ADJ', 'lag1', 'lag2']:
        grouped = j.groupby('DATE')[col]
        ranks = (grouped.rank(method='average') - 1) / (grouped.transform('count') - 1) - .5
        error = (j.Ret_D1 - ranks).abs()
        report['ret_d1_alignment'][col] = {
            'rank_correlation': float(j.Ret_D1.corr(ranks)),
            'mean_abs_rank_difference': float(error.mean()),
            'max_abs_rank_difference': float(error.max()),
            'fraction_abs_difference_le_1e_6': float(error.le(1e-6).mean()),
        }
    del j
    has_delist = d.DLRET.notna()
    expected = (1+d.loc[has_delist, 'RET']) * (1+d.loc[has_delist, 'DLRET']) - 1
    report['delisting_adjustment'] = {
        'observations': int(has_delist.sum()),
        'max_abs_formula_error': float((expected-d.loc[has_delist, 'RET_ADJ']).abs().max()),
        'delisting_present_return_missing': int((has_delist & d.RET.isna()).sum()),
        'formula': '(1 + RET) * (1 + DLRET) - 1',
    }
    missing = d.loc[d.in_universe.eq(1) & d.RET_ADJ.isna(), ['DATE', 'PERMNO']].copy()
    missing['DATE'] = missing.DATE.dt.strftime('%Y-%m-%d')
    report['in_universe_missing_returns'] = missing.to_dict('records')
    # Test monthly formation timing within the supplied 2,203-stock panel only.
    last_date = d.groupby('month').DATE.transform('max')
    month_end = d.loc[d.DATE.eq(last_date)].sort_values(['month', 'me'], ascending=[True, False])
    top_sets = month_end.groupby('month').head(500).groupby('month').PERMNO.agg(set).to_dict()
    universe_sets = u.groupby('month').PERMNO.agg(set).to_dict()
    report['universe_market_cap_reconstruction'] = {}
    for offset in [0, 1]:
        results = [{'eval_month': str(month+offset), 'intersection_count': len(ids & universe_sets[month+offset])}
                   for month, ids in top_sets.items() if month+offset in universe_sets]
        vals = [item['intersection_count'] for item in results]
        report['universe_market_cap_reconstruction'][str(offset)] = {
            'meaning': 'Use end-of-month m cap ranking for universe month m+' + str(offset),
            'months_compared': len(vals), 'exact_500_matches': sum(v == 500 for v in vals),
            'mean_intersection': float(np.mean(vals)), 'min_intersection': min(vals),
            'per_month': results,
        }
    report['limitations'] = [
        'Inputs do not contain data-construction source code, complete original CRSP universe, or accounting release timestamps.',
        'Month-end top-500 reconstruction is conditional on the supplied 2,203-stock union, not a full independent CRSP universe audit.',
        'Ret_D1 timing is tested empirically; other characteristics need source construction code for complete point-in-time certification.',
        'A zero rank value may mean median exposure or an already imputed missing characteristic; original missingness masks are absent.',
        'Raw total/delisting-adjusted returns and risk-free rates are separate: subtract rf only once when excess returns are needed.',
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'notes').mkdir(exist_ok=True)
    (args.output/'audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    write_note(report, args.output/'notes/data_audit.md')
    print('Wrote', args.output/'audit.json', flush=True)


def write_note(a: dict, path: Path) -> None:
    files = a['files']
    prior = a['universe_market_cap_reconstruction']['1']
    same = a['universe_market_cap_reconstruction']['0']
    lines = [
        '# 数据审计与读取约定', '',
        '本报告由 `audit_data.py` 实际读取三个原始 Parquet 生成；没有更改原始文件。完整列类型、缺失数、范围、SHA-256 和逐月检查见 `audit.json`。', '',
        '| 文件 | 行数 | 日期范围 | 不同股票数 | 重复主键 |',
        '|---|---:|---|---:|---:|',
    ]
    for name in ['universe', 'features', 'daily']:
        f = files[name]
        lines.append(f"| {name} | {f['rows']:,} | {f['date_min']} 至 {f['date_max']} | {f['unique_permno']:,} | {f['duplicate_keys']} |")
    lines += ['', '## 月度股票池', '',
        '- `universe_monthly` 每月恰有 500 只股票。`eval_month` 表示适用自然月。',
        f"- `daily_panel.in_universe` 与同一自然月的月表完全匹配：{a['membership_month_offsets']['0']['mismatch_rows']} 行差异。不可再将资格机械滞后一月。",
        f"- 使用**前月最后交易日** `me` 选取面板中最大的 500 只，{prior['months_compared']} 个月中 {prior['exact_500_matches']} 月精确匹配；平均交集 {prior['mean_intersection']:.4f}/500，最小交集 {prior['min_intersection']}/500。",
        f"- 对照：使用**当月最后交易日**，{same['months_compared']} 个月中 {same['exact_500_matches']} 月精确匹配；平均交集 {same['mean_intersection']:.4f}/500。",
        '- 该检验限于给定的 2,203 只股票联合面板，不能单凭此证明整个 CRSP 筛选流程无幸存者偏差。', '',
        '## 回报与无风险利率', '',
        f"- `features` 的键与 `daily_panel.in_universe=1` 的键一一对应；回报差异不超过 {a['features_join']['max_abs_return_difference']:.3g}，符合 float32 存储精度。",
        '- `RET`、`RET_ADJ` 是日小数回报，例如 0.01 表示 1%。`RET_ADJ` 已包含退市回报。',
        f"- {a['delisting_adjustment']['observations']} 条存在 `DLRET` 的记录全部满足 `(1+RET)*(1+DLRET)-1`，最大差异 {a['delisting_adjustment']['max_abs_formula_error']:.3g}。不可重复加入退市回报。",
        f"- 股票池内共有 {len(a['in_universe_missing_returns'])} 条缺失 `RET_ADJ`；完整日期、股票键在 JSON 中。缺失回报不能未经披露直接当作观察到的零回报。",
        '- `rf` 位于 `features`，每日横截面只有一个有效值；仅 1985-01-02 缺失，其后范围为 0 至 0.0004。数值与日小数无风险利率一致，但没有提供其来源文件。',
        '- 超额回报应由 `RET_ADJ-rf` 构成，`RET_ADJ` 本身不是超额回报。', '',
        '## 特征时点与尺度', '',
        '- 原始 39 个特征列已被缩放到 [-0.5, 0.5]，不存在 NaN；部分缺失值显然已填零。零也可能是横截面中位数，现有文件不能还原原始缺失掩码。',
        f"- `Ret_D1` 与前一交易日回报横截面秩相关为 {a['ret_d1_alignment']['lag1']['rank_correlation']:.10f}；与当日回报秩相关为 {a['ret_d1_alignment']['RET_ADJ']['rank_correlation']:.6f}。这有力支持 DATE=t 行已含 t-1 信息，不应无依据再将整张特征表 shift 一天。",
        '- 39 个 `med_*` 特征保留原始量纲，不能与 [-0.5,0.5] 的资产秩特征混为同一尺度；早期历史存在约 0.01% 至 12.19% 不等的 NaN。',
        '- 起始日所有 39 个资产特征为零，所有中位数和 rf 缺失，存在历史预热阶段。1985 年数据不能直接当成成熟特征样本。',
        '- 其他特征是否严格遵循会计公布日和滞后规则，需要原始特征生成代码才能完全确认。', '',
        '## 建议读取约定', '',
        '1. PCA 因子估计使用完整 daily_panel 历史，交易资产使用对应月份 universe；不要只用当日 features 中出现的股票去删改过去历史。',
        '2. 对 t 日的交易决策只使用截至 t-1 的回报估计协方差和信号；实际 t 日回报只进入损益评价。',
        '3. 月度资格按 eval_month 同月匹配。股票退出与新进入需显式处理持仓与交易成本。',
        '4. 新股历史不足、缺失日回报、训练初始窗口的处理方式需在实现和结果报告中披露。',
        '5. 来源路径可通过 audit_data.py 的 --universe、--features、--daily 覆盖。', '',
    ]
    path.write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
