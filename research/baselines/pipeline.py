"""Strictly lagged PCA data and portfolio-accounting utilities.

All dates label the return being earned. PCA for date t uses [t-252,t),
and the signal for t uses residual returns [t-30,t). The tradable universe
is the supplied eval_month membership, subject only to lagged data checks.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent


def read_config(path=None):
    return json.loads(Path(path or ROOT / "config.json").read_text())


def dump_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def fingerprint(cfg):
    relevant = {k: cfg[k] for k in ["cache_start", "pca_window", "lookback", "factor_counts", "pca_mode"]}
    for k in ["daily_panel", "features", "universe_monthly"]:
        p = Path(cfg[k]); st = p.stat()
        relevant[k] = [str(p.resolve()), st.st_size, st.st_mtime_ns]
    relevant["cache_version"] = 2
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()


def cache_definition(cfg):
    """Configuration that defines PCA data, excluding neural-network settings."""
    keys = ["cache_start", "pca_window", "lookback", "factor_counts", "pca_mode",
            "daily_panel", "features", "universe_monthly"]
    return {key: cfg[key] for key in keys}


def seal_cache(cache, cfg, source_hashes=None):
    """Hash completed base arrays so a deleted temporary source cannot mask damage."""
    cache=Path(cache)
    complete=json.loads((cache/"complete.json").read_text())
    names=["dates.npy","permnos.npy","returns.npy","returns_missing.npy","rf.npy",
           "ids.npy","basis.npy","scales.npy","market_proxy.npy"]
    names += [f"residuals_k{k}.npy" for k in cfg["factor_counts"]]
    hashes={}
    for name in names:
        digest=hashlib.sha256()
        with (cache/name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(8*1024*1024), b""):
                digest.update(chunk)
        hashes[name]=digest.hexdigest()
    dump_json(cache/"seal.json", {"fingerprint":complete["fingerprint"],
        "definition":cache_definition(cfg), "source_sha256":source_hashes or {}, "sha256":hashes})


def load_sealed_cache(cfg, cache):
    """Explicit offline continuation from PCA arrays verified against a seal."""
    cache=Path(cache)
    complete=json.loads((cache/"complete.json").read_text())
    seal=json.loads((cache/"seal.json").read_text())
    if seal["fingerprint"] != complete["fingerprint"] or seal["definition"] != cache_definition(cfg):
        raise RuntimeError("Sealed cache does not match the current PCA configuration")
    for name, expected in seal["sha256"].items():
        digest=hashlib.sha256()
        with (cache/name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(8*1024*1024), b""):
                digest.update(chunk)
        if digest.hexdigest()!=expected:
            raise RuntimeError(f"Sealed PCA cache changed: {name}")
    return load_cache(cache)


def pca_basis(history, k, mode="covariance"):
    """Centered PCA using time-by-time Gram matrix, no current-day data.

    In correlation mode P = I - D V V' D^-1; covariance mode D=I.
    Returns eigenvectors ordered from largest eigenvalue, and scales.
    """
    h = np.asarray(history, dtype=np.float64)
    z = h - h.mean(axis=0, keepdims=True)
    scales = np.ones(z.shape[1])
    if mode == "correlation":
        scales = z.std(axis=0, ddof=1)
        z = z / scales
    elif mode != "covariance":
        raise ValueError(mode)
    rank_bound = min(z.shape[0] - 1, z.shape[1])
    if k >= rank_bound:
        raise ValueError("Factor count must be smaller than historical rank bound")
    val, left = eigh(z @ z.T, subset_by_index=[z.shape[0] - k, z.shape[0] - 1],
                     check_finite=False, driver="evr")
    order = np.arange(k - 1, -1, -1)
    val, left = val[order], left[:, order]
    if val[-1] <= 1e-12:
        raise ValueError("PCA factors are rank deficient")
    basis = (z.T @ left) / np.sqrt(val)
    return basis.astype(np.float32), scales.astype(np.float32)


def residualize(r, basis, scales):
    return r - scales * (basis @ (basis.T @ (r / scales)))


def map_scores(q, basis, scales):
    """P' q (not P q); matters when correlation PCA is selected."""
    return q - (basis @ (basis.T @ (scales * q))) / scales


def normalize_weights(w):
    denom = np.abs(w).sum(axis=-1, keepdims=True)
    return np.divide(w, denom, out=np.zeros_like(w), where=denom > 1e-12)


def prepare(cfg, cache=None):
    cache = Path(cache or ROOT / "cache"); cache.mkdir(parents=True, exist_ok=True)
    signature = fingerprint(cfg)
    if (cache / "complete.json").exists():
        old = json.loads((cache / "complete.json").read_text())
        if old["fingerprint"] == signature:
            print("Using completed PCA cache", flush=True)
            return load_cache(cache)
        raise RuntimeError("Cache/config mismatch. Choose another --cache directory or remove the old cache.")

    print("Loading supplied daily returns and monthly membership", flush=True)
    raw = pd.read_parquet(cfg["daily_panel"], columns=["DATE", "PERMNO", "RET_ADJ", "me", "DLRET"])
    if raw.duplicated(["DATE", "PERMNO"]).any():
        raise ValueError("Duplicate date/PERMNO in daily_panel")
    wide = raw.pivot(index="DATE", columns="PERMNO", values="RET_ADJ").sort_index()
    cap = raw.pivot(index="DATE", columns="PERMNO", values="me").reindex(index=wide.index, columns=wide.columns).to_numpy()
    dates_all = wide.index; permnos = wide.columns.to_numpy()
    # DLRET is part of the delisting-date return. That event is public only
    # after date t; the stock stops being tradable from t+1 even if its old
    # monthly top-500 label persists until month end.
    delisting_dates = (raw.loc[raw.DLRET.notna()].groupby("PERMNO")["DATE"].min()
                       .reindex(permnos).to_numpy())
    r_all = wide.to_numpy(dtype=np.float64)
    feat = pd.read_parquet(cfg["features"], columns=["DATE", "rf"])
    if feat.groupby("DATE")["rf"].nunique().max() != 1:
        raise ValueError("Risk free rate is not constant within a date")
    rf_all = feat.groupby("DATE")["rf"].first().reindex(dates_all).to_numpy(dtype=np.float64)
    start = dates_all.searchsorted(pd.Timestamp(cfg["cache_start"]))
    if start < cfg["pca_window"] or not np.isfinite(rf_all[start-cfg["pca_window"]:]).all():
        raise ValueError("Insufficient risk-free/return history")
    r_all -= rf_all[:, None]
    universe = pd.read_parquet(cfg["universe_monthly"])
    universe["month"] = universe.eval_month.dt.to_period("M").astype(str)
    by_month = {m: np.searchsorted(permnos, g.PERMNO.to_numpy()) for m, g in universe.groupby("month")}
    if not np.isin(universe.PERMNO, permnos).all():
        raise ValueError("Universe contains identifiers absent from daily_panel")
    counts = universe.groupby("month").size()
    slots = int(counts.max()); dates = dates_all[start:]; nt, ns = len(dates), len(permnos)
    kmax = max(cfg["factor_counts"])
    np.save(cache / "dates.npy", dates.to_numpy()); np.save(cache / "permnos.npy", permnos)
    np.save(cache / "rf.npy", rf_all[start:].astype(np.float32))
    # Retain missing flags; filling missing realized returns is an explicit valuation convention.
    np.save(cache / "returns_missing.npy", ~np.isfinite(r_all[start:]))
    np.save(cache / "returns.npy", np.nan_to_num(r_all[start:], nan=0.0).astype(np.float32))
    ids = np.lib.format.open_memmap(cache / "ids.npy", mode="w+", dtype=np.int32, shape=(nt, slots)); ids[:] = -1
    basis = np.lib.format.open_memmap(cache / "basis.npy", mode="w+", dtype=np.float32, shape=(nt, slots, kmax)); basis[:] = 0
    scales = np.lib.format.open_memmap(cache / "scales.npy", mode="w+", dtype=np.float32, shape=(nt, slots)); scales[:] = 1
    resids = {k: np.lib.format.open_memmap(cache / f"residuals_k{k}.npy", mode="w+", dtype=np.float32, shape=(nt, ns)) for k in cfg["factor_counts"]}
    for a in resids.values(): a[:] = np.nan
    market = np.zeros(nt); neligible = np.zeros(nt, dtype=int)
    missing_realized = 0; ts = time.perf_counter()
    with threadpool_limits(cfg["threads"]):
        for t, date in enumerate(dates):
            g = start + t; candidates = np.sort(by_month[str(date.to_period("M"))])
            still_listed = pd.isna(delisting_dates[candidates]) | (delisting_dates[candidates] >= np.datetime64(date))
            candidates = candidates[still_listed]
            history = r_all[g-cfg["pca_window"]:g, candidates]
            good = np.isfinite(history).all(axis=0) & (np.nanstd(history, axis=0) > 1e-8)
            selected = candidates[good]; h = history[:, good]
            v, scale = pca_basis(h, kmax, cfg["pca_mode"])
            n = len(selected); ids[t, :n] = selected; basis[t, :n] = v; scales[t, :n] = scale
            neligible[t] = n
            realized = r_all[g, selected]
            observed = np.isfinite(realized); missing_realized += int((~observed).sum())
            # Unknown realized data is valued at zero excess return, never used to select holdings.
            realized = np.where(observed, realized, 0.0)
            loading_return = v.T @ (realized / scale)
            for k, arr in resids.items():
                eps = realized - scale * (v[:, :k] @ loading_return[:k])
                # A missing observed return invalidates the subsequent residual signal history.
                eps[~observed] = np.nan
                arr[t, selected] = eps
            cw = np.nan_to_num(cap[g-1, candidates], nan=0.0); cw = np.maximum(cw, 0)
            if cw.sum(): market[t] = np.dot(cw / cw.sum(), np.nan_to_num(r_all[g, candidates], nan=0.0))
            if t % 252 == 0 or t == nt-1:
                print(f"PCA {date.date()}: {t+1}/{nt}, eligible={n}, elapsed={time.perf_counter()-ts:.1f}s", flush=True)
    np.save(cache / "market_proxy.npy", market.astype(np.float32))
    pd.DataFrame({"date": dates, "eligible_assets": neligible}).to_csv(cache / "eligibility.csv", index=False)
    for a in [ids, basis, scales, *resids.values()]: a.flush()
    dump_json(cache / "complete.json", {"fingerprint": signature, "dates": nt, "assets_union": ns,
              "slots": slots, "missing_realized_eligible_asset_days": missing_realized,
              "eligible_min": int(neligible.min()), "eligible_mean": float(neligible.mean()),
              "pca_mode": cfg["pca_mode"], "elapsed_seconds": time.perf_counter()-ts})
    seal_cache(cache, cfg)
    return load_cache(cache)


def load_cache(cache=None):
    path = Path(cache or ROOT / "cache")
    data = {name: np.load(path / f"{name}.npy", mmap_mode="r") for name in
            ["dates", "permnos", "returns", "returns_missing", "rf", "ids", "basis", "scales", "market_proxy"]}
    data["path"] = path; data["calendar"] = pd.DatetimeIndex(data["dates"])
    return data


def make_inputs(data, k, lookback=30):
    """Histories follow permanent identifiers, not changing cross-sectional slots."""
    path = data["path"] / f"inputs_k{k}_l{lookback}.npy"
    valid_path = data["path"] / f"input_valid_k{k}_l{lookback}.npy"
    marker = data["path"] / f"input_complete_k{k}_l{lookback}.json"
    parent = json.loads((data["path"] / "complete.json").read_text())["fingerprint"]
    if path.exists() and valid_path.exists() and marker.exists() and json.loads(marker.read_text()).get("fingerprint") == parent:
        return np.load(path, mmap_mode="r"), np.load(valid_path, mmap_mode="r")
    marker.unlink(missing_ok=True)
    eps = np.load(data["path"] / f"residuals_k{k}.npy", mmap_mode="r")
    ids = data["ids"]; nt, slots = ids.shape
    x = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(nt, slots, lookback)); x[:] = 0
    valid = np.lib.format.open_memmap(valid_path, mode="w+", dtype=np.bool_, shape=(nt, slots)); valid[:] = False
    for t in range(lookback, nt):
        active = ids[t] >= 0; current_ids = ids[t, active]
        h = np.asarray(eps[t-lookback:t, current_ids]).T
        ok = np.isfinite(h).all(axis=1)
        x[t, active] = np.where(np.isfinite(h), h, 0)
        valid[t, np.flatnonzero(active)] = ok
    x.flush(); valid.flush()
    dump_json(marker, {"fingerprint": parent, "k": k, "lookback": lookback})
    return x, valid


def performance(returns, prefix=""):
    r = np.asarray(returns, dtype=float); n = len(r)
    mu = r.mean() * 252 if n else np.nan
    vol = r.std(ddof=0) * np.sqrt(252) if n else np.nan
    wealth = np.cumprod(1 + r)
    peak = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
    return {prefix+"annual_excess_return": mu, prefix+"annual_volatility": vol,
            prefix+"sharpe": mu / vol if vol > 0 else np.nan,
            prefix+"max_drawdown": float(np.min(wealth / peak - 1)) if n else np.nan,
            prefix+"cumulative_excess_wealth": float(wealth[-1]) if n else np.nan}


def account(weights, data, indices, cfg, previous=None):
    """Weights indexed by all PERMNOs so exits, entries and year boundaries cost money."""
    prev = np.zeros(weights.shape[1], dtype=np.float32) if previous is None else previous
    turnover = np.abs(weights - np.vstack([prev, weights[:-1]])).sum(axis=1)
    short = np.maximum(-weights, 0).sum(axis=1)
    gross = np.einsum("tn,tn->t", weights, data["returns"][indices])
    tcost = cfg["transaction_cost"] * turnover; bcost = cfg["short_cost"] * short
    missing_exposure = np.einsum("tn,tn->t", np.abs(weights), data["returns_missing"][indices])
    return pd.DataFrame({"date": data["calendar"][indices], "gross_excess": gross,
        "transaction_cost": tcost, "short_cost": bcost, "net_excess": gross-tcost-bcost,
        "turnover_l1": turnover, "short_exposure": short, "gross_exposure": np.abs(weights).sum(axis=1),
        "net_exposure": weights.sum(axis=1), "missing_return_exposure": missing_exposure,
        "rf": data["rf"][indices], "market_proxy_excess": data["market_proxy"][indices]})


def summarize(frame, model, k):
    out = {"model": model, "K": int(k), "start": str(frame.date.min())[:10],
           "end": str(frame.date.max())[:10], "days": len(frame)}
    out.update(performance(frame.gross_excess, "gross_")); out.update(performance(frame.net_excess, "net_"))
    x = frame.market_proxy_excess.to_numpy(); y = frame.gross_excess.to_numpy()
    out["beta_market_proxy"] = float(np.cov(x, y, ddof=0)[0, 1] / np.var(x)) if np.var(x) else np.nan
    out["mean_daily_turnover_l1"] = float(frame.turnover_l1.mean())
    out["annual_transaction_cost"] = float(frame.transaction_cost.mean()*252)
    out["annual_short_cost"] = float(frame.short_cost.mean()*252)
    out["mean_short_exposure"] = float(frame.short_exposure.mean())
    out["active_fraction"] = float((frame.gross_exposure > 1e-6).mean())
    out["days_missing_return_exposure"] = int((frame.missing_return_exposure > 1e-10).sum())
    return out
