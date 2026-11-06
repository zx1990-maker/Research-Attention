"""Run the paper's two non-attention benchmarks on the supplied data.

Examples:
    python run.py prepare
    python run.py ou --ks 1 3 5 8 10 15 30 100
    python run.py longconv --ks 30
    python run.py collect
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from models import LongConvPolicy
from pipeline import (ROOT, account, dump_json, load_cache, make_inputs, normalize_weights,
                      performance, prepare, read_config, summarize, load_sealed_cache)


def ou_scores(history, valid, entry=1.25, r2_min=0.25):
    """DLSA-style daily OU threshold; no position hysteresis or half-life cutoff."""
    cumulative = np.cumsum(history.astype(np.float64), axis=-1)
    x, y = cumulative[..., :-1], cumulative[..., 1:]
    xc, yc = x-x.mean(-1, keepdims=True), y-y.mean(-1, keepdims=True)
    xx = np.sum(xc*xc, axis=-1); yy = np.sum(yc*yc, axis=-1)
    xy = np.sum(xc*yc, axis=-1)
    b = np.divide(xy, xx, out=np.zeros_like(xx), where=xx>1e-18)
    a = y.mean(-1)-b*x.mean(-1)
    errors = y - (a[..., None]+b[..., None]*x)
    error_var = np.mean(errors*errors, axis=-1)
    r2 = np.divide(xy*xy, xx*yy, out=np.zeros_like(xx), where=xx*yy>1e-30)
    stationary = valid & (b>0) & (b<1) & (error_var>1e-18) & (r2>r2_min)
    safe_b = np.where(stationary, b, 0.0)
    mean = a/(1-safe_b); equilibrium_sd = np.sqrt(error_var/np.maximum(1-safe_b**2, 1e-12))
    score = np.divide(cumulative[..., -1]-mean, equilibrium_sd,
                      out=np.zeros_like(xx), where=equilibrium_sd>1e-12)
    return np.where(stationary & (score>entry), -1.0,
                    np.where(stationary & (score < -entry), 1.0, 0.0)).astype(np.float32)


def numpy_weights(q, indices, data, k):
    v = np.asarray(data["basis"][indices, :, :k]); s = np.asarray(data["scales"][indices])
    w = q - np.einsum("bnk,bk->bn", v, np.einsum("bnk,bn->bk", v, s*q))/s
    w = normalize_weights(w)
    all_weights = np.zeros((len(indices), len(data["permnos"])), dtype=np.float32)
    ids = data["ids"][indices]; rows, slots = np.nonzero(ids>=0)
    all_weights[rows, ids[rows, slots]] = w[rows, slots]
    return all_weights


def save_run(frame, weights, data, indices, cfg, model, k, out, details):
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "daily.csv", index=False)
    np.savez_compressed(out / "weights.npz", dates=data["dates"][indices],
                        permnos=data["permnos"], weights=weights)
    summary = summarize(frame, model, k)
    dump_json(out / "summary.json", summary)
    yearly = [summarize(g, model, k) | {"year": int(year)}
              for year, g in frame.groupby(frame.date.dt.year)]
    pd.DataFrame(yearly).to_csv(out / "yearly.csv", index=False)
    dump_json(out / "run_manifest.json", {"config": cfg, "model": model, "K": k,
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "torch": torch.__version__, "cache": json.loads((data["path"] / "complete.json").read_text()),
        **details})
    print(json.dumps(summary, indent=2), flush=True)


def test_indices(data, cfg):
    years = data["calendar"].year
    return np.flatnonzero((years>=cfg["test_start_year"]) & (years<=cfg["test_end_year"]))


def run_ou(data, cfg, k, out):
    x, valid = make_inputs(data, k, cfg["lookback"])
    ix = test_indices(data, cfg); weights = np.zeros((len(ix), len(data["permnos"])), dtype=np.float32)
    for start in range(0, len(ix), 252):
        batch = ix[start:start+252]
        q = ou_scores(x[batch], valid[batch], cfg["ou_entry_threshold"], cfg["ou_r2_threshold"])
        weights[start:start+len(batch)] = numpy_weights(q, batch, data, k)
    frame = account(weights, data, ix, cfg)
    # This diagnostic is not used to pick the best test-period model or change thresholds.
    valix = np.flatnonzero((data["calendar"].year>=1996) & (data["calendar"].year<=1997))
    validation = []
    for entry in [1., 1.25, 1.5]:
        for r2_min in [.25, .5, .75]:
            parts = []
            for start in range(0, len(valix), 126):
                batch = valix[start:start+126]
                parts.append(numpy_weights(ou_scores(x[batch], valid[batch], entry, r2_min), batch, data, k))
            vf = account(np.concatenate(parts), data, valix, cfg)
            validation.append({"entry": entry, "r2_threshold": r2_min,
                               **performance(vf.net_excess, "net_")})
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(validation).to_csv(out/"validation_thresholds.csv", index=False)
    save_run(frame, weights, data, ix, cfg, "PCA+OU", k, out,
             {"threshold_selection": "Fixed to DLSA published 1.25/0.25; validation grid is diagnostic only"})


def as_tensor(a, dtype=None):
    # Materialize memmap slices, avoiding read-only NumPy tensor warnings.
    return torch.as_tensor(np.array(a, copy=True), dtype=dtype)


def tensor_batch(data, x, valid, indices, k, input_scale=1.0):
    ids = as_tensor(data["ids"][indices], torch.long)
    return {
        "x": as_tensor(x[indices]) * input_scale, "valid": as_tensor(valid[indices]), "ids": ids,
        "v": as_tensor(data["basis"][indices, :, :k]), "s": as_tensor(data["scales"][indices]),
        "r": as_tensor(data["returns"][indices]),
    }


def torch_weights(model, batch, nassets):
    q = model(batch["x"]) * batch["valid"]
    v, s = batch["v"], batch["s"]
    w = q - torch.bmm(v, torch.bmm(v.transpose(1, 2), (s*q).unsqueeze(-1))).squeeze(-1)/s
    denom = w.abs().sum(-1, keepdim=True)
    w = w / denom.clamp_min(1e-12)
    active = batch["ids"]>=0
    weights = torch.zeros((len(w), nassets), dtype=w.dtype)
    return weights.scatter_add(1, batch["ids"].clamp_min(0), w*active)


def train_one(data, x, valid, cfg, k, train_indices, seed, label, checkpoint_dir):
    torch.manual_seed(seed); np.random.seed(seed % (2**32))
    model = LongConvPolicy(lookback=cfg["lookback"], hidden_dim=cfg["hidden_dim"],
                           dropout=cfg["dropout"], squash=cfg["squash"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    batches = []
    # Cache a training year's small tensor batches once, not on every epoch.
    # The prior date overlaps to charge turnover at minibatch boundaries.
    for start in range(0, len(train_indices), cfg["batch_days"]):
        begin = max(0, start-1); batch_ix = train_indices[begin:start+cfg["batch_days"]]
        batches.append((tensor_batch(data, x, valid, batch_ix, k, cfg.get("longconv_input_scale", 1.0)), 1 if start else 0))
    log = []; ts = time.perf_counter(); nassets = len(data["permnos"])
    for epoch in range(cfg["epochs"]):
        model.train(); stats = []
        for batch, overlap in batches:
            optimizer.zero_grad(set_to_none=True)
            w = torch_weights(model, batch, nassets)
            gross = (w*batch["r"]).sum(-1)
            prev = torch.cat([torch.zeros_like(w[:1]), w[:-1]], dim=0)
            turnover = (w-prev).abs().sum(-1)
            short = (-w).clamp_min(0).sum(-1)
            net = (gross-cfg["transaction_cost"]*turnover-cfg["short_cost"]*short)[overlap:]
            sharpe = net.mean()/net.std(unbiased=False).clamp_min(1e-6)
            loss = -sharpe
            if not torch.isfinite(loss): raise FloatingPointError(f"Nonfinite loss {label} epoch={epoch}")
            loss.backward(); optimizer.step()
            stats.append(float(sharpe.detach()))
        row = {"fit": label, "epoch": epoch+1, "mean_batch_daily_net_sharpe": np.mean(stats),
               "elapsed_seconds": time.perf_counter()-ts, "seed": seed}
        log.append(row)
        if epoch in [0, 9, 19, cfg["epochs"]-1]:
            print(f"LongConv K={k} fit={label} epoch={epoch+1}/{cfg['epochs']} "
                  f"batch_SR={row['mean_batch_daily_net_sharpe']:.4f} time={row['elapsed_seconds']:.1f}s", flush=True)
    model.eval()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": cfg, "k": k, "seed": seed,
                "training_start": str(data["calendar"][train_indices[0]].date()),
                "training_end": str(data["calendar"][train_indices[-1]].date())}, checkpoint_dir/f"{label}.pt")
    return model, log


def infer(model, data, x, valid, indices, k, batch_days=252, input_scale=1.0):
    result = np.zeros((len(indices), len(data["permnos"])), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_days):
            batch_ix = indices[start:start+batch_days]
            batch = tensor_batch(data, x, valid, batch_ix, k, input_scale)
            result[start:start+len(batch_ix)] = torch_weights(model, batch, len(data["permnos"])).numpy()
    return result


def run_longconv(data, cfg, k, out):
    out.mkdir(parents=True, exist_ok=True)
    state_path = out/"progress.json"
    identity = {"config": cfg, "cache_fingerprint": json.loads((data["path"]/"complete.json").read_text())["fingerprint"], "K": k}
    if state_path.exists():
        if json.loads(state_path.read_text()) != identity:
            raise RuntimeError(f"Cannot resume {out}: configuration or data fingerprint changed")
    else:
        dump_json(state_path, identity)
    x, valid = make_inputs(data, k, cfg["lookback"]); years = data["calendar"].year
    ix = test_indices(data, cfg); allweights = np.zeros((len(ix), len(data["permnos"])), dtype=np.float32)
    log_path = out/"training_log.csv"
    logs = pd.read_csv(log_path).to_dict("records") if log_path.exists() else []
    start_time = time.perf_counter()
    # Published hyperparameters are fixed in advance. This pre-OOS diagnostic
    # uses 1990-1995 for fitting and 1996-1997 for validation; no selection on OOS.
    train_ix = np.flatnonzero((years>=1990) & (years<=1995))
    val_ix = np.flatnonzero((years>=1996) & (years<=1997))
    if not (out/"validation_summary.json").exists():
        model, log = train_one(data, x, valid, cfg, k, train_ix, cfg["seed"]+k*1000,
                               "validation_1996_1997", out/"checkpoints")
        logs.extend(log)
        val_weights = infer(model, data, x, valid, val_ix, k, cfg["batch_days"], cfg.get("longconv_input_scale", 1.0))
        valframe = account(val_weights, data, val_ix, cfg)
        valframe.to_csv(out/"validation_daily.csv", index=False)
        pd.DataFrame(logs).to_csv(log_path, index=False)
        dump_json(out/"validation_summary.json", summarize(valframe, "PCA+LongConv", k))
    for year in range(cfg["test_start_year"], cfg["test_end_year"]+1):
        train_ix = np.flatnonzero((years>=year-cfg["training_years"]) & (years<year))
        test_ix = np.flatnonzero(years==year)
        if not len(train_ix) or not len(test_ix): raise ValueError(f"Missing train/test dates for {year}")
        if train_ix[-1] >= test_ix[0]: raise AssertionError("Training overlaps evaluation")
        saved = out/f"weights_{year}.npy"
        rows = np.searchsorted(ix, test_ix)
        if saved.exists():
            restored = np.load(saved)
            if restored.shape != (len(test_ix), len(data["permnos"])):
                raise ValueError(f"Invalid completed year weights: {saved}")
            allweights[rows] = restored
            print(f"Resumed K={k} OOS={year}", flush=True)
            continue
        model, log = train_one(data, x, valid, cfg, k, train_ix,
                               cfg["seed"]+k*1000+year, str(year), out/"checkpoints")
        logs.extend(log)
        w = infer(model, data, x, valid, test_ix, k, cfg["batch_days"], cfg.get("longconv_input_scale", 1.0))
        allweights[rows] = w
        pd.DataFrame(logs).to_csv(log_path, index=False)
        tmp = out/f"weights_{year}.partial.npy"
        np.save(tmp, w); os.replace(tmp, saved)
        print(f"Completed K={k} OOS={year}; total {time.perf_counter()-start_time:.1f}s", flush=True)
    # Evaluate whole OOS series in one pass: do not reset costs on January 1.
    frame = account(allweights, data, ix, cfg)
    save_run(frame, allweights, data, ix, cfg, "PCA+LongConv", k, out,
             {"elapsed_seconds": time.perf_counter()-start_time,
              "hyperparameters": "Published Table 4, fixed before OOS. Validation is diagnostic, no search.",
              "optimizer": "torch.optim.Adam; coupled weight decay; daily Sharpe minibatch objective",
              "batch_days": cfg["batch_days"], "independent_initialization_each_year": True})


def collect(results):
    rows = []
    for path in sorted(results.glob("*/summary.json")):
        rows.append(json.loads(path.read_text()))
    if rows:
        pd.DataFrame(rows).sort_values(["model", "K"]).to_csv(results/"metrics.csv", index=False)
    print(f"Collected {len(rows)} completed model/K runs", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prepare", "ou", "longconv", "all", "collect"])
    p.add_argument("--config", type=Path, default=ROOT/"config.json")
    p.add_argument("--cache", type=Path, default=ROOT/"cache")
    p.add_argument("--results", type=Path, default=ROOT/"results")
    p.add_argument("--ks", nargs="+", type=int)
    p.add_argument("--use-cache", action="store_true", help="Continue from SHA-256 sealed PCA arrays when source parquet files are unavailable")
    args = p.parse_args(); cfg = read_config(args.config); args.results.mkdir(parents=True, exist_ok=True)
    if args.command=="collect": collect(args.results); return
    torch.set_num_threads(cfg["threads"])
    with threadpool_limits(cfg["threads"]):
        # prepare also validates the data/config fingerprint before reusing a cache.
        data = load_sealed_cache(cfg, args.cache) if args.use_cache else prepare(cfg, args.cache)
        if args.command=="prepare": return
        for k in args.ks or cfg["factor_counts"]:
            if k not in cfg["factor_counts"]: raise ValueError(f"K={k} not in prepared cache")
            if args.command in ["ou", "all"]: run_ou(data, cfg, k, args.results/f"ou_k{k}")
            if args.command in ["longconv", "all"]: run_longconv(data, cfg, k, args.results/f"longconv_k{k}")
        collect(args.results)


if __name__ == "__main__": main()
