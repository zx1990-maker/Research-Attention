"""Recalculate every daily P&L from saved stock weights and input returns."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import ROOT, account, load_cache, read_config, summarize, dump_json


def main():
    cfg=read_config(); data=load_cache(); results=ROOT/"results"; checks=[]
    for path in sorted(results.glob("*/summary.json")):
        run=path.parent
        if not (run/"weights.npz").exists(): continue
        summary=json.loads(path.read_text()); model=summary["model"]; k=summary["K"]
        z=np.load(run/"weights.npz"); dates=pd.DatetimeIndex(z["dates"])
        ix=data["calendar"].get_indexer(dates)
        if (ix<0).any() or dates.has_duplicates or not dates.is_monotonic_increasing:
            raise AssertionError(f"Invalid dates {run}")
        target=np.flatnonzero((data["calendar"].year>=cfg["test_start_year"]) &
                             (data["calendar"].year<=cfg["test_end_year"]))
        np.testing.assert_array_equal(ix,target)
        np.testing.assert_array_equal(z["permnos"],data["permnos"])
        weights=z["weights"]
        if not np.isfinite(weights).all(): raise AssertionError(f"Nonfinite weights {run}")
        gross=np.abs(weights).sum(axis=1)
        if not np.all((np.abs(gross-1)<2e-5) | (gross<2e-5)):
            raise AssertionError(f"Weight L1 not one or zero {run}")
        recomputed=account(weights,data,ix,cfg)
        saved=pd.read_csv(run/"daily.csv",parse_dates=["date"])
        for column in ["gross_excess","net_excess","transaction_cost","short_cost","turnover_l1",
                       "short_exposure","gross_exposure","net_exposure","missing_return_exposure"]:
            np.testing.assert_allclose(recomputed[column],saved[column],atol=2e-7,rtol=2e-5,
                                       err_msg=f"{run}: {column}")
        recalculated=summarize(recomputed,model,k)
        for key in ["gross_sharpe","net_sharpe","gross_annual_excess_return","net_annual_excess_return"]:
            np.testing.assert_allclose(recalculated[key],summary[key],atol=1e-7,
                                       err_msg=f"{run}: {key}")
        checks.append({"model":model,"K":k,"days":len(ix),"max_gross_exposure":float(gross.max()),
                       "pnl_recalculated":True})
    if len(checks)!=2*len(cfg["factor_counts"]):
        raise AssertionError(f"Expected {2*len(cfg['factor_counts'])} runs; found {len(checks)}")
    dump_json(results/"verification.json",checks)
    print(f"Verified {len(checks)} runs against saved weights and daily P&L")


if __name__=="__main__": main()
