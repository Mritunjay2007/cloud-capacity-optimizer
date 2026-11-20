from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

# Allow running both as `python optimizer/run_phase2_baseline.py`
# and from inside `optimizer/` as `python run_phase2_baseline.py`
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from baseline_ilp import BaselineCapacityOptimizer, InstancePricing


def load_shaved_ice_csv(
    csv_path: Path,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Load and normalize the Phase 1 Snowflake Shaved Ice CSV."""
    df = pd.read_csv(csv_path)

    rename_map = {
        "USAGE_HOUR": "Timestamp",
        "REGION_NUM": "Region",
        "INSTANCE_TYPE": "VM Type",
        "NORM_USAGE": "Count",
    }
    missing = set(rename_map.keys()) - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing expected columns: {missing}")

    df = df.rename(columns=rename_map)[["Timestamp", "Region", "VM Type", "Count"]]
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["Region"] = df["Region"].astype(int)
    df["VM Type"] = df["VM Type"].astype(str).str.strip()
    df["Count"] = df["Count"].round().astype(int)

    if start:
        df = df[df["Timestamp"] >= pd.to_datetime(start)]
    if end:
        df = df[df["Timestamp"] < pd.to_datetime(end)]

    if df.empty:
        raise ValueError("No rows left after applying date filter.")

    # Aggregate duplicates if any exist
    df = (
        df.groupby(["Timestamp", "Region", "VM Type"], as_index=False)["Count"]
        .sum()
        .sort_values(["Region", "VM Type", "Timestamp"])
        .reset_index(drop=True)
    )
    return df


def pad_missing_hours(region_df: pd.DataFrame, full_hours: pd.DatetimeIndex) -> pd.DataFrame:
    """Ensure completely idle hours still appear in the horizon for accurate reserved billing."""
    present = set(region_df["Timestamp"].unique())
    missing_ts = [ts for ts in full_hours if ts not in present]
    if not missing_ts:
        return region_df

    region_id = int(region_df["Region"].iloc[0])
    anchor_vm = str(region_df["VM Type"].iloc[0])
    pad_df = pd.DataFrame(
        {
            "Timestamp": missing_ts,
            "Region": region_id,
            "VM Type": anchor_vm,
            "Count": 0,
        }
    )
    return pd.concat([region_df, pad_df], ignore_index=True)


def build_pricing(
    df: pd.DataFrame,
    discount: float,
    pricing_json: Optional[Path] = None,
) -> Dict[Tuple[int, str], InstancePricing]:
    """Build (Region, VM Type) pricing table using default unit rates or a JSON override."""
    custom_od: Dict[str, float] = {}
    if pricing_json is not None:
        with open(pricing_json, "r", encoding="utf-8") as f:
            custom_od = json.load(f)

    pricing: Dict[Tuple[int, str], InstancePricing] = {}
    pairs = sorted({(int(r), str(v)) for r, v in zip(df["Region"], df["VM Type"])})
    for r, v in pairs:
        od_rate = float(custom_od.get(v, 1.0))
        res_rate = round(od_rate * (1.0 - discount), 6)
        pricing[(r, v)] = InstancePricing(
            region=r,
            vm_type=v,
            on_demand_rate=od_rate,
            reserved_rate=res_rate,
        )
    return pricing


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 2 Baseline ILP Cost-Minimization on Shaved Ice data."
    )
    parser.add_argument("--data", type=Path, required=True, help="Path to Phase 1 CSV file.")
    parser.add_argument("--start", type=str, default=None, help="Inclusive start date (e.g. 2021-02-01).")
    parser.add_argument("--end", type=str, default=None, help="Exclusive end date (e.g. 2021-03-01).")
    parser.add_argument("--discount", type=float, default=0.40, help="Commitment discount fraction (default: 0.40).")
    parser.add_argument("--pricing-json", type=Path, default=None, help="Optional JSON mapping INSTANCE_TYPE -> OD rate.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase2"), help="Output directory for CSV artifacts.")
    parser.add_argument("--solver-msg", action="store_true", help="Show raw CBC solver logs.")
    args = parser.parse_args()

    print(f"Loading dataset: {args.data}")
    df = load_shaved_ice_csv(args.data, start=args.start, end=args.end)
    min_ts, max_ts = df["Timestamp"].min(), df["Timestamp"].max()
    full_hours = pd.date_range(min_ts, max_ts, freq="h")
    print(
        f"Loaded {len(df):,} rows | Window: {min_ts} -> {max_ts} ({len(full_hours):,} hrs) "
        f"| Regions: {sorted(df['Region'].unique().tolist())} "
        f"| Instance Types: {sorted(df['VM Type'].unique().tolist())}\n"
    )

    pricing = build_pricing(df, discount=args.discount, pricing_json=args.pricing_json)
    optimizer = BaselineCapacityOptimizer(pricing=pricing)

    region_summaries = []
    allocation_rows = []
    dispatch_frames = []

    for region in sorted(df["Region"].unique()):
        t0 = time.perf_counter()
        region_df = pad_missing_hours(df[df["Region"] == region].copy(), full_hours)
        res = optimizer.solve(region_df, solver_msg=args.solver_msg)
        elapsed = time.perf_counter() - t0

        region_summaries.append(
            {
                "Region": int(region),
                "Hours": len(full_hours),
                "Status": res.status,
                "PureOnDemandCost": round(res.pure_on_demand_cost, 2),
                "OptimizedTotalCost": round(res.total_cost, 2),
                "ReservedCost": round(res.reserved_cost, 2),
                "OnDemandCost": round(res.on_demand_cost, 2),
                "IdleCommitmentCost": round(res.idle_commitment_cost, 2),
                "SavingsUSD": round(res.savings_usd, 2),
                "SavingsPct": round(res.savings_pct, 2),
                "ReservedUtilizationPct": round(res.reserved_utilization_pct, 2),
                "SolveTimeSec": round(elapsed, 3),
            }
        )

        for (r, v), r_count in sorted(res.reserved_allocation.items()):
            p = pricing[(r, v)]
            sub = res.hourly_dispatch[
                (res.hourly_dispatch["Region"] == r) & (res.hourly_dispatch["VM Type"] == v)
            ]
            allocation_rows.append(
                {
                    "Region": r,
                    "VM Type": v,
                    "OnDemandRate": p.on_demand_rate,
                    "ReservedRate": p.reserved_rate,
                    "OptimalReserved": int(round(r_count)),
                    "MeanDemand": round(float(sub["Demand"].mean()), 2),
                    "PeakDemand": int(sub["Demand"].max()),
                    "MeanOnDemand": round(float(sub["OnDemand"].mean()), 2),
                    "MeanIdleReserved": round(float(sub["IdleReserved"].mean()), 2),
                }
            )

        dispatch_frames.append(res.hourly_dispatch)
        print(
            f"[Region {region}] Optimal in {elapsed:.2f}s | "
            f"Pure OD: ${res.pure_on_demand_cost:,.2f} -> Opt: ${res.total_cost:,.2f} "
            f"(Savings: {res.savings_pct:.2f}%, Util: {res.reserved_utilization_pct:.2f}%)"
        )

    # Export Phase 2 baseline artifacts for Phases 3–5
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(region_summaries)
    alloc_df = pd.DataFrame(allocation_rows)
    dispatch_df = pd.concat(dispatch_frames, ignore_index=True)

    summary_path = args.output_dir / "phase2_region_summary.csv"
    alloc_path = args.output_dir / "phase2_reserved_allocations.csv"
    dispatch_path = args.output_dir / "phase2_hourly_dispatch.csv"

    summary_df.to_csv(summary_path, index=False)
    alloc_df.to_csv(alloc_path, index=False)
    dispatch_df.to_csv(dispatch_path, index=False)

    total_od = summary_df["PureOnDemandCost"].sum()
    total_opt = summary_df["OptimizedTotalCost"].sum()
    total_sav = total_od - total_opt
    total_sav_pct = (total_sav / total_od * 100.0) if total_od > 0 else 0.0

    print("\n=== Global Phase 2 Baseline Summary ===")
    print(f"Total Pure On-Demand Cost: ${total_od:,.2f}")
    print(f"Total Optimized Cost:      ${total_opt:,.2f}")
    print(f"Total Savings:             ${total_sav:,.2f} ({total_sav_pct:.2f}%)")
    print(f"\nSaved artifacts to: {args.output_dir.resolve()}")
    print(f"  - {summary_path.name}")
    print(f"  - {alloc_path.name}")
    print(f"  - {dispatch_path.name}")


if __name__ == "__main__":
    main()