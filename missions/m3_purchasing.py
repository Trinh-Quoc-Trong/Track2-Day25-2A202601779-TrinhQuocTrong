"""M3 — Purchasing Strategy: break-even, tier choice, spot-checkpoint sim (deck §4).

Run: python missions/m3_purchasing.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import pricing

DAYS = 30


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    on_demand_monthly = optimized_monthly = 0.0
    recs = []
    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]
        gpu_hours = hpd * DAYS * ngpu
        od = num(c["on_demand_hr"])
        on_demand_cost = gpu_hours * od

        tier = pricing.recommend_tier(hpd, interruptible)
        if tier == "spot":
            sim = pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), od)
            opt_cost = sim["spot_cost"]
        elif tier == "reserved":
            opt_cost = gpu_hours * num(c["reserved_3yr_hr"])
        else:
            opt_cost = on_demand_cost

        on_demand_monthly += on_demand_cost
        optimized_monthly += opt_cost
        recs.append({"job_id": j["job_id"], "gpu_type": gtype, "tier": tier,
                     "on_demand": round(on_demand_cost), "optimized": round(opt_cost)})

    savings = on_demand_monthly - optimized_monthly
    savings_pct = savings / on_demand_monthly * 100 if on_demand_monthly else 0.0

    if verbose:
        print("== M3 Purchasing Strategy ==")
        print(f"break-even utilization @ 45% reserved discount = {pricing.break_even_utilization(0.45):.0%}")
        print(f"{'job':18}{'gpu':7}{'tier':11}{'on-demand':>12}{'optimized':>12}")
        for r in recs:
            print(f"{r['job_id']:18}{r['gpu_type']:7}{r['tier']:11}${r['on_demand']:>11,}${r['optimized']:>11,}")
        print(f"\nmonthly: on-demand ${on_demand_monthly:,.0f} -> optimized ${optimized_monthly:,.0f}  ({savings_pct:.1f}% saved)")

    return {"recommendations": recs, "on_demand_monthly": round(on_demand_monthly),
            "optimized_monthly": round(optimized_monthly), "savings_pct": round(savings_pct, 1)}


TIER_COST = {
    "spot": lambda gpu_hours, c: pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), num(c["on_demand_hr"]))["spot_cost"],
    "reserved": lambda gpu_hours, c: gpu_hours * num(c["reserved_3yr_hr"]),
    "reserved_3yr": lambda gpu_hours, c: gpu_hours * num(c["reserved_3yr_hr"]),
    "reserved_1yr": lambda gpu_hours, c: gpu_hours * num(c["reserved_1yr_hr"]),
    "on_demand": lambda gpu_hours, c: gpu_hours * num(c["on_demand_hr"]),
}

# Production inference services run indefinitely -> long enough to amortize a
# multi-year reserved term; training/dev jobs stop at their listed `days`.
INDEFINITE_HORIZON_DAYS = 365


def run_risk_adjusted_policy(verbose: bool = True) -> dict:
    """Your Turn #1 — apply the GPU-aware / duration-aware recommend_tier() and
    compare its monthly bill against the base policy used by run().
    """
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    base_monthly = risk_monthly = 0.0
    changes = []
    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]
        gpu_hours = hpd * DAYS * ngpu
        horizon = num(j["days"]) if j.get("kind") == "train" or j.get("kind") == "dev" else INDEFINITE_HORIZON_DAYS

        base_tier = pricing.recommend_tier(hpd, interruptible)
        risk_tier = pricing.recommend_tier(hpd, interruptible, gpu_type=gtype, job_days=horizon)

        base_cost = TIER_COST[base_tier](gpu_hours, c)
        risk_cost = TIER_COST[risk_tier](gpu_hours, c)
        base_monthly += base_cost
        risk_monthly += risk_cost
        if risk_tier != base_tier:
            changes.append({"job_id": j["job_id"], "gpu_type": gtype,
                            "base_tier": base_tier, "risk_tier": risk_tier,
                            "delta_usd": round(risk_cost - base_cost)})

    base_savings_pct = 0.0  # base policy IS the optimized baseline here; compare vs on-demand instead
    on_demand_monthly = sum(num(j["hours_per_day"]) * DAYS * int(num(j["num_gpus"])) * num(cat[j["gpu_type"]]["on_demand_hr"]) for j in jobs)
    base_savings_pct = (1 - base_monthly / on_demand_monthly) * 100 if on_demand_monthly else 0.0
    risk_savings_pct = (1 - risk_monthly / on_demand_monthly) * 100 if on_demand_monthly else 0.0

    if verbose:
        print("== M3 Extension: Risk-Adjusted Tier Policy (Your Turn #1) ==")
        if changes:
            print("Recommendation changes vs. base policy:")
            for ch in changes:
                print(f"  {ch['job_id']:18} {ch['base_tier']:12} -> {ch['risk_tier']:12}  ({ch['delta_usd']:+,} $/mo)")
        else:
            print("No recommendation changed for this workload mix.")
        print(f"\nmonthly: base policy ${base_monthly:,.0f} ({base_savings_pct:.1f}% vs on-demand)"
              f"  ->  risk-adjusted ${risk_monthly:,.0f} ({risk_savings_pct:.1f}% vs on-demand)")
        print("\nTier matrix — GPU type x duty cycle x interruptible (job_days=365, i.e. steady production):")
        print(f"{'gpu_type':9}{'interrupt':10}{'duty=20%':>14}{'duty=55%':>14}{'duty=100%':>14}")
        for gtype in ("H100", "A100", "A10G", "L4"):
            for interr in (True, False):
                row = [pricing.recommend_tier(duty * 24, interr, gpu_type=gtype, job_days=INDEFINITE_HORIZON_DAYS)
                       for duty in (0.20, 0.55, 1.00)]
                print(f"{gtype:9}{str(interr):10}{row[0]:>14}{row[1]:>14}{row[2]:>14}")

    return {
        "base_monthly": round(base_monthly), "risk_adjusted_monthly": round(risk_monthly),
        "base_savings_pct": round(base_savings_pct, 1), "risk_savings_pct": round(risk_savings_pct, 1),
        "changes": changes,
    }


if __name__ == "__main__":
    run()
    print()
    run_risk_adjusted_policy()
