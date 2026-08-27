"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num
from finops import pricing, sustainability

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    base_cost = opt_cost = 0.0
    total_tokens = 0
    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        total_tokens += inp + out
        # BASELINE: naive deployment — everything on the large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        base_cost += pricing.request_cost(inp, out, lin, lout)
        # OPTIMIZED: cascade (route_tier), prompt caching, batch API
        pin, pout = MODEL_PRICES[r["route_tier"]]
        opt_cost += pricing.request_cost(inp, out, pin, pout, cached_in=cached, batch=is_batch)

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
    }


CACHE_WRITE_PREMIUM = 1.25  # writing a prefix to cache ~1.25x its base input price


def cache_economics(verbose: bool = True) -> dict:
    """Your Turn #3 — gate cache savings on finops.pricing.cache_is_worth_it().

    Groups requests by (team, route_tier) as a proxy for "how many times a
    day is this team's cached system-prompt prefix read back". Caching pays
    for its write premium once reuse clears cache_breakeven_reads(); a prefix
    reused only once or twice would actually cost MORE than never caching.
    """
    rows = load_csv("token_usage.csv")
    groups = defaultdict(int)
    for r in rows:
        if int(num(r["cached_input_tokens"])) > 0:
            groups[(r["team"], r["route_tier"])] += 1

    results = []
    for (team, tier), reads_per_day in groups.items():
        price_in, _ = MODEL_PRICES[tier]
        write_cost = price_in * CACHE_WRITE_PREMIUM
        breakeven = pricing.cache_breakeven_reads(write_cost, price_in)
        worth_it = pricing.cache_is_worth_it(reads_per_day, write_cost, price_in)
        results.append({"team": team, "tier": tier, "reads_per_day": reads_per_day,
                        "breakeven_reads": round(breakeven, 2), "worth_it": worth_it})

    # boundary illustration: a prefix used only once (e.g. a one-off eval run)
    one_off_worth_it = pricing.cache_is_worth_it(1, MODEL_PRICES["large"][0] * CACHE_WRITE_PREMIUM, MODEL_PRICES["large"][0])

    if verbose:
        print("== M2 Extension: Cache Economics (Your Turn #3) ==")
        print(f"write premium: {CACHE_WRITE_PREMIUM}x input price | break-even ~= {CACHE_WRITE_PREMIUM/0.9:.2f} reads (tier-independent ratio)")
        print(f"{'team':11}{'tier':7}{'reads/day':>11}{'breakeven':>11}{'worth it?':>11}")
        for r in sorted(results, key=lambda x: -x["reads_per_day"]):
            print(f"{r['team']:11}{r['tier']:7}{r['reads_per_day']:>11}{r['breakeven_reads']:>11}{'YES' if r['worth_it'] else 'NO':>11}")
        print(f"\nBoundary case — a prefix read exactly once (one-off request): worth caching? {one_off_worth_it}")
        print("-> every team here reuses its system prompt dozens of times/day, clearing break-even by 1-2 orders")
        print("   of magnitude; only genuinely one-shot prompts (ad-hoc eval runs) should skip caching.")

    return {"groups": results, "one_off_worth_it": one_off_worth_it}


REASONING_TARGET_TRAFFIC_FRAC = 0.10


def reasoning_budget_analysis(verbose: bool = True) -> dict:
    """Your Turn #4 — split $/Wh cost of is_reasoning traffic and estimate the
    saving from capping reasoning to REASONING_TARGET_TRAFFIC_FRAC of traffic.

    Downgraded requests reverse the generator's reasoning tax (data/generate.py:
    ~6x output tokens, sustainability.REASONING_ENERGY_MULTIPLIER=80x energy).
    """
    rows = load_csv("token_usage.csv")
    n_total = len(rows)
    reasoning_rows = [r for r in rows if int(num(r["is_reasoning"])) == 1]
    normal_rows = [r for r in rows if int(num(r["is_reasoning"])) == 0]

    def cost_and_wh(rs, is_reasoning):
        cost = wh = 0.0
        for r in rs:
            inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
            pin, pout = MODEL_PRICES[r["route_tier"]]
            cost += pricing.request_cost(inp, out, pin, pout,
                                         cached_in=int(num(r["cached_input_tokens"])),
                                         batch=bool(int(num(r["is_batch"]))))
            wh += sustainability.wh_per_query(inp + out, is_reasoning=is_reasoning)
        return cost, wh

    reasoning_cost, reasoning_wh = cost_and_wh(reasoning_rows, True)
    normal_cost, normal_wh = cost_and_wh(normal_rows, False)
    total_cost, total_wh = reasoning_cost + normal_cost, reasoning_wh + normal_wh
    traffic_pct = len(reasoning_rows) / n_total * 100 if n_total else 0.0
    cost_pct = reasoning_cost / total_cost * 100 if total_cost else 0.0
    wh_pct = reasoning_wh / total_wh * 100 if total_wh else 0.0

    def cap_to(frac):
        target_n = int(frac * n_total)
        if len(reasoning_rows) <= target_n:
            return total_cost, total_wh, target_n
        ordered = sorted(reasoning_rows, key=lambda r: int(num(r["output_tokens"])))
        keep, downgrade = ordered[:target_n], ordered[target_n:]
        keep_cost, keep_wh = cost_and_wh(keep, True)
        down_cost = down_wh = 0.0
        for r in downgrade:
            inp = int(num(r["input_tokens"]))
            out_dg = max(1, int(num(r["output_tokens"])) // 6)  # reverse the ~6x reasoning output tax
            pin, pout = MODEL_PRICES[r["route_tier"]]
            down_cost += pricing.request_cost(inp, out_dg, pin, pout,
                                              cached_in=int(num(r["cached_input_tokens"])),
                                              batch=bool(int(num(r["is_batch"]))))
            down_wh += sustainability.wh_per_query(inp + out_dg, is_reasoning=False)
        return keep_cost + down_cost + normal_cost, keep_wh + down_wh + normal_wh, target_n

    cap10_cost, cap10_wh, target_n10 = cap_to(REASONING_TARGET_TRAFFIC_FRAC)
    cap5_cost, cap5_wh, target_n5 = cap_to(0.05)
    cost_saved, wh_saved = total_cost - cap10_cost, total_wh - cap10_wh
    cost_saved5, wh_saved5 = total_cost - cap5_cost, total_wh - cap5_wh

    if verbose:
        print("== M2 Extension: Reasoning Budget (Your Turn #4) ==")
        print(f"reasoning traffic: {len(reasoning_rows)}/{n_total} requests ({traffic_pct:.1f}% of traffic)")
        print(f"reasoning cost share: {cost_pct:.1f}% of $  |  reasoning energy share: {wh_pct:.1f}% of Wh")
        print(f"-> reasoning tokens cost {cost_pct/traffic_pct:.1f}x its traffic share in $ (6x output tokens)")
        print(f"   but {wh_pct/traffic_pct:.1f}x its traffic share in Wh (80x energy multiplier, deck sec.11) --")
        print("   $ pricing barely reflects the true energy cost of reasoning tokens.")
        print(f"\nCurrent traffic ({traffic_pct:.1f}%) is already under a {REASONING_TARGET_TRAFFIC_FRAC:.0%} cap "
              f"({len(reasoning_rows)} <= {target_n10} requests) -> capping alone saves "
              f"${cost_saved:.2f}/day, {wh_saved:.0f} Wh/day (no forced downgrades needed).")
        print(f"A tighter 5% cap ({target_n5} requests) WOULD force downgrades: "
              f"${cost_saved5:.2f}/day saved ({cost_saved5/total_cost*100:.1f}%), "
              f"{wh_saved5:.0f} Wh/day saved ({wh_saved5/total_wh*100:.1f}%).")
        print("Recommended routing rule: don't cap by raw traffic quota -- gate reasoning on the small")
        print("model's own confidence/self-consistency score (e.g. only escalate when confidence < 0.6);")
        print("that targets the queries that actually need it instead of rationing a fixed budget.")

    return {
        "reasoning_traffic_pct": round(traffic_pct, 1), "reasoning_cost_pct": round(cost_pct, 1),
        "reasoning_wh_pct": round(wh_pct, 1),
        "cap10_cost_saved_per_day": round(cost_saved, 2), "cap10_wh_saved_per_day": round(wh_saved, 1),
        "cap5_cost_saved_per_day": round(cost_saved5, 2), "cap5_wh_saved_per_day": round(wh_saved5, 1),
    }


if __name__ == "__main__":
    run()
    print()
    cache_economics()
    print()
    reasoning_budget_analysis()
