"""Tests for the "Your Turn" extensions (additive — does not modify graded tests).

Covers: risk-adjusted recommend_tier() (#1) and cache_is_worth_it() (#3).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finops import pricing


def test_recommend_tier_base_policy_unchanged():
    # calling with only the original args must reproduce the exact base policy
    assert pricing.recommend_tier(2, True) == "spot"
    assert pricing.recommend_tier(24, False) == "reserved"
    assert pricing.recommend_tier(4, False) == "on_demand"


def test_recommend_tier_gpu_risk_overrides_spot():
    # A10G's spot pool reclaims above SPOT_RISK_CEILING -> too flaky for spot
    # even though the job is interruptible; falls through to on-demand.
    assert pricing.recommend_tier(8, True, gpu_type="A10G") == "on_demand"
    # H100's deep pool stays under the ceiling -> spot still recommended
    assert pricing.recommend_tier(8, True, gpu_type="H100") == "spot"


def test_recommend_tier_job_days_horizon():
    # steady duty cycle but short horizon -> shallower 1yr term, not 3yr
    assert pricing.recommend_tier(24, False, gpu_type="H100", job_days=200) == "reserved_1yr"
    # long horizon -> deeper 3yr commitment
    assert pricing.recommend_tier(24, False, gpu_type="H100", job_days=600) == "reserved_3yr"
    # too short for any commitment, falls back to on-demand
    assert pricing.recommend_tier(24, False, gpu_type="H100", job_days=30) == "on_demand"


def test_cache_breakeven_reads_is_tier_independent():
    # break-even = write_premium / (1 - read_discount), independent of price level
    be_small = pricing.cache_breakeven_reads(0.20 * 1.25, 0.20)
    be_large = pricing.cache_breakeven_reads(3.00 * 1.25, 3.00)
    assert abs(be_small - be_large) < 1e-9
    assert abs(be_small - (1.25 / 0.9)) < 1e-9


def test_cache_is_worth_it():
    write_cost = 3.00 * 1.25
    assert pricing.cache_is_worth_it(50, write_cost, 3.00) is True    # reused constantly
    assert pricing.cache_is_worth_it(1, write_cost, 3.00) is False    # one-off prefix
