"""
sizing_experiment_decide.py
===========================
Applies the decision rule of reports/preregistration_sizing_vol.md MECHANICALLY to
the development-period backtests, so the choice is made by the rule written before
the results existed, not by eyeballing them.

    D:\\trader_env\\Scripts\\python.exe sizing_experiment_decide.py [--write]

Reads   reports/backtest_daily_report_dev_{live,ewma_vol_21,ewma_vol_63}.json
Prints  the comparison and the decision; with --write it also writes
        reports/decision_sizing_vol_dev.md (commit that BEFORE any test-period run).

Rule (verbatim from the pre-registration): a candidate qualifies only if, against
live, it has a higher Sharpe ratio by at least 0.05, AND a CAGR that is not lower,
AND a maximum drawdown no worse than 1.25 times live's. If both qualify, take the
higher Sharpe. If neither qualifies, v3 stays unchanged.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date

from titanq_config import PROJECT_DIR

REPORTS = os.path.join(PROJECT_DIR, "reports")
CANDIDATES = ("ewma_vol_21", "ewma_vol_63")


def load(name: str) -> dict:
    with open(os.path.join(REPORTS, f"backtest_daily_report_dev_{name}.json"), encoding="ascii") as f:
        return json.load(f)


def qualifies(c: dict, base: dict) -> tuple:
    checks = {
        "sharpe >= live + 0.05": c["sharpe"] >= base["sharpe"] + 0.05,
        "cagr >= live": c["cagr"] >= base["cagr"],
        "max drawdown <= 1.25 x live": c["max_drawdown"] <= 1.25 * base["max_drawdown"],
    }
    return all(checks.values()), checks


def main() -> None:
    live = load("live")
    base = live["strategy"]
    rows = [("live", base, None)]
    passed = []
    for name in CANDIDATES:
        s = load(name)["strategy"]
        ok, checks = qualifies(s, base)
        rows.append((name, s, checks))
        if ok:
            passed.append((s["sharpe"], name))
    choice = max(passed)[1] if passed else "live"

    lines = [f"# Decision: sizing volatility, development period ({date.today()})", "",
             "Applied mechanically by sizing_experiment_decide.py to the development backtests "
             f"({live['window'][0]} to {live['window'][1]}), using the rule in "
             "reports/preregistration_sizing_vol.md. Written BEFORE any test-period run.", "",
             "| Variant | CAGR | Sharpe | Max drawdown | Avg invested | Checks |",
             "|---|---|---|---|---|---|"]
    for name, s, checks in rows:
        rep = load(name)
        chk = "baseline" if checks is None else ", ".join(
            f"{k}: {'pass' if v else 'FAIL'}" for k, v in checks.items())
        lines.append(f"| {name} | {100 * s['cagr']:.1f}% | {s['sharpe']:.2f} | "
                     f"{100 * s['max_drawdown']:.1f}% | {100 * rep['average_invested']:.1f}% | {chk} |")
    lines += ["", f"**Decision: {choice}.** " + (
        "No candidate qualifies; v3 sizing stays unchanged and no test-period run is needed."
        if choice == "live" else
        f"Next (step 4, once): run {choice} and live over 2021-01-01 to the latest date with "
        f"--tag _test_<name>; adopt {choice} only if its test Sharpe is not below live's."), ""]
    text = "\n".join(lines)
    print(text)
    if "--write" in sys.argv:
        with open(os.path.join(REPORTS, "decision_sizing_vol_dev.md"), "w", encoding="ascii") as f:
            f.write(text)
        print("written: reports/decision_sizing_vol_dev.md")


if __name__ == "__main__":
    main()
