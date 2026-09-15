#!/usr/bin/env python3
"""verify_tuner_runtime.py -- run against evaluation_results/trials/*.json"""
import json, glob, sys, os

def main():
    trials_dir = sys.argv[1] if len(sys.argv) > 1 else "evaluation_results/trials"
    json_files = sorted(glob.glob(os.path.join(trials_dir, "trial_*.json")))

    if not json_files:
        print(f"No trial JSON files found in {trials_dir}")
        sys.exit(0)

    fail = 0
    for f in json_files:
        try:
            d = json.load(open(f))
        except Exception as e:
            print(f"[FAIL] {f}: error reading JSON ({e})")
            fail += 1
            continue

        dl = d.get("tuner_decision_log") or []
        rh = d.get("risk_state_history") or []
        if not dl or not rh:
            print(f"[FAIL] {f}: empty decision log or risk history"); fail += 1; continue

        rw_run = [r["r_width"] for r in rh if isinstance(r, dict) and "r_width" in r]
        rw_tun = [x["risk"][1] for x in dl if x.get("risk") and len(x["risk"]) > 1]

        if rw_run and rw_tun:
            span_run = max(rw_run) - min(rw_run)
            span_tun = max(rw_tun) - min(rw_tun)
            if span_run > 0.5 and span_tun < 0.5 * span_run:
                print(f"[FAIL] {f}: D1 stale context. runner span {span_run:.2f} m, "
                      f"tuner span {span_tun:.2f} m"); fail += 1

        run = 0
        for x in dl:
            if x.get("arm_target") == "tucked" and x.get("arm_verified") == "carry":
                run += 1
                if run > 5:
                    print(f"[FAIL] {f}: D2 arm never retracted "
                          f"(n_arm_switches={d.get('n_arm_switches')})"); fail += 1; break
            else:
                run = 0

        for k in ("n_stale_ticks", "n_tick_overruns"):
            if d.get(k, 0):
                print(f"[FAIL] {f}: {k} = {d[k]}"); fail += 1

        if rw_run and min(rw_run) < 1.173 and d.get("n_arm_switches", 0) == 0:
            print(f"[FAIL] {f}: passed a constriction with no arm switch"); fail += 1

    print("PASS" if fail == 0 else f"{fail} check(s) failed")
    sys.exit(1 if fail else 0)

if __name__ == "__main__":
    main()
