#!/usr/bin/env python3
"""goal_feasibility.py -- distance from each trial's goal to the nearest occupied cell."""
import glob
import json
import sys

import numpy as np
import yaml
from PIL import Image
from scipy.ndimage import distance_transform_edt

INSCRIBED_TUCKED = 0.24      # m, from the tucked footprint polygon
XY_TOL = 0.35                # general_goal_checker.xy_goal_tolerance


def main(map_yaml, trial_dir):
    m = yaml.safe_load(open(map_yaml))
    img_path = map_yaml.rsplit("/", 1)[0] + "/" + m["image"]
    img = np.array(Image.open(img_path).convert("L"))
    res, origin = float(m["resolution"]), m["origin"]
    occ_thresh = float(m.get("occupied_thresh", 0.65))

    # Occupied where the greyscale value is dark enough.
    occupied = (img.astype(float) / 255.0) < (1.0 - occ_thresh)
    dist_cells = distance_transform_edt(~occupied)
    dist_m = dist_cells * res
    h = img.shape[0]

    need = INSCRIBED_TUCKED + XY_TOL + res
    print(f"Required goal clearance: {need:.2f} m "
          f"(inscribed {INSCRIBED_TUCKED} + xy_tol {XY_TOL} + res {res})\n")

    files = sorted(glob.glob(trial_dir + "/trial_*.json"))
    if not files:
        print(f"No trial_*.json files found in {trial_dir}")
        return

    for f in files:
        with open(f) as fp:
            d = json.load(fp)
        g = d.get("goal_pose")
        if not g:
            continue
        col = int((g["x"] - origin[0]) / res)
        row = h - 1 - int((g["y"] - origin[1]) / res)
        if not (0 <= row < dist_m.shape[0] and 0 <= col < dist_m.shape[1]):
            print(f"{f.split('/')[-1]:<40} goal off map")
            continue
        c = dist_m[row, col]
        verdict = "OK" if c >= need else ("MARGINAL" if c >= INSCRIBED_TUCKED else "INFEASIBLE")
        status = d.get("status", "UNKNOWN")
        print(f"{f.split('/')[-1]:<40} clearance={c:5.2f} m  {status:<10} {verdict}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 goal_feasibility.py <map_yaml> <trial_dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
