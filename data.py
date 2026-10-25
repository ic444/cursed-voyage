#Synthetic data for the AI Captain game.
import numpy as np
import pandas as pd

ROUTE_NAMES = ["A · North", "B · Middle", "C · South"]
NOISE_SD = 0.4  # random variation in real voyages (days)


def true_days(distance, waves, wind):
    """Therule used to generate voyages
    days = 4 + distance/50 + 0.8*waves + 0.1*wind + noise"""
    return 4 + distance / 50 + 0.8 * waves + 0.1 * wind


def make_voyages(n=300, seed=1031):
    rng = np.random.default_rng(seed)
    distance = rng.uniform(250, 450, n)
    waves = rng.uniform(0.5, 4.5, n)
    wind = rng.uniform(0, 15, n)
    days = true_days(distance, waves, wind) + rng.normal(0, NOISE_SD, n)
    return pd.DataFrame({
        "distance": distance.round(0),
        "waves": waves.round(1),
        "wind": wind.round(0),
        "days": days.round(1),
    })

def make_scenario(seed):
  
    rng = np.random.default_rng(seed)
    ranges = [(380, 450), (320, 380), (250, 320)]
    rows = []
    for name, (lo, hi) in zip(ROUTE_NAMES, ranges):
        distance = int(rng.integers(lo, hi))
        waves = round(float(rng.uniform(0.5, 4.5)), 1)
        wind = int(rng.integers(0, 16))
        truth = true_days(distance, waves, wind) + rng.normal(0, NOISE_SD)
        rows.append({"route": name, "distance": distance, "waves": waves,
                     "wind": wind, "truth": round(float(truth), 1)})
    return pd.DataFrame(rows)