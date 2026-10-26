import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from models import FEATURES

LAMBDA = 6.0  # days of delay we accept to avoid one full unit of risk
LETTERS = ["A", "B", "C"]
DIST_RANGES = [(380, 450), (320, 380), (250, 320)]  # same as the game scenarios
COMPASS_FEATURES = ([f"{L} days" for L in LETTERS] + [f"{L} risk" for L in LETTERS])


def risk_score(waves):
    """Storm risk between 0 and 1: 0 for calm seas, growing with wave height."""
    return float(np.clip(0.25 * (waves - 0.7), 0, 1))


def _vector(days, risks):
    return list(days) + list(risks)


def train_compass(model, n=3000, seed=7, max_depth=4):
    """Simulate n scenarios, label them with the lowest-cost route, fit the tree.
    Returns (tree, accuracy on 25% of scenarios the tree never saw)."""
    rng = np.random.default_rng(seed)
    dist = np.column_stack([rng.integers(lo, hi, n) for lo, hi in DIST_RANGES])
    waves = rng.uniform(0.5, 4.5, (n, 3)).round(1)
    wind = rng.integers(0, 16, (n, 3))
    X = pd.DataFrame({"distance": dist.ravel(), "waves": waves.ravel(),
                      "wind": wind.ravel()})[FEATURES]
    days = model.predict(X).reshape(n, 3)          # one call for all 9000 routes
    risks = np.clip(0.25 * (waves - 0.7), 0, 1)    # same formula as risk_score
    labels = (days + LAMBDA * risks).argmin(axis=1)
    rows = np.hstack([days, risks])
    F = pd.DataFrame(rows, columns=COMPASS_FEATURES)
    F_tr, F_te, y_tr, y_te = train_test_split(F, labels, test_size=0.25, random_state=0)
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=0).fit(F_tr, y_tr)
    return tree, float(tree.score(F_te, y_te))


def recommend(tree, days, risks):
    """Returns (index of recommended route, list of sentences: the tree's questions)."""
    x = pd.DataFrame([_vector(days, risks)], columns=COMPASS_FEATURES)
    choice = int(tree.predict(x)[0])
    t = tree.tree_
    steps = []
    for node in tree.decision_path(x).indices:
        if t.children_left[node] == t.children_right[node]:  # leaf
            continue
        name = COMPASS_FEATURES[t.feature[node]]
        value, thr = float(x.iloc[0, t.feature[node]]), float(t.threshold[node])
        letter, kind = name.split()
        if kind == "days":
            question = f"Is the predicted time of route {letter} at most {thr:.1f} days?"
            fact = f"{value:.1f} days"
        else:
            question = f"Is the storm risk of route {letter} at most {thr:.2f}?"
            fact = f"{value:.2f}"
        steps.append(f"{question} **{'Yes' if value <= thr else 'No'}** (it is {fact})")
    return choice, steps