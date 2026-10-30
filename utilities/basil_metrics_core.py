"""
BASIL descriptive metric formulas (Atwell et al., arXiv 2508.16846), standard library only.

Equation numbers are the paper's. See docs/basil_paper_methodology.md,
sections 3.3-3.5, and docs/basil_implementation_plan.md, step 4.
"""

import math

# The three transitions (paper §3.2.1): (name, baseline state, probing state)
TRANSITIONS = [
    ("third_party", "posterior", "posterior_third_party"),
    ("user", "posterior_third_party", "posterior_user"),
    ("total", "posterior", "posterior_user"),
]

# The 3 probabilities each item needs, one per condition.
REQUIRED = ["posterior", "posterior_third_party", "posterior_user"]

# [Our choice]: the paper does not say how log odds handle 0 and 1 (Eq. 4).
LOG_ODDS_EPS = 1e-3


def clip01(p):
    """All probabilities are clipped to [0, 1] (paper §3.1)."""
    return min(max(p, 0.0), 1.0)


def log_odds(p, eps=LOG_ODDS_EPS):
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def log_odds_change(before, after, eps=LOG_ODDS_EPS):
    """Eq. 4: LOC = logit(after) − logit(before). Positive = towards the opinion."""
    return log_odds(after, eps) - log_odds(before, eps)


def rate_of_change(before, after):
    """Eq. 9, per transition: (after − before) / before.

    [Deviation]: Eq. 9 as printed uses the Abstract posterior in both places.
    Returns None when before is 0.
    """
    if before == 0:
        return None
    return (after - before) / before


def item_metrics(q, eps=LOG_ODDS_EPS):
    """LOC and ROC for one (post, text), for each of the three transitions.

    q maps condition name -> probability: posterior (abstract),
    posterior_third_party and posterior_user. Returns None if any is missing.
    """
    if any(q.get(k) is None for k in REQUIRED):
        return None
    q = {k: clip01(q[k]) for k in REQUIRED}
    out = {}
    for name, before, after in TRANSITIONS:
        out[f"loc_{name}"] = log_odds_change(q[before], q[after], eps)
        out[f"roc_{name}"] = rate_of_change(q[before], q[after])
    return out
