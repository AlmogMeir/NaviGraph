"""Model-fitting and cross-family evaluation for behavioral-state HMMs.

Companion to :mod:`navigraph.analysis.state_features` (features/anchors/
:func:`~navigraph.analysis.state_features.validate_states`, which stays
dependency-light and must not import hmmlearn/IOHMM/ssm). This module holds
everything that *does* need those heavier optional dependencies: fitting
several HMM families on the same per-node-visit feature frame and scoring
them on a shared set of metrics, so they are directly comparable.

Families
--------
- ``IO-HMM`` (:func:`iohmm_scores`) -- input-driven (``IOHMM`` package):
  covariates -> state -> ``turn_type`` emission. No persistence prior.
- ``Sticky-Categorical`` (:func:`categorical_scores`) -- ``hmmlearn``
  ``CategoricalHMM`` on the ``turn_type`` sequence alone (no covariates),
  with a diagonal-heavy ``transmat_prior`` for stickiness.
- ``Sticky-Gaussian`` / plain ``Gaussian`` (:func:`gaussian_scores`) --
  ``hmmlearn`` ``GaussianHMM`` on the continuous covariate matrix, with an
  optional sticky ``transmat_prior``.
- ``ssm GLM-HMM (sticky)`` (:func:`glmhmm_scores`) -- the ``ssm`` package's
  true input-driven GLM-HMM with a built-in sticky transition prior. Optional:
  guarded by :data:`HAS_SSM`, since ``ssm`` needs a C++ compiler to install.

All four ``*_scores`` functions return the same dict schema:
``{family, K, ll, aic, bic, ll_test, seed, model, states}``, so downstream
comparison code never has to branch per family.

Stickiness
----------
:func:`sticky_transmat_prior` and ``ssm``'s ``StickyTransitions`` use the
mathematically equivalent pseudo-count mechanism (confirmed against both
libraries' source): ``kappa`` extra "stayed in this state" counts are added
to the diagonal of the transition-count matrix before normalizing. Using the
same ``kappa`` (default 100, ``ssm``'s own default) in both keeps the
hmmlearn-based and ssm-based sticky variants on equal footing.
"""

from __future__ import annotations

import signal
from contextlib import contextmanager
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from navigraph.analysis.state_features import TURN_CATEGORIES

# ---------------------------------------------------------------------------
# Optional heavy deps
# ---------------------------------------------------------------------------

from hmmlearn.hmm import GaussianHMM, CategoricalHMM

try:
    import ssm  # noqa: F401
    HAS_SSM = True
except Exception:  # pragma: no cover - depends on optional install
    ssm = None
    HAS_SSM = False


def apply_iohmm_sklearn_shim() -> None:
    """IOHMM 0.0.7 calls sklearn's ``label_binarize`` positionally; sklearn
    >=1.3 requires ``classes`` as keyword-only. Patch once; idempotent."""
    import sklearn.preprocessing as _skp
    import IOHMM.linear_models as _iolm

    orig_lb = _skp.label_binarize
    _iolm.label_binarize = lambda Y, classes, **k: orig_lb(Y, classes=classes, **k)


apply_iohmm_sklearn_shim()

from IOHMM import UnSupervisedIOHMM, DiscreteMNL, CrossEntropyMNL  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STICKY_KAPPA = 100          # ssm's own StickyTransitions default; reused for hmmlearn too
DEFAULT_TIMEOUT_S = 90      # hard wall-clock cap per model fit (see time_limit)


# ---------------------------------------------------------------------------
# Infra: wall-clock guard, held-out split, turn encoding, sticky prior
# ---------------------------------------------------------------------------

@contextmanager
def time_limit(seconds: float):
    """Hard wall-clock cap for a single model fit (Unix only, SIGALRM-based).

    A handful of (K, seed) combinations can pathologically slow an EM fit;
    this bounds a K-sweep's worst case regardless of cause.
    """
    def _handler(signum, frame):
        raise TimeoutError(f"fit exceeded {seconds}s")
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def trial_blocked_test_mask(trial_idx: pd.Series, every: int = 5) -> np.ndarray:
    """Boolean test mask: every `every`-th trial id is held out."""
    uniq = np.sort(pd.unique(trial_idx))
    test_trials = set(uniq[::every])
    return np.isin(trial_idx, list(test_trials))


def encode_turns(turn_type: pd.Series) -> np.ndarray:
    """Integer-encode ``turn_type`` in :data:`TURN_CATEGORIES` order (0..4)."""
    cat = pd.Categorical(turn_type, categories=TURN_CATEGORIES)
    return cat.codes.astype(int)


def sticky_transmat_prior(K: int, kappa: float = STICKY_KAPPA) -> np.ndarray:
    """Diagonal-heavy Dirichlet prior over transition-matrix rows.

    ``hmmlearn``'s M-step is ``transmat_ = max(transmat_prior - 1 + counts, 0)``
    (see ``hmmlearn/base.py``), so this adds exactly ``kappa`` pseudo "stayed
    in this state" counts to the diagonal -- the same mechanism as ``ssm``'s
    ``StickyTransitions`` (``expected_joints += kappa * eye(K)``).
    """
    return np.eye(K) * kappa + 1.0


# ---------------------------------------------------------------------------
# Family: IO-HMM (input-driven, IOHMM package) -- reference, no sticky prior
# ---------------------------------------------------------------------------

def _io_frame(Xmat: np.ndarray, turn: np.ndarray, covs: Sequence[str]) -> pd.DataFrame:
    d = pd.DataFrame(Xmat, columns=covs)
    d["turn"] = turn
    return d


def fit_iohmm(K, Xmat, turn, covs, seed=0, max_em_iter=25, em_tol=1e-3,
              inner_max_iter=50, alpha=1.0, timeout_s=DEFAULT_TIMEOUT_S):
    # alpha is the L2 regularization strength for DiscreteMNL/CrossEntropyMNL. IOHMM
    # defaults alpha=0 = no regularization (internally C = 1/alpha -> inf), which lets
    # coefficients blow up toward perfect separation at K>=4 and crashes every seed.
    # alpha=1.0 keeps them bounded (verified: 15/15 seeds converge at K=3/4/5).
    np.random.seed(seed)
    mod = UnSupervisedIOHMM(num_states=K, max_EM_iter=max_em_iter, EM_tol=em_tol)
    mod.set_models(model_emissions=[DiscreteMNL(reg_method="l2", alpha=alpha, max_iter=inner_max_iter)],
                   model_transition=CrossEntropyMNL(solver="lbfgs", reg_method="l2", alpha=alpha,
                                                     max_iter=inner_max_iter),
                   model_initial=CrossEntropyMNL(solver="lbfgs", reg_method="l2", alpha=alpha,
                                                  max_iter=inner_max_iter))
    mod.set_inputs(covariates_initial=[], covariates_transition=list(covs),
                   covariates_emissions=[list(covs)])
    mod.set_outputs([["turn"]])
    mod.set_data([_io_frame(Xmat, turn, covs)])
    with time_limit(timeout_s):
        mod.train()
    return mod


def io_states(mod) -> np.ndarray:
    return np.array(mod.log_gammas[0]).argmax(1)


def iohmm_scores(K, X, turn_out, covs, seeds=range(5), family_label="IO-HMM",
                  timeout_s=DEFAULT_TIMEOUT_S, alpha=1.0) -> Dict:
    best = None
    for s in seeds:
        try:
            mod = fit_iohmm(K, X, turn_out, covs, seed=s, timeout_s=timeout_s, alpha=alpha)
            ll = float(np.array(mod.log_likelihoods)[-1])
        except Exception:
            continue
        if best is None or ll > best[0]:
            best = (ll, mod, s)
    if best is None:
        raise RuntimeError(f"{family_label}: all {len(list(seeds))} seeds failed/timed out at K={K}")
    ll, mod, s = best
    states = io_states(mod)
    p = X.shape[1]
    C = len(set(turn_out))
    n_params = K * C * (p + 1) + K * K * (p + 1) + (K - 1)
    aic = -2 * ll + 2 * n_params
    bic = -2 * ll + n_params * np.log(len(X))
    return dict(family=family_label, K=K, ll=ll, aic=aic, bic=bic, ll_test=np.nan,
                seed=s, model=mod, states=states)


# ---------------------------------------------------------------------------
# Family: Gaussian-HMM (hmmlearn), plain or sticky
# ---------------------------------------------------------------------------

def fit_gaussian(K, Xmat, seed=0, n_iter=150, timeout_s=DEFAULT_TIMEOUT_S,
                  transmat_prior: Optional[np.ndarray] = None):
    kwargs = dict(n_components=K, covariance_type="diag", n_iter=n_iter, random_state=seed)
    if transmat_prior is not None:
        kwargs["transmat_prior"] = transmat_prior
    m = GaussianHMM(**kwargs)
    with time_limit(timeout_s):
        m.fit(Xmat)
    return m


def gaussian_scores(K, X, seeds=range(5), train_mask=None, test_mask=None,
                     transmat_prior: Optional[np.ndarray] = None, family_label="Gaussian",
                     timeout_s=DEFAULT_TIMEOUT_S) -> Dict:
    best = None
    for s in seeds:
        try:
            m = fit_gaussian(K, X, seed=s, transmat_prior=transmat_prior, timeout_s=timeout_s)
            ll = m.score(X)
        except Exception:
            continue
        if best is None or ll > best[0]:
            best = (ll, m, s)
    if best is None:
        raise RuntimeError(f"{family_label}: all {len(list(seeds))} seeds failed/timed out at K={K}")
    ll, m, s = best
    states = m.predict(X)
    n_params = K * X.shape[1] * 2 + K * (K - 1) + (K - 1)   # means+vars + trans + init
    aic = -2 * ll + 2 * n_params
    bic = -2 * ll + n_params * np.log(len(X))
    ll_test = np.nan
    if train_mask is not None and test_mask is not None:
        try:
            m_tr = fit_gaussian(K, X[train_mask], seed=s, transmat_prior=transmat_prior,
                                 timeout_s=timeout_s)
            ll_test = m_tr.score(X[test_mask])
        except Exception:
            pass
    return dict(family=family_label, K=K, ll=ll, aic=aic, bic=bic, ll_test=ll_test,
                seed=s, model=m, states=states)


# ---------------------------------------------------------------------------
# Family: Categorical-HMM on turn_type alone (hmmlearn), plain or sticky
# ---------------------------------------------------------------------------

def fit_categorical(K, turn_codes, seed=0, n_iter=150, timeout_s=DEFAULT_TIMEOUT_S,
                     transmat_prior: Optional[np.ndarray] = None, n_features: Optional[int] = None):
    if n_features is None:
        n_features = len(TURN_CATEGORIES)
    kwargs = dict(n_components=K, n_features=n_features, random_state=seed, n_iter=n_iter)
    if transmat_prior is not None:
        kwargs["transmat_prior"] = transmat_prior
    m = CategoricalHMM(**kwargs)
    Y = np.asarray(turn_codes).reshape(-1, 1)
    with time_limit(timeout_s):
        m.fit(Y)
    return m


def categorical_scores(K, turn_codes, seeds=range(5), train_mask=None, test_mask=None,
                        transmat_prior: Optional[np.ndarray] = None, n_features: Optional[int] = None,
                        family_label="Sticky-Categorical", timeout_s=DEFAULT_TIMEOUT_S) -> Dict:
    if n_features is None:
        n_features = len(TURN_CATEGORIES)
    turn_codes = np.asarray(turn_codes)
    best = None
    for s in seeds:
        try:
            m = fit_categorical(K, turn_codes, seed=s, transmat_prior=transmat_prior,
                                 n_features=n_features, timeout_s=timeout_s)
            ll = m.score(turn_codes.reshape(-1, 1))
        except Exception:
            continue
        if best is None or ll > best[0]:
            best = (ll, m, s)
    if best is None:
        raise RuntimeError(f"{family_label}: all {len(list(seeds))} seeds failed/timed out at K={K}")
    ll, m, s = best
    states = m.predict(turn_codes.reshape(-1, 1))
    n_params = K * (n_features - 1) + K * (K - 1) + (K - 1)   # emission + trans + init
    aic = -2 * ll + 2 * n_params
    bic = -2 * ll + n_params * np.log(len(turn_codes))
    ll_test = np.nan
    if train_mask is not None and test_mask is not None:
        try:
            m_tr = fit_categorical(K, turn_codes[train_mask], seed=s, transmat_prior=transmat_prior,
                                    n_features=n_features, timeout_s=timeout_s)
            ll_test = m_tr.score(turn_codes[test_mask].reshape(-1, 1))
        except Exception:
            pass
    return dict(family=family_label, K=K, ll=ll, aic=aic, bic=bic, ll_test=ll_test,
                seed=s, model=m, states=states)


# ---------------------------------------------------------------------------
# Family: ssm GLM-HMM with sticky transitions (optional, guarded by HAS_SSM)
# ---------------------------------------------------------------------------

def fit_glmhmm(K, X, turn_codes, seed=0, kappa=STICKY_KAPPA, num_iters=150):
    if not HAS_SSM:
        raise RuntimeError(
            "ssm not installed. Install with:\n"
            "  sudo dnf install -y gcc-c++\n"
            '  .venv/bin/pip install "git+https://github.com/lindermanlab/ssm.git"\n'
            "(PyPI's `ssm` package is a different, unrelated, non-building package -- do not use it.)"
        )
    np.random.seed(seed)
    turn_codes = np.asarray(turn_codes).reshape(-1, 1)
    C = len(TURN_CATEGORIES)
    model = ssm.HMM(K, D=1, M=X.shape[1], observations="input_driven_obs",
                     observation_kwargs=dict(C=C),
                     transitions="sticky", transition_kwargs=dict(kappa=kappa))
    model.fit([turn_codes], inputs=[X], method="em", num_iters=num_iters, verbose=0)
    return model


def glmhmm_scores(K, X, turn_codes, seeds=range(5), kappa=STICKY_KAPPA, num_iters=150,
                   family_label="ssm GLM-HMM (sticky)") -> Dict:
    if not HAS_SSM:
        raise RuntimeError("ssm not installed; see fit_glmhmm for install instructions")
    turn_codes = np.asarray(turn_codes)
    turn_col = turn_codes.reshape(-1, 1)
    best = None
    for s in seeds:
        try:
            mod = fit_glmhmm(K, X, turn_codes, seed=s, kappa=kappa, num_iters=num_iters)
            ll = float(mod.log_likelihood([turn_col], inputs=[X]))
        except Exception:
            continue
        if best is None or ll > best[0]:
            best = (ll, mod, s)
    if best is None:
        raise RuntimeError(f"{family_label}: all {len(list(seeds))} seeds failed/timed out at K={K}")
    ll, mod, s = best
    states = np.asarray(mod.most_likely_states(turn_col, input=X))
    C = len(TURN_CATEGORIES)
    p = X.shape[1]
    n_params = K * C * (p + 1) + K * K * (p + 1) + (K - 1)   # parity with IO-HMM's count
    aic = -2 * ll + 2 * n_params
    bic = -2 * ll + n_params * np.log(len(turn_codes))
    return dict(family=family_label, K=K, ll=ll, aic=aic, bic=bic, ll_test=np.nan,
                seed=s, model=mod, states=states)


# ---------------------------------------------------------------------------
# Cross-family evaluation helpers (state-column-agnostic; promoted from the
# original single-family notebook, state_hmm.ipynb)
# ---------------------------------------------------------------------------

def empirical_transition_matrix(states, K: int) -> np.ndarray:
    """Row-normalized count of observed state[t] -> state[t+1] transitions."""
    states = np.asarray(states)
    mat = np.zeros((K, K))
    for a, b in zip(states[:-1], states[1:]):
        mat[a, b] += 1
    row_sums = mat.sum(axis=1, keepdims=True)
    return np.divide(mat, row_sums, out=np.zeros_like(mat), where=row_sums > 0)


def state_run_lengths(states) -> Dict[int, list]:
    """Consecutive-node-visit run-length distribution, per state.

    A "run" is a maximal stretch of consecutive node-visits assigned the same
    state (i.e. how many nodes in a row the model says the mouse stayed in
    that state before switching). Same run-detection logic as
    :func:`navigraph.analysis.state_features.validate_states`'s stickiness
    block, but returns the raw list of run lengths per state (not just the
    mean) so the full distribution can be plotted.
    """
    states = np.asarray(states)
    out: Dict[int, list] = {}
    if len(states) == 0:
        return out
    cur_state = states[0]
    run = 1
    for v in states[1:]:
        if v == cur_state:
            run += 1
        else:
            out.setdefault(int(cur_state), []).append(run)
            cur_state = v
            run = 1
    out.setdefault(int(cur_state), []).append(run)
    return out


def alternation_purity(feat: pd.DataFrame, state_col: str):
    """Per-alternation-segment fraction of node-visits in that segment's single
    most common state. One value per direct within-patch alternation path.

    Returns (purities, dominant_states, seg_ids), each length = n_segments.
    """
    seg_ids = feat.loc[feat["anchor_exploit"], "_seg"].unique()
    purities, dominant_states = [], []
    for seg in seg_ids:
        vals = feat.loc[feat["_seg"] == seg, state_col]
        vc = vals.value_counts(normalize=True)
        purities.append(vc.iloc[0])
        dominant_states.append(vc.index[0])
    return np.array(purities), np.array(dominant_states), seg_ids


def change_of_mind_window(feat: pd.DataFrame, state_col: str, window: int = 6):
    """State trace around each change-of-mind apex, relative node-visit index
    offset -window..+window (0 = the apex itself). Returns (matrix, apex_idx).
    """
    apex_idx = feat.loc[feat["anchor_change_of_mind"], "node_visit_idx"].values
    state_arr = feat.set_index("node_visit_idx")[state_col]
    mat = np.full((len(apex_idx), 2 * window + 1), np.nan)
    for i, a in enumerate(apex_idx):
        for j, off in enumerate(range(-window, window + 1)):
            mat[i, j] = state_arr.get(a + off, np.nan)
    return mat, apex_idx


def turn_prediction_uplift(X, turn_out, state_seq, train_mask, test_mask) -> Dict:
    """Held-out turn_type accuracy: covariates alone vs. covariates + one-hot state.

    Well-defined and comparable across *every* family (not just the ones whose
    emission natively models turn_type) since it's a downstream logistic
    regression, not the HMM's own likelihood.
    """
    y = np.asarray(turn_out)
    lr = LogisticRegression(max_iter=500)
    lr.fit(X[train_mask], y[train_mask])
    acc_cov = accuracy_score(y[test_mask], lr.predict(X[test_mask]))

    onehot = pd.get_dummies(np.asarray(state_seq)).values.astype(float)
    Xs = np.hstack([X, onehot])
    lr2 = LogisticRegression(max_iter=500)
    lr2.fit(Xs[train_mask], y[train_mask])
    acc_state = accuracy_score(y[test_mask], lr2.predict(Xs[test_mask]))

    return dict(acc_covariates_only=acc_cov, acc_covariates_plus_state=acc_state,
                uplift=acc_state - acc_cov)


__all__ = [
    "HAS_SSM", "STICKY_KAPPA", "DEFAULT_TIMEOUT_S",
    "apply_iohmm_sklearn_shim", "time_limit", "trial_blocked_test_mask",
    "encode_turns", "sticky_transmat_prior",
    "fit_iohmm", "io_states", "iohmm_scores",
    "fit_gaussian", "gaussian_scores",
    "fit_categorical", "categorical_scores",
    "fit_glmhmm", "glmhmm_scores",
    "empirical_transition_matrix", "state_run_lengths", "alternation_purity", "change_of_mind_window",
    "turn_prediction_uplift",
]
