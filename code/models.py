"""Proposed Conformal Hybrid IDS (CH-IDS) and baseline detectors."""
import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import lightgbm as lgb
import xgboost as xgb
import os
NJ = int(os.environ.get("CHIDS_NJOBS", "2"))


# ----------------------------------------------------------------------------- utilities
def conformal_pvalues(cal_scores, scores):
    """p(x) = (1 + #{s_c >= s(x)}) / (n + 1): valid p-value under exchangeability of benign flows."""
    c = np.sort(cal_scores)
    n = len(c)
    ge = n - np.searchsorted(c, scores, side="left")
    return (1.0 + ge) / (n + 1.0)


# ----------------------------------------------------------------------------- Tier 1: rules
class RuleLayer:
    """Signature-style rules induced from labelled flows.
    A shallow tree is grown; each leaf is a conjunctive IF-THEN rule. A leaf is kept as an
    attack signature only if it is (i) attack-pure on the fit data (precision >= prec),
    (ii) supported by >= min_support flows and (iii) fires on no benign calibration flow."""

    def __init__(self, max_depth=8, min_leaf=30, prec=0.999, min_support=50, seed=0):
        self.tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_leaf,
                                           random_state=seed)
        self.prec, self.min_support = prec, min_support

    def fit(self, Xf, yf, Xc, yc):
        self.tree.fit(Xf, yf)
        lf = self.tree.apply(Xf)
        lc = self.tree.apply(Xc)
        self.keep = set()
        for leaf in np.unique(lf):
            m = lf == leaf
            n, a = m.sum(), yf[m].sum()
            if n >= self.min_support and a / n >= self.prec:
                if ((lc == leaf) & (yc == 0)).sum() == 0:
                    self.keep.add(leaf)
        self.keep_arr = np.array(sorted(self.keep))
        return self

    def match(self, X):
        if len(self.keep_arr) == 0:
            return np.zeros(len(X), bool)
        return np.isin(self.tree.apply(X), self.keep_arr)

    def rules(self, names):
        """Return list of (conditions, support) for kept leaves."""
        t = self.tree.tree_
        out = []

        def rec(node, conds):
            if t.feature[node] == _tree.TREE_UNDEFINED:
                if node in self.keep:
                    out.append((list(conds), int(t.n_node_samples[node])))
                return
            f, th = names[t.feature[node]], t.threshold[node]
            rec(t.children_left[node], conds + [(f, "<=", th)])
            rec(t.children_right[node], conds + [(f, ">", th)])
        rec(0, [])
        return out


# ----------------------------------------------------------------------------- Tier 3: anomaly
class AutoEncoder:
    """Under-complete autoencoder trained on benign flows only; score = reconstruction MSE."""

    def __init__(self, seed=0, hidden=(64, 16, 64), max_n=60000):
        self.m = MLPRegressor(hidden_layer_sizes=hidden, activation="relu", solver="adam",
                              batch_size=256, learning_rate_init=1e-3, max_iter=60,
                              early_stopping=True, n_iter_no_change=5, random_state=seed)
        self.max_n, self.seed = max_n, seed

    def fit(self, Xb):
        rng = np.random.default_rng(self.seed)
        if len(Xb) > self.max_n:
            Xb = Xb[rng.choice(len(Xb), self.max_n, replace=False)]
        Xb = np.clip(Xb, -10, 10)
        self.m.fit(Xb, Xb)
        return self

    def score(self, X):
        X = np.clip(X, -10, 10)
        return ((self.m.predict(X) - X) ** 2).mean(1)


def make_lgbm(seed):
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              random_state=seed, n_jobs=NJ, verbose=-1)


# ----------------------------------------------------------------------------- proposed
def loo_pvalues(c):
    """Leave-one-out conformal p-values of the calibration scores themselves."""
    c = np.asarray(c)
    srt = np.sort(c)
    ge = len(c) - np.searchsorted(srt, c, side="left")  # includes itself
    return ge / len(c)


def tippett(ps, pa, lam):
    """Weighted Tippett statistic; small = more anomalous. lam = share of budget for anomaly tier."""
    if lam <= 0:
        return ps
    if lam >= 1:
        return pa
    return np.minimum(ps / (1 - lam), pa / lam)


def calibrated_threshold(Tcal, alpha):
    """Largest t such that the fraction of benign calibration points with T <= t is <= alpha."""
    srt = np.sort(Tcal)
    k = int(np.floor(alpha * (len(srt) + 1))) - 1
    if k < 0:
        return 0.0
    return srt[k]


LAM_GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9)


class CHIDS:
    """Conformal Hybrid IDS (CH-IDS).
    Tier 1  induced signature rules: fast path, decides immediately on a match
    Tier 2  supervised gradient-boosting detector  -> conformal p-value p_s
    Tier 3  benign-only autoencoder                  -> conformal p-value p_a
    Fusion  weighted Tippett statistic T = min(p_s/(1-lam), p_a/lam), thresholded at the
            alpha-quantile of T over benign calibration flows (joint calibration), so that the
            false-alarm rate on exchangeable benign traffic is ~alpha.
    lam     'auto': chosen by an inner leave-one-family-out simulation on training data only
            (zero-day-aware budget allocation), or a fixed number."""

    def __init__(self, alpha=0.01, lam="auto", seed=0, use_rules=True, use_sup=True, use_ano=True,
                 ano=None, fam=None, inner_trees=100):
        self.alpha, self.lam, self.seed = alpha, lam, seed
        self.use_rules, self.use_sup, self.use_ano = use_rules, use_sup, use_ano
        self.ano, self.inner_trees = ano, inner_trees

    def _select_lambda(self, Xf, yf, ff, Xc, yc, fc, ca, pa_cal):
        fams = [f for f in np.unique(ff[yf == 1]) if ((fc == f).sum() >= 5)]
        dr = {l: [] for l in LAM_GRID}
        bc = Xc[yc == 0]
        pa_lib = None
        for f in fams:
            keep = ff != f
            m = lgb.LGBMClassifier(n_estimators=self.inner_trees, learning_rate=0.1, num_leaves=63,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                   random_state=self.seed, n_jobs=NJ, verbose=-1).fit(Xf[keep], yf[keep])
            cs = m.predict_proba(bc)[:, 1]
            ps_cal = loo_pvalues(cs)
            Xa = Xc[fc == f]
            ps_a = conformal_pvalues(cs, m.predict_proba(Xa)[:, 1])
            pa_a = conformal_pvalues(ca, self.ano.score(Xa))
            for l in LAM_GRID:
                t = calibrated_threshold(tippett(ps_cal, pa_cal, l), self.alpha)
                dr[l].append(np.mean(tippett(ps_a, pa_a, l) <= t))
        self.inner_dr = {l: float(np.mean(v)) for l, v in dr.items()}
        best = max(LAM_GRID, key=lambda l: (round(self.inner_dr[l], 4), -l))
        return best

    def fit(self, Xf, yf, Xc, yc, ff=None, fc=None):
        t0 = time.time()
        if self.use_rules:
            self.rl = RuleLayer(seed=self.seed).fit(Xf, yf, Xc, yc)
        if self.use_sup:
            self.sup = make_lgbm(self.seed).fit(Xf, yf)
            self.cal_s = self.sup.predict_proba(Xc[yc == 0])[:, 1]
        if self.use_ano:
            if self.ano is None:
                self.ano = AutoEncoder(seed=self.seed).fit(Xf[yf == 0])
            self.cal_a = self.ano.score(Xc[yc == 0])
        if self.use_sup and self.use_ano:
            pa_cal = loo_pvalues(self.cal_a)
            if self.lam == "auto":
                self.lam_ = self._select_lambda(Xf, yf, ff, Xc, yc, fc, self.cal_a, pa_cal)
            else:
                self.lam_ = float(self.lam)
            Tcal = tippett(loo_pvalues(self.cal_s), pa_cal, self.lam_)
        elif self.use_sup:
            self.lam_ = 0.0
            Tcal = loo_pvalues(self.cal_s)
        else:
            self.lam_ = 1.0
            Tcal = loo_pvalues(self.cal_a)
        self.t_ = calibrated_threshold(Tcal, self.alpha)
        self.fit_time = time.time() - t0
        return self

    def threshold_for(self, lam, alpha=None):
        """Jointly calibrated threshold for an arbitrary budget split (used for ablations)."""
        alpha = self.alpha if alpha is None else alpha
        T = tippett(loo_pvalues(self.cal_s), loo_pvalues(self.cal_a), lam)
        return calibrated_threshold(T, alpha)

    def raw_components(self, X):
        """Rule matches and both p-values for every flow (no fast path); for analysis/ablation."""
        r = self.rl.match(X) if self.use_rules else np.zeros(len(X), bool)
        rs = self.sup.predict_proba(X)[:, 1]
        ra = self.ano.score(X)
        return r, conformal_pvalues(self.cal_s, rs), conformal_pvalues(self.cal_a, ra), rs, ra

    def set_alpha(self, alpha):
        """Re-calibrate the operating point for a new false-alarm budget (no retraining)."""
        self.alpha = alpha
        ps = loo_pvalues(self.cal_s) if self.use_sup else np.ones(1)
        pa = loo_pvalues(self.cal_a) if self.use_ano else np.ones(1)
        if self.use_sup and self.use_ano:
            T = tippett(ps, pa, self.lam_)
        else:
            T = ps if self.use_sup else pa
        self.t_ = calibrated_threshold(T, alpha)

    def predict_score(self, X, fast_path=True):
        n = len(X)
        r = self.rl.match(X) if self.use_rules else np.zeros(n, bool)
        rest = ~r if fast_path else np.ones(n, bool)
        ps, pa = np.ones(n), np.ones(n)
        raw_s, raw_a = np.zeros(n), np.zeros(n)
        if rest.any():
            Xr = X[rest]
            if self.use_sup:
                raw_s[rest] = self.sup.predict_proba(Xr)[:, 1]
                ps[rest] = conformal_pvalues(self.cal_s, raw_s[rest])
            if self.use_ano:
                raw_a[rest] = self.ano.score(Xr)
                pa[rest] = conformal_pvalues(self.cal_a, raw_a[rest])
        if self.use_sup and self.use_ano:
            T = tippett(ps, pa, self.lam_)
        else:
            T = ps if self.use_sup else pa
        pred = r | (T <= self.t_)
        # continuous score for threshold-free metrics; ties in the saturated tail broken by raw scores
        z = -np.log(T) + 1e-3 * (raw_s / (1 + raw_s)) * (self.lam_ < 1) \
            + 1e-3 * (raw_a / (1 + raw_a)) * (self.lam_ > 0)
        z[r] = 1e6
        return pred.astype(int), z, dict(rule=r, ps=ps, pa=pa, T=T)


# ----------------------------------------------------------------------------- baselines
def make_baseline(name, seed):
    if name == "LR":
        return LogisticRegression(max_iter=2000, C=1.0)
    if name == "DT":
        return DecisionTreeClassifier(random_state=seed, min_samples_leaf=2)
    if name == "RF":
        return RandomForestClassifier(n_estimators=150, n_jobs=NJ, random_state=seed,
                                      min_samples_leaf=1)
    if name == "SVM-RBF":
        return SVC(kernel="rbf", C=10, gamma="scale", random_state=seed)
    if name == "MLP":
        return MLPClassifier(hidden_layer_sizes=(128, 64), batch_size=256, max_iter=60,
                             early_stopping=True, n_iter_no_change=5, random_state=seed)
    if name == "XGBoost":
        return xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
                                 subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                                 n_jobs=NJ, random_state=seed, verbosity=0)
    if name == "LightGBM":
        return make_lgbm(seed)
    raise KeyError(name)


def sup_score(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


class IForest:
    def __init__(self, seed=0):
        self.m = IsolationForest(n_estimators=200, max_samples=256, random_state=seed, n_jobs=NJ)

    def fit(self, Xb):
        self.m.fit(Xb)
        return self

    def score(self, X):
        return -self.m.score_samples(X)


# ============================================================================= revised method
class SplitCHIDS:
    """CH-IDS with split calibration (revised method).

    D_fit   trains the rule tree, LightGBM and the autoencoder.
    D_cal1  (i) screens the rules (no benign hit), (ii) is the reference set for the per-tier conformal
            p-values, (iii) is used by the inner leave-one-family-out search for lambda.
    D_cal2  is touched only once: the complete decision statistic G(x) (rules + fusion) is evaluated on its
            benign flows, and an alert is raised iff the conformal p-value of G(x) w.r.t. these flows <= alpha.
    Because G is a fixed function of x given D_fit and D_cal1, the split-conformal argument applies to the whole
    decision procedure (Theorem 1 in the paper)."""

    RULE_G = -1.0

    def __init__(self, alpha=0.01, lam="auto", seed=0, ano=None, inner_trees=100):
        self.alpha, self.lam, self.seed, self.ano, self.inner_trees = alpha, lam, seed, ano, inner_trees

    # ---------------------------------------------------------------- training
    def fit(self, Xf, yf, ff, X1, y1, f1, X2, y2):
        t0 = time.time()
        self.rl = RuleLayer(seed=self.seed).fit(Xf, yf, X1, y1)
        self.sup = make_lgbm(self.seed).fit(Xf, yf)
        if self.ano is None:
            self.ano = AutoEncoder(seed=self.seed).fit(Xf[yf == 0])
        b1 = X1[y1 == 0]
        self.ref_s = np.sort(self.sup.predict_proba(b1)[:, 1])
        self.ref_a = np.sort(self.ano.score(b1))
        if self.lam == "auto":
            helper = CHIDS(alpha=self.alpha, seed=self.seed, ano=self.ano, inner_trees=self.inner_trees)
            self.lam_ = helper._select_lambda(Xf, yf, ff, X1, y1, f1, self.ref_a, loo_pvalues(self.ref_a))
            self.inner_dr = helper.inner_dr
        else:
            self.lam_ = float(self.lam)
            self.inner_dr = {}
        # final calibration set: benign flows of D_cal2 only
        self.b2 = X2[y2 == 0]
        self.comp2 = self.components(self.b2)
        self.fit_time = time.time() - t0
        return self

    # ---------------------------------------------------------------- statistics
    def components(self, X):
        r = self.rl.match(X)
        rs = self.sup.predict_proba(X)[:, 1]
        ra = self.ano.score(X)
        ps = conformal_pvalues(self.ref_s, rs)
        pa = conformal_pvalues(self.ref_a, ra)
        u = 0.5 * rs + 0.5 * ra / (1.0 + ra)          # in [0, 1): tie-breaker only
        return dict(r=r, rs=rs, ra=ra, ps=ps, pa=pa, u=u)

    # evidence used to order flows inside a block of equal fused values (tie-breaker); each variant uses only the
    # tiers it contains, so that an ablated tier has no influence at all on the decision
    @staticmethod
    def tiebreak(c, variant="full"):
        if variant == "no_ano":
            return c["rs"]
        if variant == "no_sup":
            return c["ra"] / (1.0 + c["ra"])
        if variant == "rules_only":
            return np.zeros(len(c["r"]))
        return c["u"]

    def G(self, c, variant="full", lam=None):
        """System statistic (smaller = more suspicious).
        full / no_rules : T_lambda(p_s, p_a) - eps*u          (lambda = lambda* unless given)
        no_ano          : p_s - eps*s_s                        (supervised tier only, i.e. lambda = 0)
        no_sup          : p_a - eps*a/(1+a)                    (anomaly tier only, i.e. lambda = 1)
        rules_only      : 1 for unmatched flows
        Rule matches (all variants except no_rules): RULE_G - u_variant, i.e. below every unmatched flow."""
        lam = self.lam_ if lam is None else lam
        u = self.tiebreak(c, variant)
        if variant == "rules_only":
            g = np.ones(len(c["r"]))
        elif variant == "no_ano":
            g = c["ps"] - 1e-9 * u
        elif variant == "no_sup":
            g = c["pa"] - 1e-9 * u
        else:
            g = tippett(c["ps"], c["pa"], lam) - 1e-9 * u
        if variant != "no_rules":
            g = np.where(c["r"], self.RULE_G - u, g)
        return g

    def threshold(self, Gcal, alpha):
        """Order-statistic form of the decision P(x) <= alpha:
        k = floor(alpha (n+1)); alert iff k >= 1 and G(x) < G_(k), the k-th smallest calibration value.
        Returns (k, G_(k)) with G_(k) = -inf when k = 0 (no alert is ever raised)."""
        srt = np.sort(Gcal); n = len(srt)
        k = int(np.floor(alpha * (n + 1) + 1e-12))
        return k, (srt[k - 1] if k >= 1 else -np.inf)

    def predict_fast(self, X, alpha, Gcal, variant="full", lam=None):
        """Exact equivalent of decide() for the full statistic, evaluating the learned tiers only when needed.
        h = number of benign calibration flows that match a rule. If h < k, every rule match is below G_(k) and is
        flagged without the learned tiers; otherwise only rule matches can be flagged and unmatched flows are
        benign without the learned tiers."""
        k, gk = self.threshold(Gcal, alpha)
        out = np.zeros(len(X), bool)
        if k == 0:
            return out
        r = self.rl.match(X)
        h = int((np.asarray(Gcal) <= self.RULE_G).sum())
        if h < k:
            out[r] = True
            rest = ~r
            if rest.any():
                c = self.components(X[rest])
                out[rest] = self.G(c, variant, lam) < gk
        elif r.any():
            c = self.components(X[r])
            out[r] = self.G(c, variant, lam) < gk
        return out

    @staticmethod
    def system_pvalues(Gcal, Gx):
        """Conformal p-value of the complete decision statistic (smaller G = more suspicious)."""
        srt = np.sort(Gcal)
        le = np.searchsorted(srt, Gx, side="right")       # number of calibration values <= G(x)
        return (1.0 + le) / (len(srt) + 1.0)

    def decide(self, c, alpha=None, variant="full", lam=None, Gcal=None):
        alpha = self.alpha if alpha is None else alpha
        Gx = self.G(c, variant, lam)
        if Gcal is None:
            Gcal = self.G(self.comp2, variant, lam)
        P = self.system_pvalues(Gcal, Gx)
        return self.alerts(Gcal, Gx, alpha), -Gx, P

    @staticmethod
    def alerts(Gcal, Gx, alpha):
        """P(x) <= alpha evaluated in integer arithmetic: 1 + #{G_j <= G(x)} <= k = floor(alpha (n+1)),
        which is the same as G(x) < G_(k) (and never true when k = 0)."""
        srt = np.sort(Gcal); n = len(srt)
        k = int(np.floor(alpha * (n + 1) + 1e-12))
        le = np.searchsorted(srt, Gx, side="right")
        return (1 + le) <= k

    def rule_benign_hits(self):
        return int(self.comp2["r"].sum()), int(len(self.comp2["r"]))
