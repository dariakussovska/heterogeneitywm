pip install --upgrade scipy statsmodels

"""
Trial-level behavioral link analyses for response to R1.1

Goal: to test whether trial-by-trial neural state (burst rate,
decoder confidence) predicts trial-by-trial behavior (accuracy, RT)

Two analyses:
  Test 1 (accuracy): does neural state predict whether trial was correct?
  Test 2 (RT, correct trials only): does neural state predict RT?

Outputs:
  - coefficients with CIs and p-values for each predictor
  - per-subject diagnostics/info
  - power check to interpret null results (if null)

"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


# --- Expected trial table schema --------------------------------------------
#
# Build a DataFrame `trials` with one row per trial. Required columns:
#
#   subject              str/int     patient ID
#   trial_idx            int         trial number within subject
#   correct              {0, 1}      did they answer correctly?
#   rt_s                 float       reaction time in secs (NaN if no resp)
#   load                 {1, 2, 3}   memory load
#   decoder_confidence   float       mean Bayesian-decoder margin during
#                                    this trial's maintenance (use the
#                                    existing cross-epoch decoder output)
#

REQUIRED_COLS = [
    "subject", "trial_idx", "correct", "rt_s", "load",
    "decoder_confidence",
]


def validate(trials: pd.DataFrame) -> None:
    """Sanity-check input to see if anything is missing/off"""
    missing = [c for c in REQUIRED_COLS if c not in trials.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if not trials["correct"].isin([0, 1]).all():
        raise ValueError("`correct` should be 0/1")

    if not set(trials["load"].unique()).issubset({1, 2, 3}):
        raise ValueError("`load` should be in {1, 2, 3}")

    n_subj = trials["subject"].nunique()
    if n_subj < 2:
        raise ValueError(
            f"Only {n_subj} subject — need at least 2 for clustering. "
            "Use within-subject aggregation + non-parametric test instead."
        )

    n_errors = int((trials["correct"] == 0).sum())
    if n_errors < 30:
        # Not fatal, but worth knowing
        print(f"WARNING: only {n_errors} error trials. Test 1 likely "
              "underpowered. Run power_diagnostic() before interpreting nulls.")


def zscore(s: pd.Series) -> pd.Series:
    """Standard z-score -> returns zeros if SD is 0 or NaN."""
    s = s.astype(float)
    sd = s.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return s * 0.0
    return (s - s.mean()) / sd


def prepare(trials: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score the predictors so coefficients are interpretable per-SD.

    Also keeps the load on its own z-scaled axis even though it's a 1/2/3
    integer. This makes the load coefficient comparable to the others
    """
    df = trials.copy()
    df["decoder_conf_z"] = zscore(df["decoder_confidence"])
    df["load_z"] = zscore(df["load"].astype(float))
    return df


# --- Test 1: does neural state predict perf accuracy? -------------------------

def test1_accuracy_gee(trials: pd.DataFrame) -> Dict:
    """
    Population-averaged logistic regression via GEE, clustered by subject.

    Why GEE and not a mixed-effects logistic regression:
      - For binary outcomes with subject clustering, the two principled
        options are GEE (population-averaged) and GLMM (subject-specific).
      - GLMM (e.g., statsmodels' BinomialBayesMixedGLM) often doesn't work to
        converge cleanly on small datasets (especially with low error rates)
      - GEE converges robustly and gives the population-level effect, which
        is what we want for "does X predict Y across this cohort."
    """
    df = prepare(trials).dropna(
        subset=["decoder_conf_z", "load_z", "correct", "subject"]
    )

    predictors = ["decoder_conf_z", "load_z"]
    formula = "correct ~ " + " + ".join(predictors)

    fam = sm.families.Binomial()
    cov = sm.cov_struct.Exchangeable()

    gee = smf.gee(
        formula=formula,
        groups="subject",
        data=df,
        cov_struct=cov,
        family=fam,
    ).fit()

    # Build results dict.
    out = {
        "n_trials": int(len(df)),
        "n_subjects": int(df["subject"].nunique()),
        "n_errors": int((df["correct"] == 0).sum()),
        "formula": formula,
        "coef": {p: float(gee.params[p]) for p in predictors},
        "se": {p: float(gee.bse[p]) for p in predictors},
        "pvalue": {p: float(gee.pvalues[p]) for p in predictors},
        # CI on the log-odds scale
        "ci_log_odds": {
            p: (float(gee.conf_int().loc[p, 0]),
                float(gee.conf_int().loc[p, 1]))
            for p in predictors
        },
        # CI on the odds-ratio scale (exp of log-odds)
        "odds_ratio": {p: float(np.exp(gee.params[p])) for p in predictors},
        "odds_ratio_ci": {
            p: (float(np.exp(gee.conf_int().loc[p, 0])),
                float(np.exp(gee.conf_int().loc[p, 1])))
            for p in predictors
        },
        "intercept": float(gee.params["Intercept"]),
        "summary_text": gee.summary().as_text(),  # save the full output
    }
    return out


def test1_per_subject_diagnostic(trials: pd.DataFrame) -> pd.DataFrame:
    """
    Quick sanity check: per-subject mean decoder_confidence
    on correct vs error trials. If the GEE shows an effect, per-subject
    deltas should mostly go in the same direction. If they don't, the
    population-average effect might be driven by a few subjects.

    *Subjects with fewer than 3 errors or 3 correct trials are excluded
    because their per-subject means aren't meaningful.
    """
    rows = []
    for s, g in trials.groupby("subject"):
        n_c = (g["correct"] == 1).sum()
        n_e = (g["correct"] == 0).sum()
        if n_c < 3 or n_e < 3:
            continue
        c = g[g["correct"] == 1]
        e = g[g["correct"] == 0]
        rows.append({
            "subject": s,
            "n_correct": int(n_c),
            "n_error": int(n_e),
            "decoder_conf_correct": c["decoder_confidence"].mean(),
            "decoder_conf_error": e["decoder_confidence"].mean(),
            "decoder_conf_delta": c["decoder_confidence"].mean() - e["decoder_confidence"].mean(),
        })
    return pd.DataFrame(rows)


def test1_within_subject_wilcoxon(per_subject: pd.DataFrame) -> Dict:
    """
    Non-parametric backup: Wilcoxon signed-rank on per-subject deltas.
    We can use this if GEE convergence fails or as a robustness check.
    Tests whether decoder_conf differs between correct and
    error trials within subjects, treating each subject as one observation.
    """
    out = {}
    predictor = "decoder_conf_delta"
    vals = per_subject[predictor].dropna().values
    if len(vals) < 5:
        out[predictor] = {"note": "too few subjects with enough trials"}
    else:
        res = stats.wilcoxon(vals)
        out[predictor] = {
            "n_subjects": int(len(vals)),
            "median_delta": float(np.median(vals)),
            "iqr": [float(np.percentile(vals, 25)),
                    float(np.percentile(vals, 75))],
            "statistic": float(res.statistic),
            "p_value": float(res.pvalue),
        }
    return out


# --- Test 2: does neural state predict RT? ----------------------------------

def test2_rt_mixedeffects(trials: pd.DataFrame) -> Dict:
    """
    Linear mixed-effects: log(rt) ~ predictors + (1|subject)

    Restricted to correct trials only — RT on incorrect trials is
    contaminated by guessing/randomness.
    Log-transform because raw RT is right-skewed

    Subject random intercepts handle baseline RT differences across patients.

    NOTE: smf.mixedlm sometimes fails to converge. If so, the alternative is
    OLS with subject as a fixed-effects factor (this is a bit more reliable).
    """
    df = prepare(trials).copy()
    # Only correct trials with valid RT
    df = df.loc[
        (df["correct"] == 1) &
        df["rt_s"].notna() &
        (df["rt_s"] > 0)
    ]
    df = df.dropna(subset=["decoder_conf_z", "load_z"])

    if df.empty:
        return {"error": "no usable trials available after filtering"}

    df["log_rt"] = np.log(df["rt_s"].astype(float))

    predictors = ["decoder_conf_z", "load_z"]
    formula = "log_rt ~ " + " + ".join(predictors)

    md = smf.mixedlm(formula, df, groups=df["subject"])
    try:
        mdf = md.fit(reml=True, method="lbfgs")
        converged = bool(mdf.converged)
    except Exception as e:
        return {"error": f"mixed model failed: {e}"}

    out = {
        "n_trials": int(len(df)),
        "n_subjects": int(df["subject"].nunique()),
        "converged": converged,
        "formula": formula,
        "coef": {p: float(mdf.params[p]) for p in predictors},
        "se": {p: float(mdf.bse[p]) for p in predictors},
        "pvalue": {p: float(mdf.pvalues[p]) for p in predictors},
        "ci_wald": {
            p: (float(mdf.conf_int().loc[p, 0]),
                float(mdf.conf_int().loc[p, 1]))
            for p in predictors
        },
        "summary_text": mdf.summary().as_text(),
    }
    return out


def test2_rt_bootstrap_ci(
    trials: pd.DataFrame,
    n_boots: int = 500,
    seed: int = 0,
) -> Dict:
    """
    Cluster-bootstrap by subject to get more robust CIs on the RT
    coefficients. Resample subjects with replacement (not individual
    trials - that would understate uncertainty due to within-subject
    correlation).
    """
    rng = np.random.default_rng(seed)
    df = prepare(trials).copy()
    df = df.loc[
        (df["correct"] == 1) &
        df["rt_s"].notna() &
        (df["rt_s"] > 0)
    ]
    df = df.dropna(subset=["decoder_conf_z", "load_z"])
    if df.empty:
        return {"error": "no usable trials"}
    df["log_rt"] = np.log(df["rt_s"].astype(float))

    subjects = df["subject"].unique()
    predictors = ["decoder_conf_z", "load_z"]
    boot_coefs = {p: [] for p in predictors}

    formula = "log_rt ~ " + " + ".join(predictors)

    for b in range(n_boots):
        # Resample subjects with replacement
        boot_subjects = rng.choice(subjects, size=len(subjects), replace=True)
        # Build bootstrap dataset - each resampled subject contributes all
        # its trials, so within-subject correlation is preserved
        parts = [df.loc[df["subject"] == s] for s in boot_subjects]
        bdf = pd.concat(parts, ignore_index=True)
        try:
            # Use ML (reml=False) here - faster, and CI from bootstrap so
            # don't need REML's small-sample correction
            bm = smf.mixedlm(formula, bdf, groups=bdf["subject"]).fit(
                reml=False, method="lbfgs",
            )
            for p in predictors:
                boot_coefs[p].append(float(bm.params.get(p, np.nan)))
        except Exception:
            for p in predictors:
                boot_coefs[p].append(np.nan)

    # 95% CI from bootstrap distribution
    out = {}
    for p, vals in boot_coefs.items():
        arr = np.array(vals, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) < 50:
            out[p] = None  # too few converged bootstrap iterations
        else:
            out[p] = (
                float(np.percentile(arr, 2.5)),
                float(np.percentile(arr, 97.5)),
            )
    return out


# --- Power diagnostic to properly interpret nulls properly -------------------
def power_diagnostic(
    trials: pd.DataFrame,
    true_or: float = 1.5,
    n_sims: int = 200,
    seed: int = 0,
) -> Dict:
    """
    Run before interpreting any null result from Test 1.

    Here, we'll simulates accuracy under a known true odds ratio (per SD of decoder_confidence)
    using our actual subject structure and predictor distribution, then
    runs the same GEE pipeline. Reports how often the true effect would
    have been detected.

    If power < 0.5: a true effect of that size is more likely missed than
    detected. A null result is genuinely inconclusive (need to report this)

    Reference effect sizes for context:
      OR = 1.2 = small effect
      OR = 1.5 = moderate effect
      OR = 2.0 = large effect
    """
    rng = np.random.default_rng(seed)
    df = prepare(trials).dropna(
        subset=["decoder_conf_z", "load_z", "subject"]
    ).reset_index(drop=True)

    base_rate = df["correct"].mean()
    if not (0 < base_rate < 1):
        return {"error": "base rate is 0 or 1, can't simulate"}
    base_logodds = float(np.log(base_rate / (1 - base_rate)))

    def one_sim(or_val: float) -> bool:
        beta = float(np.log(or_val))
        # Construct log-odds per trial given the assumed effect
        log_odds = base_logodds + beta * df["decoder_conf_z"].values
        p = 1.0 / (1.0 + np.exp(-log_odds))
        sim_correct = (rng.uniform(size=len(p)) < p).astype(int)
        sim_df = df.copy()
        sim_df["correct"] = sim_correct
        try:
            fit = test1_accuracy_gee(sim_df)
            return fit["pvalue"]["decoder_conf_z"] < 0.05
        except Exception:
            return False

    n_pos_at_effect = sum(one_sim(true_or) for _ in range(n_sims))
    n_pos_at_null = sum(one_sim(1.0) for _ in range(n_sims))

    return {
        "true_or_tested": true_or,
        "power_at_effect": n_pos_at_effect / n_sims,
        "typeI_at_null": n_pos_at_null / n_sims,
        "n_sims": n_sims,
        "interpretation": (
            "Power < 0.5: null result is inconclusive - could just be "
            "underpowered. Report nulls with explicit power info. "
            "Power > 0.8: null result more informative. "
            "Type-I rate should be close to 0.05 if pipeline is calibrated."
        ),
    }


# --- Top-level driver ------------------------------------------------------

def run_all(trials: pd.DataFrame, do_power: bool = True) -> Dict:
    """Run everything and return a single results dict"""
    validate(trials)

    out = {}

    # Test 1: accuracy
    out["test1_accuracy"] = test1_accuracy_gee(trials)

    # Diagnostic: per-subject breakdown
    per_subj = test1_per_subject_diagnostic(trials)
    out["test1_per_subject"] = per_subj.to_dict("records")
    out["test1_within_subject_wilcoxon"] = test1_within_subject_wilcoxon(per_subj)

    # Test 2: RT
    out["test2_rt"] = test2_rt_mixedeffects(trials)
    out["test2_rt_bootstrap_ci"] = test2_rt_bootstrap_ci(trials, n_boots=500)

    # Power
    if do_power:
        out["power"] = power_diagnostic(trials, true_or=1.5)

    return out


# --- Formatting output -------------------------------------------------------

def format_for_response_letter(results: Dict) -> str:
    """
    Quick formatted output
    """
    acc = results["test1_accuracy"]
    rt = results["test2_rt"]
    power = results.get("power", {})

    def sig_p(p):
        return "P < 0.001" if p < 0.001 else f"P = {p:.3f}"

    lines = []
    lines.append(f"Test 1 (accuracy GEE, n={acc['n_trials']} trials, "
                 f"n={acc['n_subjects']} subjects, n={acc['n_errors']} errors):")
    for pred in ["decoder_conf_z"]:
        or_val = acc["odds_ratio"][pred]
        ci_lo, ci_hi = acc["odds_ratio_ci"][pred]
        p = acc["pvalue"][pred]
        lines.append(f"  {pred}: OR = {or_val:.2f} per SD, "
                     f"95% CI [{ci_lo:.2f}, {ci_hi:.2f}], {sig_p(p)}")
    lines.append("")
    lines.append(f"Test 2 (RT mixed-effects, n={rt['n_trials']} correct trials, "
                 f"n={rt['n_subjects']} subjects):")
    for pred in ["decoder_conf_z"]:
        b = rt["coef"][pred]
        ci_lo, ci_hi = rt["ci_wald"][pred]
        p = rt["pvalue"][pred]
        lines.append(f"  {pred}: β = {b:+.3f} log-s per SD, "
                     f"95% CI [{ci_lo:+.3f}, {ci_hi:+.3f}], {sig_p(p)}")
    if power:
        lines.append("")
        lines.append(f"Power for Test 1 at OR=1.5: {power.get('power_at_effect', 0):.0%}")
        lines.append(f"Type-I at null: {power.get('typeI_at_null', 0):.0%}")
    return "\n".join(lines)


# --- Demo --------------------------------------------------------------------

if __name__ == "__main__":
    # Load real data from Excel file
    trials = pd.read_excel("GEE_beh_analysis.xlsx")
    
    print(f"Loaded {len(trials)} trials from {trials['subject'].nunique()} subjects")
    print(f"Error rate: {(trials['correct'] == 0).mean():.1%}")
    print(f"Columns: {list(trials.columns)}")
    print()
    
    results = run_all(trials, do_power=True)
    print(format_for_response_letter(results))
    print("\n" + "="*80)
    print("Full Test 1 Summary:")
    print("="*80)
    print(results["test1_accuracy"]["summary_text"])
    print("\n" + "="*80)
    print("Full Test 2 Summary:")
    print("="*80)
    print(results["test2_rt"]["summary_text"])
