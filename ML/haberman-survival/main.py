"""Haberman's Survival: do positive axillary nodes predict 5-year survival after surgery?

Story flow: hypotheses -> data inspection -> statistical tests -> baseline -> takeaways.

Optional headless runner. The narrated lesson is haberman_survival.ipynb in this folder.

Run:
    uv run python ML/haberman-survival/main.py [--publish-docs]

Outputs (``outputs/``): four annotated PNG figures and ``metrics.json``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data"
COLUMNS = ["age", "op_year", "nodes", "status"]
EXPECTED_ROWS = 306
SEED = 42
ALPHA = 0.05
N_BOOTSTRAP = 10_000
CV_SPLITS, CV_REPEATS = 5, 10
NODE_BIN_EDGES = [-1, 0, 4, 9, np.inf]
NODE_BIN_LABELS = ["0", "1-4", "5-9", "10+"]
COLORS = {"Survived 5y+": "#2a9d8f", "Died <5y": "#e76f51"}
HERE = Path(__file__).resolve().parent
DOCS_ASSETS = HERE.parents[1] / "docs" / "assets" / "haberman-survival"


def load_data(data_dir: Path) -> pd.DataFrame:
    """Loads the Haberman dataset, downloading it from UCI on first use.

    The raw file has no header row. Reading it with a default header silently
    swallows the first patient, so the column names are supplied explicitly.

    Args:
        data_dir: Directory used to cache the raw file.

    Returns:
        DataFrame with ``age``, ``op_year`` (full year), ``nodes`` and a binary
        ``survived`` target (1 = survived 5 years or longer).

    Raises:
        ValueError: If the file does not have the expected shape or has nulls.
    """
    path = data_dir / "haberman.csv"
    if not path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, path)  # noqa: S310 - fixed https URL
    df = pd.read_csv(path, header=None, names=COLUMNS)
    if len(df) != EXPECTED_ROWS or df.isna().any().any():
        raise ValueError(f"Unexpected data: {df.shape}, nulls={int(df.isna().sum().sum())}")
    df["survived"] = (df["status"] == 1).astype(int)
    df["op_year"] = df["op_year"] + 1900
    df["outcome"] = df["survived"].map({1: "Survived 5y+", 0: "Died <5y"})
    return df.drop(columns="status")


def mann_whitney(survived: np.ndarray, died: np.ndarray) -> dict[str, float]:
    """Runs a two-sided Mann-Whitney U test with Cliff's delta as effect size.

    Args:
        survived: Feature values for patients who survived 5+ years.
        died: Feature values for patients who died within 5 years.

    Returns:
        U statistic, p-value and Cliff's delta (positive = survivors tend higher).
    """
    res = stats.mannwhitneyu(survived, died, alternative="two-sided")
    delta = 2 * res.statistic / (len(survived) * len(died)) - 1
    return {"U": float(res.statistic), "p": float(res.pvalue), "cliffs_delta": float(delta)}


def bootstrap_median_diff(
    survived: np.ndarray, died: np.ndarray, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Bootstraps a 95% CI for the difference in medians (survived - died).

    Args:
        survived: Feature values for survivors.
        died: Feature values for non-survivors.
        rng: Random generator for reproducibility.

    Returns:
        Observed difference, CI lower bound, CI upper bound.
    """
    s = rng.choice(survived, (N_BOOTSTRAP, len(survived)))
    d = rng.choice(died, (N_BOOTSTRAP, len(died)))
    diffs = np.median(s, axis=1) - np.median(d, axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(np.median(survived) - np.median(died)), float(lo), float(hi)


def feature_tests(df: pd.DataFrame, rng: np.random.Generator) -> dict[str, dict[str, float]]:
    """Tests every feature against survival and corrects for multiple testing.

    H0: the feature has the same distribution in survivors and non-survivors.
    Ha: the distributions differ. Mann-Whitney is used because ``nodes`` is
    heavily right-skewed; p-values get Holm adjustment across the three tests.

    Args:
        df: Cleaned dataset from :func:`load_data`.
        rng: Random generator for the bootstrap.

    Returns:
        Per-feature dict of test statistics, effect size, median-diff CI and
        Holm-adjusted p-value.
    """
    results: dict[str, dict[str, float]] = {}
    for feature in ["age", "op_year", "nodes"]:
        surv = df.loc[df["survived"] == 1, feature].to_numpy()
        died = df.loc[df["survived"] == 0, feature].to_numpy()
        res = mann_whitney(surv, died)
        diff, lo, hi = bootstrap_median_diff(surv, died, rng)
        res.update(median_diff=diff, median_diff_ci_low=lo, median_diff_ci_high=hi)
        results[feature] = res
    adjusted = multipletests([r["p"] for r in results.values()], method="holm")[1]
    for res, p_adj in zip(results.values(), adjusted, strict=True):
        res["p_holm"] = float(p_adj)
    return results


def node_bin_table(df: pd.DataFrame) -> pd.DataFrame:
    """Computes survival rate with Wilson 95% CIs per node-count bin.

    Args:
        df: Cleaned dataset.

    Returns:
        One row per bin with n, survivors, rate and CI bounds.
    """
    bins = pd.cut(df["nodes"], NODE_BIN_EDGES, labels=NODE_BIN_LABELS)
    grouped = df.groupby(bins, observed=True)["survived"].agg(["size", "sum"])
    grouped.columns = ["n", "survivors"]
    lo, hi = proportion_confint(grouped["survivors"], grouped["n"], method="wilson")
    grouped["rate"] = grouped["survivors"] / grouped["n"]
    grouped["ci_low"], grouped["ci_high"] = lo, hi
    return grouped


def chi_square_nodes(df: pd.DataFrame) -> dict[str, float]:
    """Runs a chi-square test of independence between node bin and survival.

    H0: survival is independent of node bin. Ha: survival depends on node bin.

    Args:
        df: Cleaned dataset.

    Returns:
        Chi-square statistic, dof, p-value, Cramer's V and the smallest expected
        cell count (should be >= 5 for the approximation to be trusted).
    """
    bins = pd.cut(df["nodes"], NODE_BIN_EDGES, labels=NODE_BIN_LABELS)
    table = pd.crosstab(bins, df["survived"])
    chi2, p, dof, expected = stats.chi2_contingency(table)
    v = np.sqrt(chi2 / (table.to_numpy().sum() * (min(table.shape) - 1)))
    return {
        "chi2": float(chi2),
        "dof": int(dof),
        "p": float(p),
        "cramers_v": float(v),
        "min_expected_count": float(expected.min()),
    }


def _log_nodes() -> FunctionTransformer:
    return FunctionTransformer(np.log1p, feature_names_out="one-to-one")


def build_models() -> dict[str, Pipeline]:
    """Builds the baseline ladder. All preprocessing lives inside the pipeline.

    Keeping scaling and log transforms in the pipeline means they are fitted on
    training folds only, so no test-fold information leaks into training.

    Returns:
        Model name -> sklearn Pipeline.
    """
    all_features = ColumnTransformer(
        [
            ("nodes", _log_nodes(), ["nodes"]),
            ("rest", StandardScaler(), ["age", "op_year"]),
        ]
    )
    nodes_only = ColumnTransformer([("nodes", _log_nodes(), ["nodes"])])
    return {
        "Majority class (no skill)": Pipeline([("model", DummyClassifier(strategy="prior"))]),
        "Logistic: nodes only": Pipeline(
            [("prep", nodes_only), ("model", LogisticRegression(max_iter=1000))]
        ),
        "Logistic: nodes + age + year": Pipeline(
            [("prep", all_features), ("model", LogisticRegression(max_iter=1000))]
        ),
    }


def corrected_resampled_ci(
    scores: np.ndarray, n_train: int, n_test: int
) -> tuple[float, float, float]:
    """Gives a 95% CI for the mean CV score using the Nadeau-Bengio correction.

    Plain CV folds share training data, so their scores are correlated and a
    naive standard error is too small. The correction inflates it.

    Args:
        scores: Per-fold scores.
        n_train: Training rows per fold.
        n_test: Test rows per fold.

    Returns:
        Mean, CI lower bound, CI upper bound.
    """
    k = len(scores)
    se = np.sqrt((1 / k + n_test / n_train) * np.var(scores, ddof=1))
    half = stats.t.ppf(0.975, k - 1) * se
    return float(scores.mean()), float(scores.mean() - half), float(scores.mean() + half)


def corrected_paired_p(diff: np.ndarray, n_train: int, n_test: int) -> float:
    """Tests whether two models differ on identical folds (corrected resampled t-test).

    Args:
        diff: Per-fold score differences between two models.
        n_train: Training rows per fold.
        n_test: Test rows per fold.

    Returns:
        Two-sided p-value.
    """
    k = len(diff)
    se = np.sqrt((1 / k + n_test / n_train) * np.var(diff, ddof=1))
    return float(2 * stats.t.sf(abs(diff.mean() / se), k - 1))


def evaluate_models(df: pd.DataFrame) -> dict[str, Any]:
    """Cross-validates the baseline ladder with repeated stratified K-fold.

    Args:
        df: Cleaned dataset.

    Returns:
        ROC-AUC and accuracy with corrected CIs per model, plus paired tests.
    """
    x, y = df[["age", "op_year", "nodes"]], df["survived"]
    cv = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=SEED)
    n_test = len(df) // CV_SPLITS
    n_train = len(df) - n_test
    out: dict[str, Any] = {"models": {}, "scores": {}}
    for name, model in build_models().items():
        auc = cross_val_score(model, x, y, cv=cv, scoring="roc_auc")
        acc = cross_val_score(model, x, y, cv=cv, scoring="accuracy")
        out["scores"][name] = auc
        mean, lo, hi = corrected_resampled_ci(auc, n_train, n_test)
        a_mean, a_lo, a_hi = corrected_resampled_ci(acc, n_train, n_test)
        out["models"][name] = {
            "auc": mean,
            "auc_ci_low": lo,
            "auc_ci_high": hi,
            "accuracy": a_mean,
            "accuracy_ci_low": a_lo,
            "accuracy_ci_high": a_hi,
        }
    names = list(out["scores"])
    out["paired_p"] = {
        "nodes_only_vs_majority": corrected_paired_p(
            out["scores"][names[1]] - out["scores"][names[0]], n_train, n_test
        ),
        "all_features_vs_nodes_only": corrected_paired_p(
            out["scores"][names[2]] - out["scores"][names[1]], n_train, n_test
        ),
    }
    return out


def plot_survival_by_nodes(table: pd.DataFrame, overall: float, out: Path) -> None:
    """Saves survival rate per node bin with Wilson CIs.

    Args:
        table: Output of :func:`node_bin_table`.
        overall: Overall survival rate, drawn as a reference line.
        out: Destination PNG path.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    err = [table["rate"] - table["ci_low"], table["ci_high"] - table["rate"]]
    ax.bar(table.index.astype(str), table["rate"], yerr=err, capsize=6, color="#2a9d8f")
    ax.axhline(overall, ls="--", color="grey")
    ax.text(3.45, overall + 0.015, f"overall {overall:.0%}", ha="right", color="grey")
    for i, n in enumerate(table["n"]):
        ax.text(i, 0.03, f"n={n}", ha="center", color="white", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("Positive axillary nodes found at surgery")
    ax.set_ylabel("Survived 5+ years (95% Wilson CI)")
    first, last = table["rate"].iloc[0], table["rate"].iloc[-1]
    ax.set_title(
        f"Survival drops from {first:.0%} with no positive nodes to {last:.0%} with 10+",
        loc="left",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_nodes_ecdf(df: pd.DataFrame, tests: dict[str, dict[str, float]], out: Path) -> None:
    """Saves the ECDF of node counts split by outcome.

    Args:
        df: Cleaned dataset.
        tests: Output of :func:`feature_tests`.
        out: Destination PNG path.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.ecdfplot(df, x="nodes", hue="outcome", palette=COLORS, linewidth=2.5, ax=ax)
    med = df.groupby("outcome")["nodes"].median()
    t = tests["nodes"]
    ax.text(
        0.97,
        0.05,
        f"median nodes: survived {med['Survived 5y+']:.0f} vs died {med['Died <5y']:.0f}\n"
        f"Mann-Whitney p(Holm) = {t['p_holm']:.1e}, Cliff's d = {t['cliffs_delta']:+.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        bbox={"boxstyle": "round", "fc": "white", "ec": "grey"},
    )
    ax.set_xlabel("Positive axillary nodes")
    ax.set_ylabel("Share of patients with at most this many nodes")
    ax.set_title(
        "Survivors pile up at 0 nodes; patients who died carry more", loc="left", fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_age_year(df: pd.DataFrame, tests: dict[str, dict[str, float]], out: Path) -> None:
    """Saves age and operation-year distributions by outcome.

    Args:
        df: Cleaned dataset.
        tests: Output of :func:`feature_tests`.
        out: Destination PNG path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, feature, label in zip(
        axes, ["age", "op_year"], ["Age at surgery", "Year of surgery"], strict=True
    ):
        sns.boxplot(
            df,
            x="outcome",
            y=feature,
            hue="outcome",
            palette=COLORS,
            fliersize=0,
            order=list(COLORS),
            ax=ax,
            legend=False,
        )
        sns.stripplot(
            df, x="outcome", y=feature, color="black", alpha=0.25, size=3, order=list(COLORS), ax=ax
        )
        t = tests[feature]
        ax.set_title(f"{label}\nHolm p = {t['p_holm']:.2f}, Cliff's d = {t['cliffs_delta']:+.2f}")
        ax.set_xlabel("")
        ax.set_ylabel(label)
    fig.suptitle(
        "Age and surgery year overlap heavily between outcomes",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_models(results: dict[str, Any], out: Path) -> None:
    """Saves the CV ROC-AUC comparison with corrected 95% CIs.

    Args:
        results: Output of :func:`evaluate_models`.
        out: Destination PNG path.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(results["models"])
    for i, name in enumerate(names):
        m = results["models"][name]
        ax.errorbar(
            m["auc"],
            i,
            xerr=[[m["auc"] - m["auc_ci_low"]], [m["auc_ci_high"] - m["auc"]]],
            fmt="o",
            color="#264653",
            capsize=5,
            markersize=8,
        )
        ax.text(m["auc_ci_high"] + 0.01, i, f"{m['auc']:.2f}", va="center")
    ax.axvline(0.5, ls="--", color="grey")
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlim(0.4, 0.9)
    ax.set_xlabel(f"ROC-AUC, {CV_SPLITS}x{CV_REPEATS} stratified CV (95% CI, Nadeau-Bengio)")
    p = results["paired_p"]["all_features_vs_nodes_only"]
    ax.set_title(
        f"Nodes carry the signal; age + year: no detectable gain (p = {p:.2f})",
        loc="right",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def run(data_dir: Path, out_dir: Path, publish_docs: bool) -> dict[str, Any]:
    """Runs the whole analysis and writes figures and metrics.

    Args:
        data_dir: Cache directory for the raw data.
        out_dir: Directory for figures and ``metrics.json``.
        publish_docs: Also copy figures to ``docs/assets/haberman-survival``.

    Returns:
        The metrics dict that was written to disk.
    """
    sns.set_theme(style="whitegrid", context="notebook")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    df = load_data(data_dir)
    overall = float(df["survived"].mean())
    tests = feature_tests(df, rng)
    table = node_bin_table(df)
    chi = chi_square_nodes(df)
    models = evaluate_models(df)

    plot_survival_by_nodes(table, overall, out_dir / "01_survival_by_nodes.png")
    plot_nodes_ecdf(df, tests, out_dir / "02_nodes_ecdf.png")
    plot_age_year(df, tests, out_dir / "03_age_year_overlap.png")
    plot_models(models, out_dir / "04_model_comparison.png")

    metrics = {
        "n_rows": len(df),
        "n_duplicate_rows": int(df.duplicated().sum()),
        "survival_rate": overall,
        "feature_tests": tests,
        "nodes_bins": table.reset_index().rename(columns={"nodes": "bin"}).to_dict("records"),
        "chi_square_nodes_bins": chi,
        "models": models["models"],
        "paired_p": models["paired_p"],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    if publish_docs:
        DOCS_ASSETS.mkdir(parents=True, exist_ok=True)
        for png in out_dir.glob("*.png"):
            shutil.copy(png, DOCS_ASSETS / png.name)
    return metrics


def main() -> None:
    """Parses CLI arguments and prints a short results summary."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    parser.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    parser.add_argument(
        "--publish-docs", action="store_true", help="copy figures to docs/assets/haberman-survival"
    )
    args = parser.parse_args()
    m = run(args.data_dir, args.out_dir, args.publish_docs)
    print(f"rows={m['n_rows']} duplicates={m['n_duplicate_rows']}", end=" ")
    print(f"survival={m['survival_rate']:.1%}")
    for name, t in m["feature_tests"].items():
        print(f"{name:8s} p_holm={t['p_holm']:.2g} cliffs_d={t['cliffs_delta']:+.2f}")
    print(f"chi2(nodes bin) p={m['chi_square_nodes_bins']['p']:.2g}")
    for name, v in m["models"].items():
        print(f"{name:30s} AUC {v['auc']:.3f} [{v['auc_ci_low']:.3f}, {v['auc_ci_high']:.3f}]")
    print("paired p:", m["paired_p"])


if __name__ == "__main__":
    main()
