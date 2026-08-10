
import re
import logging
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from statsmodels.stats.multitest import multipletests

# ============================================================================
# ======================  USER CONFIGURATION — EDIT THIS  ==================
# ============================================================================

# Output CSV produced by script 1 (slice-level raw features)
INPUT_CSV = r"F:\PROJECT_FORNIX\For_Paper\2D_Features_Final\raw_radiomics_features.csv"

# Where step-2 outputs are saved
OUTPUT_DIR = r"F:\PROJECT_FORNIX\For_Paper\2D_Features_Final\statistical_analysis"

GROUP_COL = "group"
SUBJECT_COL = "subject_id"
GROUPS = ["AD", "CN", "MCI"]

# How to collapse each subject's multiple slices into one row.
# "median" is robust to a single noisy/mis-segmented slice; "mean" is more
# sensitive to outlier slices.
AGGREGATION_METHOD = "median"   # "median" or "mean"

# Correlation pruning threshold applied to the SUBJECT-LEVEL feature matrix
# (this happens once, up front, on the full feature set -- distinct from the
# redundancy clustering done later on just the significant-feature subset)
CORR_THRESHOLD = 0.90

ALPHA = 0.05

# --- Effect-size gate ---
# A feature must reach this Kruskal-Wallis epsilon-squared to be considered
# "significant" for reporting purposes, REGARDLESS of how small its p-value
# is. Conventional epsilon-squared / eta-squared benchmarks (Cohen-style):
#   0.01 = small, 0.06 = medium, 0.14 = large
# We default to the medium threshold so "significant features" reported to
# you actually correspond to a meaningful, not just detectable, difference.
MIN_EFFECT_SIZE = 0.06

# --- Redundancy clustering ---
# After the significant + large-effect feature set is found, cluster them by
# |correlation| using average-linkage hierarchical clustering and cut the
# tree so that within-cluster |correlation| >= REDUNDANCY_CLUSTER_THRESHOLD.
# One representative (the highest-effect-size member) is kept per cluster.
REDUNDANCY_CLUSTER_THRESHOLD = 0.85

METADATA_COLS_PATTERNS = [
    r"^base_name$", r"^subject_id$", r"^group$", r"^n_blobs_original$",
    r"^slice_path$", r"^mask_path$",
    r"^diagnostics_",
]

# ============================================================================
# ==========================  END CONFIGURATION  ============================
# ============================================================================


def setup_logging(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "pipeline_step2.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler()],
    )
    return logging.getLogger("step2")


def identify_feature_columns(df):
    meta_regex = re.compile("|".join(METADATA_COLS_PATTERNS))
    candidate_cols = [c for c in df.columns if not meta_regex.match(c)]
    feature_cols = []
    for c in candidate_cols:
        coerced = pd.to_numeric(df[c], errors="coerce")
        if coerced.notna().mean() >= 0.9:
            feature_cols.append(c)
    return feature_cols


def clean_data(df, feature_cols, logger):
    df = df.copy()
    df = df[df[GROUP_COL].isin(GROUPS)].reset_index(drop=True)
    logger.info(f"After restricting to groups {GROUPS}: {len(df)} slice-level rows")

    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    na_frac = df[feature_cols].isna().mean()
    high_na_cols = na_frac[na_frac > 0.20].index.tolist()
    if high_na_cols:
        logger.info(f"Dropping {len(high_na_cols)} features with >20% missing values")
    feature_cols = [c for c in feature_cols if c not in high_na_cols]

    variances = df[feature_cols].var(skipna=True)
    zero_var_cols = variances[variances.fillna(0) == 0].index.tolist()
    if zero_var_cols:
        logger.info(f"Dropping {len(zero_var_cols)} zero-variance (constant) features")
    feature_cols = [c for c in feature_cols if c not in zero_var_cols]

    before = len(df)
    df_clean = df.dropna(subset=feature_cols).reset_index(drop=True)
    logger.info(f"Dropped {before - len(df_clean)} slice-level rows containing NaNs")

    logger.info(f"Cleaned SLICE-level dataset: {df_clean.shape[0]} rows x {len(feature_cols)} features")
    return df_clean, feature_cols


def aggregate_to_subject_level(df, feature_cols, subject_col, group_col, method, logger):
    """
    Collapse each subject's multiple slices into a single row. This is the
    critical fix: statistical tests below run on n=subjects (truly
    independent observations), not n=slices (pseudo-replicated, since
    adjacent slices from one subject are highly correlated with each other).
    """
    agg_func = "median" if method == "median" else "mean"

    n_slices_per_subject = df.groupby(subject_col).size()

    agg_df = df.groupby(subject_col).agg({
        **{c: agg_func for c in feature_cols},
        group_col: "first",
    }).reset_index()
    agg_df["n_slices_aggregated"] = agg_df[subject_col].map(n_slices_per_subject)

    logger.info(f"Subject-level aggregation ({agg_func}): "
                f"{df.shape[0]} slices -> {agg_df.shape[0]} subjects")
    logger.info(f"Slices per subject: min={n_slices_per_subject.min()}, "
                f"median={n_slices_per_subject.median():.0f}, "
                f"max={n_slices_per_subject.max()}")
    logger.info(f"Subject counts per group:\n{agg_df.groupby(group_col)[subject_col].nunique()}")

    return agg_df


def correlation_pruning(df, feature_cols, threshold, logger, label="subject-level"):
    corr_matrix = df[feature_cols].corr(method="spearman").abs()

    n_nan = int(corr_matrix.isna().sum().sum())
    if n_nan > 0:
        logger.warning(f"[{label}] Correlation matrix contained {n_nan} NaN entries "
                        f"(near-constant/degenerate feature pairs) — treating as 0")
    corr_matrix = corr_matrix.fillna(0.0)

    remaining = list(feature_cols)
    corr_full = corr_matrix.copy()

    while len(remaining) > 1:
        sub_vals = corr_matrix.loc[remaining, remaining].values.copy()
        np.fill_diagonal(sub_vals, 0)

        max_val = np.nanmax(sub_vals)
        if not np.isfinite(max_val) or max_val <= threshold:
            break

        flat_idx = np.nanargmax(sub_vals)
        i, j = np.unravel_index(flat_idx, sub_vals.shape)
        f1, f2 = remaining[i], remaining[j]

        mean_corr_f1 = sub_vals[i, :].mean()
        mean_corr_f2 = sub_vals[j, :].mean()
        drop = f1 if mean_corr_f1 >= mean_corr_f2 else f2
        remaining.remove(drop)

    logger.info(f"[{label}] Correlation pruning (threshold=|r|>{threshold}): "
                f"{len(feature_cols)} -> {len(remaining)} features "
                f"({len(feature_cols) - len(remaining)} dropped)")
    return remaining, corr_full


def epsilon_squared(h_stat, n):
    return h_stat / ((n**2 - 1) / (n + 1)) if n > 1 else np.nan


def rank_biserial_from_mannwhitney(u_stat, n1, n2):
    return 1 - (2 * u_stat) / (n1 * n2)


def run_kruskal_wallis(df, feature_cols, group_col, groups, logger):
    rows = []
    n_total = len(df)

    for feat in feature_cols:
        samples = [df.loc[df[group_col] == g, feat].values for g in groups]
        try:
            h_stat, p_val = stats.kruskal(*samples)
        except ValueError:
            h_stat, p_val = np.nan, np.nan

        eps_sq = epsilon_squared(h_stat, n_total) if not np.isnan(h_stat) else np.nan

        row = {"feature": feat, "H_statistic": h_stat, "p_value": p_val,
               "epsilon_squared": eps_sq, "n_total": n_total}
        for g, s in zip(groups, samples):
            row[f"median_{g}"] = np.median(s) if len(s) else np.nan
            row[f"n_{g}"] = len(s)
        rows.append(row)

    kw_df = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    logger.info(f"Kruskal-Wallis complete (subject-level, n={n_total}): {len(kw_df)} features tested")
    return kw_df


def apply_fdr(kw_df, alpha, logger):
    kw_df = kw_df.copy()
    valid = kw_df["p_value"].notna()
    reject, p_adj, _, _ = multipletests(
        kw_df.loc[valid, "p_value"], alpha=alpha, method="fdr_bh"
    )
    kw_df.loc[valid, "p_value_fdr"] = p_adj
    kw_df.loc[valid, "significant_fdr"] = reject
    kw_df["significant_fdr"] = kw_df["significant_fdr"].fillna(False).astype(bool)

    n_sig = kw_df["significant_fdr"].sum()
    logger.info(f"FDR correction (alpha={alpha}): {n_sig} / {len(kw_df)} features "
                f"significant by p-value alone (BEFORE effect-size gate)")
    return kw_df.sort_values("p_value_fdr").reset_index(drop=True)


def apply_effect_size_gate(kw_fdr_df, min_effect_size, logger):
    """
    THE KEY FIX for 'statistical vs clinical significance': a feature only
    counts as truly significant if it clears BOTH the FDR-corrected p-value
    AND a minimum effect size. This is what turns '390 significant features'
    into a short, honest list.
    """
    kw_fdr_df = kw_fdr_df.copy()
    kw_fdr_df["passes_effect_size_gate"] = kw_fdr_df["epsilon_squared"] >= min_effect_size
    kw_fdr_df["truly_significant"] = (
        kw_fdr_df["significant_fdr"] & kw_fdr_df["passes_effect_size_gate"]
    )

    n_pval_only = kw_fdr_df["significant_fdr"].sum()
    n_gated = kw_fdr_df["truly_significant"].sum()
    logger.info(f"Effect-size gate (epsilon_squared >= {min_effect_size}): "
                f"{n_pval_only} p-value-significant features -> "
                f"{n_gated} features ALSO clear the effect-size bar")
    logger.info("These 'truly_significant' features are the ones that are both "
                "statistically defensible (FDR-corrected, subject-level N) AND "
                "represent a non-trivial magnitude of difference between groups.")

    return kw_fdr_df.sort_values(["truly_significant", "epsilon_squared"],
                                  ascending=[False, False]).reset_index(drop=True)


def run_pairwise_mannwhitney(df, feature_cols, group_col, groups, alpha, logger):
    all_pairs_results = []

    for g1, g2 in combinations(groups, 2):
        comparison_name = f"{g1}_vs_{g2}"
        rows = []
        for feat in feature_cols:
            s1 = df.loc[df[group_col] == g1, feat].values
            s2 = df.loc[df[group_col] == g2, feat].values

            try:
                u_stat, p_val = stats.mannwhitneyu(s1, s2, alternative="two-sided")
            except ValueError:
                u_stat, p_val = np.nan, np.nan

            r_effect = (rank_biserial_from_mannwhitney(u_stat, len(s1), len(s2))
                        if not np.isnan(u_stat) else np.nan)

            rows.append({
                "comparison": comparison_name,
                "feature": feat,
                "U_statistic": u_stat,
                "p_value": p_val,
                "rank_biserial_effect_size": r_effect,
                f"median_{g1}": np.median(s1) if len(s1) else np.nan,
                f"median_{g2}": np.median(s2) if len(s2) else np.nan,
                "n_" + g1: len(s1),
                "n_" + g2: len(s2),
            })

        pair_df = pd.DataFrame(rows)
        valid = pair_df["p_value"].notna()
        if valid.sum() > 0:
            reject, p_adj, _, _ = multipletests(
                pair_df.loc[valid, "p_value"], alpha=alpha, method="fdr_bh"
            )
            pair_df.loc[valid, "p_value_fdr"] = p_adj
            pair_df.loc[valid, "significant_fdr"] = reject
        pair_df["significant_fdr"] = pair_df["significant_fdr"].fillna(False).astype(bool)

        # also gate pairwise significance by a minimum |rank-biserial effect|
        # (0.3 corresponds roughly to the same "medium effect" tier as
        # epsilon-squared ~0.06 for the omnibus test)
        pair_df["passes_effect_size_gate"] = pair_df["rank_biserial_effect_size"].abs() >= 0.3
        pair_df["truly_significant"] = pair_df["significant_fdr"] & pair_df["passes_effect_size_gate"]

        n_sig = pair_df["significant_fdr"].sum()
        n_gated = pair_df["truly_significant"].sum()
        logger.info(f"Mann-Whitney {comparison_name}: {n_sig} p-significant -> "
                    f"{n_gated} also clear effect-size gate (|r|>=0.3)")

        all_pairs_results.append(pair_df)

    combined = pd.concat(all_pairs_results, ignore_index=True)
    return combined.sort_values(["comparison", "p_value_fdr"]).reset_index(drop=True)


def cluster_redundant_features(df, features, corr_threshold, effect_size_lookup, logger):
    """
    THE KEY FIX for 'feature redundancy': hierarchically cluster the
    significant+large-effect features by |correlation|, cut the dendrogram
    so within-cluster |r| >= corr_threshold, and keep only the single
    highest-effect-size representative per cluster.

    This turns e.g. {GLCM Energy, GLCM Joint Energy, Wavelet GLCM Energy,
    Logarithm GLCM Energy} -- which likely all encode the same underlying
    texture pattern -- into ONE representative feature, chosen as whichever
    of them has the strongest effect size.
    """
    if len(features) <= 1:
        return features, pd.DataFrame({"feature": features, "cluster": [0]*len(features)})

    corr = df[features].corr(method="spearman").abs().fillna(0.0)
    corr_vals = corr.values.copy()
    np.fill_diagonal(corr_vals, 1.0)
    dist = 1 - corr_vals
    dist = (dist + dist.T) / 2       # enforce symmetry against float error
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)

    Z = linkage(condensed, method="average")
    cluster_dist_threshold = 1 - corr_threshold
    cluster_labels = fcluster(Z, t=cluster_dist_threshold, criterion="distance")

    cluster_df = pd.DataFrame({"feature": features, "cluster": cluster_labels})
    cluster_df["effect_size"] = cluster_df["feature"].map(effect_size_lookup)

    representatives = (
        cluster_df.sort_values("effect_size", ascending=False)
        .groupby("cluster")
        .first()
        .reset_index()
    )

    n_clusters = cluster_df["cluster"].nunique()
    logger.info(f"Redundancy clustering (within-cluster |r|>={corr_threshold}): "
                f"{len(features)} significant features -> {n_clusters} clusters -> "
                f"{len(representatives)} non-redundant representative features")

    return representatives["feature"].tolist(), cluster_df


def main():
    logger = setup_logging(OUTPUT_DIR)
    output_dir = Path(OUTPUT_DIR)

    logger.info("=" * 70)
    logger.info("STEP 2 (REVISED): Statistical Analysis with subject-level "
                "aggregation, effect-size gating, and redundancy clustering")
    logger.info("=" * 70)

    df_raw = pd.read_csv(INPUT_CSV, low_memory=False)
    logger.info(f"Loaded raw SLICE-level features: {df_raw.shape[0]} rows x {df_raw.shape[1]} cols")
    logger.info(f"Slice-level group distribution:\n{df_raw[GROUP_COL].value_counts()}")

    # ---- Cleaning (slice-level) ----
    feature_cols = identify_feature_columns(df_raw)
    logger.info(f"Identified {len(feature_cols)} candidate numeric feature columns")
    df_clean, feature_cols = clean_data(df_raw, feature_cols, logger)
    df_clean.to_csv(output_dir / "01_cleaned_slice_level_data.csv", index=False)

    # ---- SUBJECT-LEVEL AGGREGATION (the critical fix) ----
    df_subject = aggregate_to_subject_level(
        df_clean, feature_cols, SUBJECT_COL, GROUP_COL, AGGREGATION_METHOD, logger
    )
    df_subject.to_csv(output_dir / "02_subject_level_aggregated_data.csv", index=False)

    n_subjects = df_subject.shape[0]
    if n_subjects < 30:
        logger.warning(f"Only {n_subjects} subjects after aggregation. Statistical "
                        f"power will be limited; interpret results cautiously.")

    # ---- Correlation pruning (on subject-level data) ----
    pruned_features, full_corr_matrix = correlation_pruning(
        df_subject, feature_cols, CORR_THRESHOLD, logger, label="subject-level"
    )
    full_corr_matrix.to_csv(output_dir / "03_full_correlation_matrix_subject_level.csv")
    pd.Series(pruned_features, name="feature").to_csv(
        output_dir / "03_pruned_feature_list.csv", index=False
    )
    df_pruned = df_subject[[GROUP_COL, SUBJECT_COL, "n_slices_aggregated"] + pruned_features].copy()
    df_pruned.to_csv(output_dir / "03_pruned_data_subject_level.csv", index=False)

    # ---- Kruskal-Wallis (omnibus, 3-group, SUBJECT-level N) ----
    kw_df = run_kruskal_wallis(df_subject, pruned_features, GROUP_COL, GROUPS, logger)
    kw_df.to_csv(output_dir / "04_kruskal_wallis_results.csv", index=False)

    # ---- FDR correction ----
    kw_fdr_df = apply_fdr(kw_df, ALPHA, logger)

    # ---- Effect-size gate (this is what turns "390 significant" into an honest number) ----
    kw_gated_df = apply_effect_size_gate(kw_fdr_df, MIN_EFFECT_SIZE, logger)
    kw_gated_df.to_csv(output_dir / "05_kruskal_wallis_fdr_effect_gated.csv", index=False)

    truly_sig_features = kw_gated_df.loc[
        kw_gated_df["truly_significant"], "feature"
    ].tolist()

    if len(truly_sig_features) == 0:
        logger.warning(f"No features passed BOTH FDR correction AND the "
                        f"epsilon_squared >= {MIN_EFFECT_SIZE} gate. Falling back "
                        f"to the top 20 features by effect size (FDR-significant "
                        f"only) so you still have candidates to inspect.")
        truly_sig_features = kw_fdr_df[kw_fdr_df["significant_fdr"]].sort_values(
            "epsilon_squared", ascending=False
        ).head(20)["feature"].tolist()

    with open(output_dir / "truly_significant_features_list.txt", "w") as f:
        f.write("\n".join(truly_sig_features))
    logger.info(f"{len(truly_sig_features)} features are FDR-significant AND "
                f"have epsilon_squared >= {MIN_EFFECT_SIZE} (medium-or-larger effect)")

    # ---- Redundancy clustering (this fixes "390 features = 390 biomarkers") ----
    effect_size_lookup = dict(zip(kw_gated_df["feature"], kw_gated_df["epsilon_squared"]))
    representative_features, cluster_assignment_df = cluster_redundant_features(
        df_subject, truly_sig_features, REDUNDANCY_CLUSTER_THRESHOLD,
        effect_size_lookup, logger
    )
    cluster_assignment_df = cluster_assignment_df.sort_values(
        ["cluster", "effect_size"], ascending=[True, False]
    )
    cluster_assignment_df.to_csv(output_dir / "06_redundancy_clusters_full_membership.csv", index=False)

    with open(output_dir / "07_final_nonredundant_top_features.txt", "w") as f:
        f.write("\n".join(representative_features))

    final_summary = kw_gated_df[kw_gated_df["feature"].isin(representative_features)].copy()
    final_summary = final_summary.sort_values("epsilon_squared", ascending=False)
    final_summary.to_csv(output_dir / "07_final_nonredundant_top_features.csv", index=False)

    logger.info(f"FINAL non-redundant feature list: {len(representative_features)} features "
                f"(down from {len(truly_sig_features)} significant+large-effect features, "
                f"which came from {len(pruned_features)} pruned features, "
                f"which came from {len(feature_cols)} total extracted features)")

    # ---- Pairwise Mann-Whitney + effect sizes, on the FINAL non-redundant set ----
    pairwise_df = run_pairwise_mannwhitney(
        df_subject, representative_features, GROUP_COL, GROUPS, ALPHA, logger
    )
    pairwise_df.to_csv(output_dir / "08_pairwise_mannwhitney_results.csv", index=False)

    # ---- Summary: which final features are robust across ALL pairwise comparisons ----
    n_comparisons = len(list(combinations(GROUPS, 2)))
    sig_counts = (
        pairwise_df[pairwise_df["truly_significant"]]
        .groupby("feature")["comparison"]
        .nunique()
        .reset_index(name="n_significant_comparisons")
    )
    sig_counts["significant_in_all_pairs"] = sig_counts["n_significant_comparisons"] == n_comparisons
    sig_counts = sig_counts.merge(
        kw_gated_df[["feature", "epsilon_squared", "p_value_fdr"]].rename(
            columns={"p_value_fdr": "kw_p_value_fdr"}
        ),
        on="feature", how="left",
    ).sort_values("epsilon_squared", ascending=False)
    sig_counts.to_csv(output_dir / "09_final_feature_significance_summary.csv", index=False)

    logger.info("=" * 70)
    logger.info("STEP 2 COMPLETE.")
    logger.info(f"PIPELINE: {df_raw.shape[0]} slices ({df_raw[SUBJECT_COL].nunique()} subjects) "
                f"-> {n_subjects} subjects (aggregated) "
                f"-> {len(pruned_features)} features (correlation-pruned) "
                f"-> {len(truly_sig_features)} significant+large-effect features "
                f"-> {len(representative_features)} FINAL non-redundant features")
    logger.info(f"Read 07_final_nonredundant_top_features.csv for your top features.")
    logger.info(f"All outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
