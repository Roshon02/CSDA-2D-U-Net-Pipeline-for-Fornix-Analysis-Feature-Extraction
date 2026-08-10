

import logging
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# ============================================================================
# ======================  USER CONFIGURATION — EDIT THIS  ==================
# ============================================================================

# OUTPUT_DIR from 02_statistical_analysis_revised.py
STATS_DIR = r"F:\PROJECT_FORNIX\For_Paper\2D_Features_Final\statistical_analysis"
OUTPUT_DIR = r"F:\PROJECT_FORNIX\For_Paper\2D_Features_Final\figures"

GROUP_COL = "group"
SUBJECT_COL = "subject_id"
GROUPS = ["AD", "CN", "MCI"]
GROUP_PALETTE = {"AD": "#d62728", "CN": "#2ca02c", "MCI": "#1f77b4"}

# Which feature set to visualize with. "final" (recommended) uses the short,
# non-redundant, effect-size-gated list -- what you'd actually report in a
# paper. "pruned" uses the larger correlation-pruned-only set, useful if you
# want to see the full structure before gating.
FEATURE_SET = "final"   # "final" or "pruned"

TOP_N_FEATURES_FOR_IMPORTANCE = 15
TOP_N_FEATURES_FOR_VOLCANO_LABELS = 15
ALPHA = 0.05

FIGSIZE_DEFAULT = (8, 6)
DPI = 300

sns.set_theme(style="whitegrid", context="talk")

# ============================================================================
# ==========================  END CONFIGURATION  ============================
# ============================================================================


def setup_logging(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "pipeline_step3.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler()],
    )
    return logging.getLogger("step3")


def load_inputs(stats_dir, feature_set, logger):
    stats_dir = Path(stats_dir)

    data = {
        # subject-level data, all pruned features (n_subjects x ~400 features)
        "pruned_data": pd.read_csv(stats_dir / "03_pruned_data_subject_level.csv"),
        "full_corr_matrix": pd.read_csv(
            stats_dir / "03_full_correlation_matrix_subject_level.csv", index_col=0
        ),
        # KW + FDR + effect-size gate results for all pruned features
        "kw_gated": pd.read_csv(stats_dir / "05_kruskal_wallis_fdr_effect_gated.csv"),
        # pairwise Mann-Whitney results, computed only on the final non-redundant set
        "pairwise": pd.read_csv(stats_dir / "08_pairwise_mannwhitney_results.csv"),
        # final non-redundant top features (with their KW stats attached)
        "final_features_df": pd.read_csv(stats_dir / "07_final_nonredundant_top_features.csv"),
    }

    with open(stats_dir / "07_final_nonredundant_top_features.txt") as f:
        final_features = [l.strip() for l in f if l.strip()]
    with open(stats_dir / "truly_significant_features_list.txt") as f:
        truly_sig_features = [l.strip() for l in f if l.strip()]

    data["final_features"] = final_features
    data["truly_sig_features"] = truly_sig_features

    active_features = final_features if feature_set == "final" else truly_sig_features
    data["active_features"] = active_features

    logger.info(f"Loaded subject-level pruned data: {data['pruned_data'].shape}")
    logger.info(f"Final non-redundant features: {len(final_features)}")
    logger.info(f"Truly-significant (pre-clustering) features: {len(truly_sig_features)}")
    logger.info(f"Using FEATURE_SET='{feature_set}' -> {len(active_features)} features "
                f"for PCA/heatmap/UMAP/importance")
    return data


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------
def plot_pca(pruned_data, feature_cols, group_col, out_dir, logger):
    X = pruned_data[feature_cols].values
    y = pruned_data[group_col].values

    X_scaled = StandardScaler().fit_transform(X)
    n_comp = min(10, X_scaled.shape[1], X_scaled.shape[0] - 1)
    pca = PCA(n_components=n_comp)
    pcs = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=FIGSIZE_DEFAULT)
    for g in GROUPS:
        mask = y == g
        ax.scatter(pcs[mask, 0], pcs[mask, 1], label=g, alpha=0.75, s=70,
                   color=GROUP_PALETTE.get(g), edgecolor="white", linewidth=0.5)
    ax.set_xlabel(f"PC1 ({explained[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}% variance)")
    ax.set_title(f"PCA of Fornix Radiomics Features (subject-level, n={len(y)})")
    ax.legend(title="Group")
    fig.tight_layout()
    fig.savefig(out_dir / "pca_scatter_pc1_pc2.png", dpi=DPI)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(range(1, len(explained) + 1), np.cumsum(explained), marker="o")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative explained variance (%)")
    ax.set_title("PCA Scree Plot")
    fig.tight_layout()
    fig.savefig(out_dir / "pca_scree_plot.png", dpi=DPI)
    plt.close(fig)

    pd.DataFrame(pcs[:, :5], columns=[f"PC{i+1}" for i in range(min(5, pcs.shape[1]))]).assign(
        group=y, subject_id=pruned_data[SUBJECT_COL].values
    ).to_csv(out_dir / "pca_scores.csv", index=False)

    logger.info("Saved PCA scatter, scree plot, and scores CSV")


# ---------------------------------------------------------------------------
# Correlation heatmap
# ---------------------------------------------------------------------------
def plot_correlation_heatmap(corr_matrix, active_features, out_dir, logger, max_features=30):
    feats = [f for f in active_features if f in corr_matrix.columns]
    if len(feats) < 5:
        feats = list(corr_matrix.columns[:max_features])
    feats = feats[:max_features]

    sub = corr_matrix.loc[feats, feats]

    fig, ax = plt.subplots(figsize=(max(10, len(feats) * 0.28), max(8, len(feats) * 0.28)))
    sns.heatmap(sub, cmap="coolwarm", center=0, vmin=-1, vmax=1, square=True,
                cbar_kws={"label": "Spearman correlation"},
                xticklabels=True, yticklabels=True, ax=ax)
    ax.set_title(f"Correlation Heatmap: Final Non-Redundant Features (n={len(feats)})")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "correlation_heatmap.png", dpi=DPI)
    plt.close(fig)
    logger.info("Saved correlation heatmap")


# ---------------------------------------------------------------------------
# Volcano plots (one per pairwise comparison, computed on final feature set)
# ---------------------------------------------------------------------------
def plot_volcano(pairwise_df, out_dir, logger, top_n_labels=10, alpha=0.05):
    """
    Volcano plots now mark 'truly_significant' (FDR AND effect-size gate)
    rather than FDR alone, so the coloring matches what the paper will
    actually claim as significant.
    """
    for comparison in pairwise_df["comparison"].unique():
        sub = pairwise_df[pairwise_df["comparison"] == comparison].copy()
        sub["neg_log10_p"] = -np.log10(sub["p_value_fdr"].clip(lower=1e-300))

        fig, ax = plt.subplots(figsize=FIGSIZE_DEFAULT)
        colors = np.where(
            sub["truly_significant"], "#d62728",
            np.where(sub["significant_fdr"], "#ff7f0e", "#7f7f7f")
        )
        ax.scatter(sub["rank_biserial_effect_size"], sub["neg_log10_p"],
                   c=colors, alpha=0.75, s=45, edgecolor="none")
        ax.axhline(-np.log10(alpha), color="black", linestyle="--", linewidth=1,
                  label=f"FDR p = {alpha}")
        ax.axvline(0.3, color="gray", linestyle=":", linewidth=1, label="|effect| = 0.3 gate")
        ax.axvline(-0.3, color="gray", linestyle=":", linewidth=1)
        ax.set_xlabel("Rank-biserial effect size")
        ax.set_ylabel("-log10(FDR-adjusted p-value)")
        ax.set_title(f"Volcano Plot: {comparison} (subject-level)")

        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728',
                       markersize=8, label='FDR-sig + effect-size gate'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e',
                       markersize=8, label='FDR-sig only (small effect)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#7f7f7f',
                       markersize=8, label='Not significant'),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

        top_hits = sub[sub["truly_significant"]].sort_values("neg_log10_p", ascending=False).head(top_n_labels)
        for _, row in top_hits.iterrows():
            short_name = row["feature"] if len(row["feature"]) <= 35 else "..." + row["feature"][-32:]
            ax.annotate(short_name, (row["rank_biserial_effect_size"], row["neg_log10_p"]),
                       fontsize=6, alpha=0.85,
                       xytext=(3, 3), textcoords="offset points")

        fig.tight_layout()
        fig.savefig(out_dir / f"volcano_{comparison}.png", dpi=DPI)
        plt.close(fig)
        logger.info(f"Saved volcano plot for {comparison} "
                    f"({sub['truly_significant'].sum()} gated-significant of {len(sub)} features shown)")


# ---------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------
def plot_umap(pruned_data, feature_cols, group_col, out_dir, logger):
    if not UMAP_AVAILABLE:
        logger.warning("umap-learn not installed (pip install umap-learn) — skipping UMAP")
        return

    X = pruned_data[feature_cols].values
    y = pruned_data[group_col].values
    X_scaled = StandardScaler().fit_transform(X)

    n_neighbors = min(15, max(2, len(X_scaled) - 1))
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=FIGSIZE_DEFAULT)
    for g in GROUPS:
        mask = y == g
        ax.scatter(embedding[mask, 0], embedding[mask, 1], label=g, alpha=0.75, s=70,
                  color=GROUP_PALETTE.get(g), edgecolor="white", linewidth=0.5)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(f"UMAP Projection (subject-level, n={len(y)})")
    ax.legend(title="Group")
    fig.tight_layout()
    fig.savefig(out_dir / "umap_scatter.png", dpi=DPI)
    plt.close(fig)

    pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"]).assign(
        group=y, subject_id=pruned_data[SUBJECT_COL].values
    ).to_csv(out_dir / "umap_scores.csv", index=False)
    logger.info("Saved UMAP scatter and scores CSV")


# ---------------------------------------------------------------------------
# Feature importance (ranked by KW effect size among the FINAL feature set)
# ---------------------------------------------------------------------------
def plot_feature_importance(final_features_df, out_dir, logger, top_n=10):
    top = final_features_df.sort_values("epsilon_squared", ascending=False).head(top_n).copy()
    top["short_name"] = top["feature"].apply(
        lambda x: x if len(x) <= 45 else "..." + x[-42:]
    )

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    ax.barh(top["short_name"], top["epsilon_squared"], color="#d62728")
    ax.invert_yaxis()
    ax.set_xlabel("Kruskal-Wallis Effect Size (epsilon-squared)")
    ax.set_title(f"Top {top_n} Non-Redundant Features (AD vs CN vs MCI, subject-level)")
    ax.axvline(0.06, color="gray", linestyle=":", linewidth=1, label="medium effect (0.06)")
    ax.axvline(0.14, color="black", linestyle=":", linewidth=1, label="large effect (0.14)")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "feature_importance_top_features.png", dpi=DPI)
    plt.close(fig)

    top.to_csv(out_dir / "feature_importance_table.csv", index=False)
    logger.info(f"Saved feature importance plot (top {top_n} of final non-redundant set)")


# ---------------------------------------------------------------------------
# ROC curves (per pairwise comparison, top individual features from final set)
# ---------------------------------------------------------------------------
def plot_roc_curves(pruned_data, active_features, final_features_df, group_col, out_dir, logger, top_n=6):
    ranked_features = (
        final_features_df[final_features_df["feature"].isin(active_features)]
        .sort_values("epsilon_squared", ascending=False)["feature"]
        .head(top_n).tolist()
    )
    if len(ranked_features) == 0:
        logger.warning("No features available for ROC curves — skipping")
        return

    for g1, g2 in combinations(GROUPS, 2):
        comparison = f"{g1}_vs_{g2}"
        sub = pruned_data[pruned_data[group_col].isin([g1, g2])].copy()
        y_true = (sub[group_col] == g1).astype(int).values

        fig, ax = plt.subplots(figsize=FIGSIZE_DEFAULT)
        auc_rows = []

        for feat in ranked_features:
            scores = sub[feat].values
            fpr, tpr, _ = roc_curve(y_true, scores)
            roc_auc = auc(fpr, tpr)
            if roc_auc < 0.5:
                fpr, tpr, _ = roc_curve(y_true, -scores)
                roc_auc = auc(fpr, tpr)

            short_name = feat if len(feat) <= 30 else "..." + feat[-27:]
            ax.plot(fpr, tpr, label=f"{short_name} (AUC={roc_auc:.2f})", linewidth=1.5)
            auc_rows.append({"comparison": comparison, "feature": feat, "AUC": roc_auc})

        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curves: {g1} vs {g2} (subject-level, top features)")
        ax.legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        fig.savefig(out_dir / f"roc_{comparison}.png", dpi=DPI)
        plt.close(fig)

        pd.DataFrame(auc_rows).sort_values("AUC", ascending=False).to_csv(
            out_dir / f"roc_auc_table_{comparison}.csv", index=False
        )
        logger.info(f"Saved ROC curves for {comparison}")


def main():
    logger = setup_logging(OUTPUT_DIR)
    out_dir = Path(OUTPUT_DIR)

    logger.info("=" * 70)
    logger.info("STEP 3 (REVISED): Visualization (subject-level)")
    logger.info("=" * 70)

    data = load_inputs(STATS_DIR, FEATURE_SET, logger)
    pruned_data = data["pruned_data"]
    active_features = data["active_features"]

    plot_pca(pruned_data, active_features, GROUP_COL, out_dir, logger)
    plot_correlation_heatmap(data["full_corr_matrix"], active_features, out_dir, logger)
    plot_volcano(data["pairwise"], out_dir, logger,
                 top_n_labels=TOP_N_FEATURES_FOR_VOLCANO_LABELS, alpha=ALPHA)
    plot_umap(pruned_data, active_features, GROUP_COL, out_dir, logger)
    plot_feature_importance(data["final_features_df"], out_dir, logger,
                             top_n=TOP_N_FEATURES_FOR_IMPORTANCE)
    plot_roc_curves(pruned_data, active_features, data["final_features_df"], GROUP_COL,
                     out_dir, logger, top_n=6)

    logger.info("STEP 3 COMPLETE.")
    logger.info(f"All figures and tables saved to {out_dir}")


if __name__ == "__main__":
    main()
