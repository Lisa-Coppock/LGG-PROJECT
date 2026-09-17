import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from tqdm import tqdm
from statsmodels.stats.multitest import multipletests


# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------

ALL_GENES_FILE = "all_genes_with_clusters.csv"
STRAIN_METADATA_FILE = "strain_metadata.csv"

FOLD_CHANGE_THRESHOLD = 0.5  # log2 fold-change threshold
Q_VALUE_THRESHOLD = 0.05

STATISTICAL_PSEUDOCOUNT = 1.0

# Matches names such as:
# Lacticaseibacillus_rhamnosus_GG_1
# Lacticaseibacillus_rhamnosus_GG_ID
# Lacticaseibacillus_rhamnosus_LR-GG-MoProbi
LGG_PATTERN = r"rhamnosus.*GG|GG.*rhamnosus"


print("=" * 70)
print("BENEFICIAL STRAIN ENRICHMENT ANALYSIS")
print("=" * 70)


# -------------------------------------------------------------------------
# Helper: Benjamini–Hochberg FDR correction
# -------------------------------------------------------------------------

def compute_q_values(p_values):
    """Adjust p-values using the Benjamini–Hochberg procedure."""
    p_values = np.asarray(p_values)
    _, q_values, _, _ = multipletests(p_values, method="fdr_bh")
    return q_values


# -------------------------------------------------------------------------
# Helper: standardise strain names for matching
# -------------------------------------------------------------------------

def normalize_strain_name(name):
    """Convert strain names to a consistent matching format."""
    if pd.isna(name):
        return name

    return (
        str(name)
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


# -------------------------------------------------------------------------
# 1. LOAD CLUSTERING RESULTS
# -------------------------------------------------------------------------

print("\n[1/5] Loading clustering results...")

gene_data = pd.read_csv(ALL_GENES_FILE)

n_clusters = gene_data["cluster"].nunique()
n_noise = (gene_data["cluster"] == -1).sum()

print(f"  Total genes: {len(gene_data):,}")
print(f"  Clusters including noise: {n_clusters:,}")
print(f"  Noise points: {n_noise:,}")


# -------------------------------------------------------------------------
# 2. ADD BENEFICIAL/NON-BENEFICIAL LABELS
# -------------------------------------------------------------------------

print("\n[2/5] Loading strain classifications...")

strain_classes = pd.read_csv(STRAIN_METADATA_FILE)

required_metadata_columns = {"strain", "beneficial"}
missing_columns = required_metadata_columns - set(strain_classes.columns)

if missing_columns:
    raise ValueError(
        f"strain_metadata.csv is missing required columns: {missing_columns}"
    )

strain_classes["strain_normalized"] = strain_classes["strain"].apply(
    normalize_strain_name
)

gene_data["strain_normalized"] = gene_data["strain"].apply(
    normalize_strain_name
)

gene_data = gene_data.merge(
    strain_classes[["strain_normalized", "beneficial"]],
    on="strain_normalized",
    how="left",
)

n_unclassified = gene_data["beneficial"].isna().sum()

if n_unclassified > 0:
    print(f"  Removing genes from {n_unclassified:,} unclassified entries")

gene_data = gene_data.dropna(subset=["beneficial"]).copy()
gene_data["beneficial"] = gene_data["beneficial"].astype(int)

print(f"  Classified genes: {len(gene_data):,}")
print(
    f"  Beneficial-strain genes: "
    f"{(gene_data['beneficial'] == 1).sum():,}"
)
print(
    f"  Non-beneficial-strain genes: "
    f"{(gene_data['beneficial'] == 0).sum():,}"
)
print(f"  Unique classified strains: {gene_data['strain'].nunique()}")


# Ensure annotations are available
gene_data["annotation"] = gene_data["annotation"].fillna(
    "hypothetical protein"
)


# -------------------------------------------------------------------------
# 3. CALCULATE CLUSTER STATISTICS
# -------------------------------------------------------------------------

print("\n[3/5] Calculating cluster statistics...")

# Exclude HDBSCAN noise points from enrichment analysis
clustered_genes = gene_data[gene_data["cluster"] != -1].copy()

cluster_stats = []

for cluster_id in tqdm(
    clustered_genes["cluster"].unique(),
    desc="Cluster statistics",
):
    cluster_genes = clustered_genes[
        clustered_genes["cluster"] == cluster_id
    ]

    annotation = (
        cluster_genes["annotation"].mode().iloc[0]
        if not cluster_genes.empty
        else "unknown"
    )

    is_hypothetical = any(
        keyword in annotation.lower()
        for keyword in [
            "hypothetical",
            "uncharacterized",
            "unknown",
            "putative",
        ]
    )

    cluster_stats.append(
        {
            "cluster_id": cluster_id,
            "n_genes": len(cluster_genes),
            "n_beneficial": int(
                (cluster_genes["beneficial"] == 1).sum()
            ),
            "n_nonbeneficial": int(
                (cluster_genes["beneficial"] == 0).sum()
            ),
            "annotation": annotation,
            "is_hypothetical": is_hypothetical,
            "n_strains": cluster_genes["strain"].nunique(),
        }
    )

cluster_stats_df = pd.DataFrame(cluster_stats)

print(f"  Clusters analysed: {len(cluster_stats_df):,}")
print(
    f"  Mean cluster size: "
    f"{cluster_stats_df['n_genes'].mean():.1f}"
)
print(
    f"  Median cluster size: "
    f"{cluster_stats_df['n_genes'].median():.0f}"
)


# -------------------------------------------------------------------------
# 4. ENRICHMENT ANALYSIS
# -------------------------------------------------------------------------

print("\n[4/5] Running Fisher's exact tests...")

total_beneficial = int(
    (clustered_genes["beneficial"] == 1).sum()
)

total_nonbeneficial = int(
    (clustered_genes["beneficial"] == 0).sum()
)

total_genes = len(clustered_genes)

if total_beneficial == 0 or total_nonbeneficial == 0:
    raise ValueError(
        "Both beneficial and non-beneficial genes are required "
        "for enrichment analysis."
    )

print(
    f"  Background genes: "
    f"{total_beneficial:,} beneficial, "
    f"{total_nonbeneficial:,} non-beneficial"
)

enrichment_results = []

for _, cluster in tqdm(
    cluster_stats_df.iterrows(),
    total=len(cluster_stats_df),
    desc="Fisher tests",
):
    n_beneficial = int(cluster["n_beneficial"])
    n_nonbeneficial = int(cluster["n_nonbeneficial"])
    total_cluster_genes = int(cluster["n_genes"])

    # 2 × 2 contingency table:
    #
    #                  In cluster   Outside cluster
    # Beneficial           a              b
    # Non-beneficial       c              d
    a = n_beneficial
    b = total_beneficial - n_beneficial
    c = n_nonbeneficial
    d = total_nonbeneficial - n_nonbeneficial

    odds_ratio, p_value = fisher_exact(
        [[a, b], [c, d]],
        alternative="two-sided",
    )

    # Expected beneficial genes under the overall background proportion
    expected_beneficial = total_cluster_genes * (
        total_beneficial / total_genes
    )

    fold_change_linear = (
        (n_beneficial + STATISTICAL_PSEUDOCOUNT)
        / (expected_beneficial + STATISTICAL_PSEUDOCOUNT)
    )

    fold_change_log2 = np.log2(fold_change_linear)

    enrichment_results.append(
        {
            "cluster_id": cluster["cluster_id"],
            "annotation": cluster["annotation"],
            "is_hypothetical": cluster["is_hypothetical"],
            "total_genes": total_cluster_genes,
            "n_beneficial": n_beneficial,
            "n_nonbeneficial": n_nonbeneficial,
            "prop_beneficial": (
                n_beneficial / total_cluster_genes
            ),
            "fold_change_linear": fold_change_linear,
            "fold_change_log2": fold_change_log2,
            "odds_ratio": odds_ratio,
            "p_value": p_value,
            "n_strains": cluster["n_strains"],
        }
    )

enrichment_df = pd.DataFrame(enrichment_results)

if enrichment_df.empty:
    raise ValueError("No clusters were available for enrichment analysis.")

# Multiple-testing correction
enrichment_df["q_value"] = compute_q_values(
    enrichment_df["p_value"]
)

# Classify clusters
enrichment_df["enrichment"] = "neutral"

enrichment_df.loc[
    (
        enrichment_df["fold_change_log2"]
        > FOLD_CHANGE_THRESHOLD
    )
    & (
        enrichment_df["q_value"]
        < Q_VALUE_THRESHOLD
    ),
    "enrichment",
] = "beneficial"

enrichment_df.loc[
    (
        enrichment_df["fold_change_log2"]
        < -FOLD_CHANGE_THRESHOLD
    )
    & (
        enrichment_df["q_value"]
        < Q_VALUE_THRESHOLD
    ),
    "enrichment",
] = "non-beneficial"

n_beneficial_enriched = (
    enrichment_df["enrichment"] == "beneficial"
).sum()

n_nonbeneficial_enriched = (
    enrichment_df["enrichment"] == "non-beneficial"
).sum()

n_neutral = (
    enrichment_df["enrichment"] == "neutral"
).sum()

print(
    f"  Beneficial-enriched clusters: "
    f"{n_beneficial_enriched:,}"
)
print(
    f"  Non-beneficial-enriched clusters: "
    f"{n_nonbeneficial_enriched:,}"
)
print(f"  Neutral clusters: {n_neutral:,}")

beneficial_enriched = enrichment_df[
    enrichment_df["enrichment"] == "beneficial"
].copy()


# -------------------------------------------------------------------------
# 5. LGG ANALYSIS AND OUTPUTS
# -------------------------------------------------------------------------

print("\n[5/5] LGG analysis and saving results...")

lgg_proteins = gene_data[
    gene_data["strain"].str.contains(
        LGG_PATTERN,
        case=False,
        regex=True,
        na=False,
    )
].copy()

lgg_beneficial = pd.DataFrame()

if not lgg_proteins.empty:
    matched_lgg_strains = sorted(
        lgg_proteins["strain"].unique()
    )

    print(f"  LGG strains matched: {matched_lgg_strains}")
    print(f"  LGG proteins: {len(lgg_proteins):,}")

    lgg_beneficial = lgg_proteins[
        lgg_proteins["cluster"].isin(
            beneficial_enriched["cluster_id"]
        )
    ].copy()

    print(
        "  LGG proteins in beneficial-enriched clusters: "
        f"{len(lgg_beneficial):,}"
    )

    if not lgg_beneficial.empty:
        lgg_beneficial.to_csv(
            "lgg_beneficial_enriched_proteins.csv",
            index=False,
        )
        print(
            "  Saved: lgg_beneficial_enriched_proteins.csv"
        )
else:
    print(
        "  No L. rhamnosus GG proteins found. "
        "Check the strain naming pattern."
    )

# Save full results
enrichment_df.to_csv(
    "cluster_enrichment_analysis.csv",
    index=False,
)

cluster_stats_df.to_csv(
    "cluster_statistics.csv",
    index=False,
)

# Save beneficial-enriched clusters sorted by statistical fold change
beneficial_enriched_sorted = beneficial_enriched.sort_values(
    "fold_change_log2",
    ascending=False,
)

beneficial_enriched_sorted.to_csv(
    "beneficial_enriched_clusters.csv",
    index=False,
)

# Save hypothetical beneficial-enriched clusters
hypothetical_clusters = beneficial_enriched[
    beneficial_enriched["is_hypothetical"]
].sort_values(
    "fold_change_log2",
    ascending=False,
)

hypothetical_clusters.to_csv(
    "beneficial_hypothetical_clusters.csv",
    index=False,
)

print("\nSaved:")
print("  cluster_enrichment_analysis.csv")
print("  cluster_statistics.csv")
print("  beneficial_enriched_clusters.csv")
print("  beneficial_hypothetical_clusters.csv")

if not lgg_beneficial.empty:
    print("  lgg_beneficial_enriched_proteins.csv")

print("\nAnalysis complete.")