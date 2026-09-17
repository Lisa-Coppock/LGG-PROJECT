import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Bio import SeqIO

import umap
import hdbscan
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, normalize

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------


UMAP_N_NEIGHBORS = 50 
UMAP_MIN_DIST = 0.0 
UMAP_N_COMPONENTS_2D = 2        # for plotting
UMAP_N_COMPONENTS_CLUSTERING = 50  # for clustering

MIN_CLUSTER_SIZE = 10  # value for HDBSCAN, can be adjusted (min noise/best silhouette score)

SAVE_INTERMEDIATE = True

# -------------------------------------------------------------------------
# 1. LOAD EMBEDDINGS AND IDS
# -------------------------------------------------------------------------

print("[1/5] Loading embeddings and gene IDs...")

embeddings = np.load("esm2_embeddings.npy") # change to your embeddings file path
gene_ids_df = pd.read_csv("gene_ids.csv") # change to your gene IDs file path

print(f"  Loaded {len(gene_ids_df):,} genes")
print(f"  Embedding shape: {embeddings.shape}")
print(f"  Approx. memory: {embeddings.nbytes / (1024**3):.2f} GB")

# -------------------------------------------------------------------------
# 2. PREPROCESS EMBEDDINGS (CLIP → STANDARDIZE → L2 NORMALIZE)
# -------------------------------------------------------------------------

print("\n[2/5] Preprocessing embeddings...")

# Clip extreme outliers
embeddings = np.clip(embeddings, -500, 500)

print(f"  After clipping: min={embeddings.min():.2f}, max={embeddings.max():.2f}")

# Standardize (z-score)
scaler = StandardScaler()
embeddings = scaler.fit_transform(embeddings)


# L2 normalize per protein
embeddings = normalize(embeddings, norm="l2", axis=1)



# -------------------------------------------------------------------------
# 3. ADD STRAIN AND ANNOTATION METADATA
# -------------------------------------------------------------------------

print("\n[3/5] Adding strain and annotation metadata...")

# Derive strain from gene_id (adjust to your naming convention if needed)
gene_ids_df["strain"] = (
    gene_ids_df["gene_id"]
    .str.split("_")
    .apply(lambda parts: "_".join(parts[:-2]))
    .str.replace(" ", "_")
)

n_strains = gene_ids_df["strain"].nunique()
print(f"  Found {n_strains} unique strains")

# Parse annotations from FASTA headers
annotations = {}
for record in SeqIO.parse("all_yes_no_proteins.faa", "fasta"):
    gene_id = record.id
    header = record.description
    parts = header.split(" ")
    if len(parts) >= 4:
        annotation = " ".join(parts[3:]).strip()
    else:
        annotation = "hypothetical protein"

    if not annotation:
        annotation = "hypothetical protein"

    annotations[gene_id] = annotation

gene_ids_df["annotation"] = gene_ids_df["gene_id"].map(annotations)
gene_ids_df["annotation"] = gene_ids_df["annotation"].fillna("hypothetical protein")

# -------------------------------------------------------------------------
# 4. UMAP + HDBSCAN CLUSTERING 
# -------------------------------------------------------------------------

print("\n[4/5] UMAP dimensionality reduction and HDBSCAN clustering ...")

# 2D UMAP for visualization
reducer_2d = umap.UMAP(
    n_neighbors=UMAP_N_NEIGHBORS,
    min_dist=UMAP_MIN_DIST,
    n_components=UMAP_N_COMPONENTS_2D,
    metric="cosine",
    random_state=42,
    verbose=True,
    n_jobs=-1,
)
embedding_2d = reducer_2d.fit_transform(embeddings)
gene_ids_df["umap_1"] = embedding_2d[:, 0]
gene_ids_df["umap_2"] = embedding_2d[:, 1]

if SAVE_INTERMEDIATE:
    np.save("umap_2d_coordinates.npy", embedding_2d)
    gene_ids_df[["gene_id", "strain", "umap_1", "umap_2"]].to_csv(
        "umap_2d_coordinates.csv", index=False
    )

# 50D UMAP for clustering
reducer_50d = umap.UMAP(
    n_neighbors=UMAP_N_NEIGHBORS,
    min_dist=UMAP_MIN_DIST,
    n_components=UMAP_N_COMPONENTS_CLUSTERING,
    metric="cosine",
    random_state=42,
    verbose=True,
    n_jobs=-1,
)
embedding_50d = reducer_50d.fit_transform(embeddings)



print(f"  Using min_cluster_size={MIN_CLUSTER_SIZE}")

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=MIN_CLUSTER_SIZE,
    metric="euclidean",
    core_dist_n_jobs=-1,
)

clusters = clusterer.fit_predict(embedding_50d)

n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
n_noise = int((clusters == -1).sum())

print(
    f"  Found {n_clusters} clusters; "
    f"noise points: {n_noise:,} ({100 * n_noise / len(clusters):.1f}%)"
)

gene_ids_df["cluster"] = clusters

# Optional: silhouette score on a sample of clustered points
clustered_mask = clusters != -1
n_clustered = clustered_mask.sum()

if n_clustered > 0 and n_clusters > 1:
    sample_size = min(10000, n_clustered)
    clustered_indices = np.where(clustered_mask)[0]
    sample_indices = np.random.choice(clustered_indices, sample_size, replace=False)

    sample_embeddings = embedding_50d[sample_indices]
    sample_clusters = clusters[sample_indices]
    sil_score = silhouette_score(sample_embeddings, sample_clusters)

    print(f"  Silhouette score (sample of {sample_size:,}): {sil_score:.3f}")
    print("  (Range -1 to 1; higher = better separation, >0.5 often considered good)")
else:
    print("  Skipping silhouette score (too few clustered points or clusters).")

# -------------------------------------------------------------------------
# 5. STRAIN-SPECIFIC ANALYSIS + OUTPUTS
# -------------------------------------------------------------------------

print("\n[5/5] Generating outputs...")

# Save main tables
gene_ids_df.to_csv("all_genes_with_clusters.csv", index=False)

cluster_stats = (
    gene_ids_df.groupby("cluster")
    .agg({"gene_id": "count", "strain": "nunique"})
    .reset_index()
)
cluster_stats.columns = ["cluster", "n_genes", "n_strains"]
cluster_stats.to_csv("cluster_statistics.csv", index=False)

# -------------------------------------------------------------------------
# 6. VISUALIZATIONS
# -------------------------------------------------------------------------

# Sample for plotting if very large
sample_size = min(100000, len(gene_ids_df))
if sample_size < len(gene_ids_df):
    sample_idx = np.random.choice(len(gene_ids_df), sample_size, replace=False)
    viz_data = gene_ids_df.iloc[sample_idx]
else:
    viz_data = gene_ids_df

# Plot 1: UMAP colored by cluster
plt.figure(figsize=(12, 9))
scatter = plt.scatter(
    viz_data["umap_1"],
    viz_data["umap_2"],
    c=viz_data["cluster"],
    s=1,
    cmap="tab20",
    alpha=0.6,
)
plt.colorbar(scatter, label="Cluster (-1 = noise)")
plt.title(
    f"UMAP: {n_strains} strains, {len(gene_ids_df):,} genes, {n_clusters} clusters",
    fontsize=14,
)
plt.xlabel("UMAP 1", fontsize=12)
plt.ylabel("UMAP 2", fontsize=12)
plt.tight_layout()
plt.savefig("umap_clusters.png", dpi=300, bbox_inches="tight")
plt.close()


print("\nDone. Outputs written:")
print("  - umap_clusters.png")
print("  - all_genes_with_clusters.csv")
print("  - cluster_statistics.csv")
print("  - umap_2d_coordinates.npy / umap_2d_coordinates.csv")
