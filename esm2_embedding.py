import torch
import esm
from Bio import SeqIO
import numpy as np
import pandas as pd
from pathlib import Path
import time

# ==========================
# USER SETTINGS – EDIT THESE
# ==========================

# >>> PUT YOUR INPUT FASTA FILE NAME HERE <<<
PROTEIN_FILE = "all_yes_no_proteins.faa"

# >>> WHERE OUTPUTS SHOULD BE SAVED <<<
OUTPUT_DIR = "esm2_embeddings"

# Batch size for inference (adjust if you hit GPU memory limits)
BATCH_SIZE = 4

# Maximum sequence length for ESM-2 (longer sequences are truncated)
MAX_LENGTH = 2000  # chosen to match the data used in your paper


# ==========================
# 1. Setup
# ==========================

output_path = Path(OUTPUT_DIR)
output_path.mkdir(exist_ok=True, parents=True)

protein_path = Path(PROTEIN_FILE)
if not protein_path.exists():
    raise FileNotFoundError(f"Input FASTA not found: {protein_path}")

print("1. Loading ESM-2 model (esm2_t36_3B_UR50D)...")
model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"   ✓ Using device: {device}")

# Use half precision only on GPU for speed; keep full precision on CPU.
if device.type == "cuda":
    model.half()
else:
    print("   WARNING: Running on CPU - this will be very slow.")
    print("   Consider requesting a GPU node on the cluster.")


# ==========================
# 2. Load sequences
# ==========================

print("\n2. Loading sequences from FASTA...")
sequences = []
for record in SeqIO.parse(str(protein_path), "fasta"):
    seq = str(record.seq)
    if len(seq) > MAX_LENGTH:
        seq = seq[:MAX_LENGTH]  # truncate long proteins
    sequences.append((record.id, seq))

n_sequences = len(sequences)
print(f"   ✓ Loaded {n_sequences:,} sequences (≤{MAX_LENGTH} aa)")


# ==========================
# 3. Generate embeddings
# ==========================

print("\n3. Generating mean embeddings from layer 33...")
all_embeddings = []
all_gene_ids = []

start_time = time.time()

for i in range(0, n_sequences, BATCH_SIZE):
    batch = sequences[i : i + BATCH_SIZE]
    batch_labels, batch_strs, batch_tokens = batch_converter(batch)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        # Representations from layer 33 (ESM-2 hidden layer index)
        # EDIT HERE if you want to use a different layer for embeddings
        results = model(batch_tokens, repr_layers=[33])
        embeddings = results["representations"][33].mean(1)

    all_embeddings.append(embeddings.cpu().numpy())
    all_gene_ids.extend([seq[0] for seq in batch])

    processed = min(i + BATCH_SIZE, n_sequences)

    # Progress print every ~5000 sequences
    if processed % 5000 == 0:
        elapsed = time.time() - start_time
        progress = processed / n_sequences
        eta = elapsed / progress - elapsed
        print(
            f"   Batch {i//BATCH_SIZE + 1} - "
            f"{100 * progress:5.1f}% - ETA: {eta/60:4.0f} min"
        )

embeddings_matrix = np.vstack(all_embeddings)
elapsed = time.time() - start_time

print(f"\n   ✓ Generated {len(embeddings_matrix):,} embeddings")
print(f"   ✓ Time: {elapsed/3600:.2f} hours")
print(f"   ✓ Embedding matrix shape: {embeddings_matrix.shape}")


# ==========================
# 4. Save outputs
# ==========================

print("\n4. Saving outputs...")

# Main embeddings array
np.save(output_path / "esm2_embeddings.npy", embeddings_matrix)

# Gene/protein IDs in the same order as rows of the embedding matrix
pd.DataFrame({"gene_id": all_gene_ids}).to_csv(
    output_path / "gene_ids.csv", index=False
)

print(f"   ✓ Saved all outputs to {output_path}/")
print("   Files:")
print("     - esm2_embeddings.npy")
print("     - gene_ids.csv")