
# LGG-clustering

Edit **esm2_embedding.py** with own file paths, then run/submit to HPC. This will compute ESM‑2 (esm2_t36_3B_UR50D) embeddings for each sequence and save **esm2_embeddings.npy** and **gene_ids.csv** in the specified output directory.
This script can run on CPU or GPU. If a CUDA device is available, run on GPU and will be converted to half precision (model.half()), which reduces memory usage.
