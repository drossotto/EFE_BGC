EFE: Exponential Family Embedding Pipeline
==========================================

Overview
--------

This module performs probabilistic modeling of Biosynthetic Gene Clusters (BGCs)
using Exponential Family Embeddings (EFE). It enables inference of novel or rare BGCs
by embedding co-occurrence patterns of biosynthetic domains.

Core Concepts
-------------

- **Input matrix**: Wide-format binary matrix (rows = BGCs, columns = PFAM domains).
- **Long-form data**: Triplet format (BGC_ID, Domain_ID, Count) for EFE training.
- **Context vectors**: Probabilistic embeddings of BGCs in a latent feature space.
- **Target embeddings**: Learnable vectors for biosynthetic features (e.g., PFAM domains).
- **Novelty scoring**: Negative log-likelihood of new BGCs under a reference GMM.

Workflow Summary
----------------

1. Preprocessing
   - Convert wide-form matrix to long-form (``process load-for-efe``)
   - Create index maps for BGCs and domains

2. Training
   - Train EFE model using long-form data (``train train-efe-model``)

3. Inference
   - Apply trained model to new BGCs (``process infer-embeddings``)

4. Extraction
   - Save embedding matrices (``process extract-embeddings``)

5. Scoring

   - Calculate GMM-based novelty (``calculate calculate-gmm-novelty``)
   - Compute embedding-space dissimilarity (``calculate calculate-dissimilarity``)

Input/Output Formats
--------------------

- **Input TSV matrix**: BGC-feature matrix with binary or integer values.
- **Index maps**: JSON files mapping BGC/domain names to indices.
- **Embeddings**: TSV files with BGC ID and embedding vector columns.
- **Trained model**: PyTorch `.pt` file with serialized model state.