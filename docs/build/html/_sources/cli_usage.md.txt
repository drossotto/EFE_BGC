# CLI

Perform Exponential Feature Embedding on Biosynthetic Gene Clusters (BGCs)

**Usage**:

```console
$ [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.

**Commands**:

* `process`: EFE preparation and processing steps
* `calculate`: Calculations for EFE-based metrics
* `train`: Train EFE model on long-form DataFrame

## `process`

EFE preparation and processing steps

**Usage**:

```console
$ process [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `load-for-efe`: Load BGC matrix for EFE model input.
* `infer-embeddings`: Infer BGC embeddings using a trained EFE...
* `extract-embeddings`: Extract embeddings from BGC probabilistic...

### `process load-for-efe`

Load BGC matrix for EFE model input.

**Usage**:

```console
$ process load-for-efe [OPTIONS]
```

**Options**:

* `--input-tsv PATH`: Path to the input BGC-feature matrix  [required]
* `--output-dir PATH`: Directory to save output files  [required]
* `--help`: Show this message and exit.

### `process infer-embeddings`

Infer BGC embeddings using a trained EFE reference model.

**Usage**:

```console
$ process infer-embeddings [OPTIONS]
```

**Options**:

* `--reference-model-path PATH`: Path to the trained EFE reference model (.pt file) (e.g., MiBiG model).  [required]
* `--domain-map-path PATH`: Path to the domain index map JSON from the reference model&#x27;s data.  [required]
* `--input-bgc-map-path PATH`: Path to the BGC index map JSON for the input (experimental) data.  [required]
* `--reference-bgc-map-path PATH`: Path to the BGC index map JSON from the reference model&#x27;s training.  [required]
* `--input-matrix-path PATH`: Path to your raw input (e.g., experimental) BGC feature matrix TSV.  [required]
* `--output-path PATH`: Path to save the inferred BGC embeddings TSV.  [required]
* `--embedding-dim INTEGER`: Dimension of the embeddings (must match trained reference model).  [default: 64]
* `--help`: Show this message and exit.

### `process extract-embeddings`

Extract embeddings from BGC probabilistic EFE models

**Usage**:

```console
$ process extract-embeddings [OPTIONS]
```

**Options**:

* `--model-path PATH`: Path to the trained EFE model file  [required]
* `--bgc-map-path PATH`: Path to the BGC index map JSON  [required]
* `--domain-map-path PATH`: Path to the domain index map JSON  [required]
* `--output-path PATH`: Directory to save the extracted embeddings  [required]
* `--embedding-dim INTEGER`: Dimensionality of the embeddings  [default: 64]
* `--data-source TEXT`: Set to &#x27;bgc&#x27; to extract BGC context embeddings, or domain_, where value is 0, 85, 170, 255 for specific domain embeddings.  [default: bgc]
* `--help`: Show this message and exit.

## `calculate`

Calculations for EFE-based metrics

**Usage**:

```console
$ calculate [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `calculate-gmm-novelty`: Calculate novelty scores for experimental...
* `calculate-dissimilarity`: Calculate dissimilarity scores for BGCs...

### `calculate calculate-gmm-novelty`

Calculate novelty scores for experimental BGCs based on a GMM fitted to reference embeddings.

**Usage**:

```console
$ calculate calculate-gmm-novelty [OPTIONS]
```

**Options**:

* `--mibig-embeddings-path PATH`: Path to the MiBiG BGC embeddings TSV (the reference anchor, output from &#x27;extract-embeddings&#x27;).  [required]
* `--experimental-embeddings-path PATH`: Path to your inferred experimental BGC embeddings TSV (output from &#x27;infer-embeddings-cli&#x27;).  [required]
* `--original-experimental-matrix-path PATH`: Path to the original experimental BGC feature matrix TSV. Novelty scores will be added to this output.  [required]
* `--output-novelty-path PATH`: Path to save the augmented experimental matrix (with novelty scores) as TSV.  [required]
* `--plot-output-path PATH`: Optional: Path to save a histogram of novelty scores (e.g., .png).
* `--gmm-n-components INTEGER`: Number of components for the Gaussian Mixture Model. If not provided, it will be auto-determined using BIC/AIC.
* `--gmm-n-components-min INTEGER`: Minimum number of components to test for GMM auto-selection.  [default: 1]
* `--gmm-n-components-max INTEGER`: Maximum number of components to test for GMM auto-selection.  [default: 20]
* `--gmm-covariance-type TEXT`: Type of covariance parameters (&#x27;full&#x27;, &#x27;tied&#x27;, &#x27;diag&#x27;, &#x27;spherical&#x27;). &#x27;full&#x27; is most flexible.  [default: full]
* `--gmm-n-init INTEGER`: Number of initializations to perform for GMM. Higher is more robust but slower.  [default: 10]
* `--random-state INTEGER`: Random state for GMM reproducibility.
* `--gmm-selection-criterion TEXT`: Information criterion to use for GMM component auto-selection (&#x27;bic&#x27; or &#x27;aic&#x27;).  [default: bic]
* `--help`: Show this message and exit.

### `calculate calculate-dissimilarity`

Calculate dissimilarity scores for BGCs using a trained EFE model.

**Usage**:

```console
$ calculate calculate-dissimilarity [OPTIONS]
```

**Options**:

* `--input-tsv PATH`: Path to input BGC-feature TSV  [required]
* `--model-path PATH`: Path to trained EFE model (.pt)  [required]
* `--output-tsv PATH`: Path to save output TSV with novelty scores  [required]
* `--help`: Show this message and exit.

## `train`

Train EFE model on long-form DataFrame

**Usage**:

```console
$ train [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `train-efe-model`: Train EFE model on long-form DataFrame

### `train train-efe-model`

Train EFE model on long-form DataFrame

**Usage**:

```console
$ train train-efe-model [OPTIONS]
```

**Options**:

* `--long-df-path PATH`: Path to the long-form DataFrame TSV  [required]
* `--bgc-map-path PATH`: Path to the BGC index map JSON  [required]
* `--domain-map-path PATH`: Path to the domain index map JSON  [required]
* `--output-dir PATH`: Directory to save the model and training history  [required]
* `--embedding-dim INTEGER`: [default: 64]
* `--batch-size INTEGER`: [default: 1024]
* `--epochs INTEGER`: [default: 30]
* `--learning-rate FLOAT`: [default: 0.001]
* `--help`: Show this message and exit.

