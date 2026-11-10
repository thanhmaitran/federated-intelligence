# Shakespeare – Dataset Card

* **Source**: Hugging Face `flwrlabs/shakespeare`; original benchmark LEAF.
* **License**: BSD 2-Clause (LEAF).
* **Content**: ~4.2M rows with `character_id`, `x` (80-character window), `y` (next character); one `train` split.
* **Partitioning**: natural, by `character_id` (a speaking role, not a real-world user).
* **Leakage control**: windows overlap heavily, so random row splits leak. Each client's rows are split into *contiguous* train/validation/test blocks separated by a gap of ≥ 81 windows (`contiguous_split_with_gap`).
* **Metrics**: next-character accuracy, cross-entropy, perplexity.
* **Loader**: `data/loaders/shakespeare.py`; model `models/nlp/gru.py`.
* **Status**: implemented and tested on schema-identical fixtures; **real-data results are not reported** because the dataset was not downloadable in the development sandbox. Run the download script and `configs/experiments/shakespeare_fedavg.yaml` locally.
* **Limitations**: row order within a role is assumed to follow the play text; single hosted split.
