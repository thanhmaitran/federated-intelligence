# FEMNIST – Dataset Card

* **Source**: Hugging Face `flwrlabs/femnist`; original benchmark LEAF (Caldas et al., 2018). 
* **License**: BSD 2-Clause (LEAF).
* **Content**: ~814K 28×28 grayscale images, 62 classes, `writer_id` (natural client), `hsf_id`, `character`. Hosted version has one `train` split.
* **Partitioning**: natural, by `writer_id`.
* **Splits (this project)**: writers are divided into *seen* writers (federation; per-writer 70/15/15 sample split) and *unseen* writers (held out entirely). Two evaluations are reported separately: held-out samples of seen writers, and generalisation to unseen writers.
* **Preprocessing**: pixels scaled to [0, 1]; no statistics fitted on test data.
* **Loader**: `data/loaders/femnist.py`; download with `scripts/download_datasets.py` (records source, licence, row count, access date and schema in `data/metadata/femnist.json`).
* **Status in this repository**: loader, CNN and config are implemented and tested against schema-identical *fixtures*. **The real dataset could not be downloaded in the development sandbox (huggingface.co blocked), so no FEMNIST results are reported.** Run `python scripts/download_datasets.py` then `python scripts/run_experiment.py --config configs/experiments/femnist_fedavg.yaml` to produce them.
* **Limitations**: writer ≠ guaranteed independent individual; hosted split policy may differ from the LEAF sampled subsets.
