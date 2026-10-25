# Data provenance and terms

The prepared NumPy files in data/processed are a small, transformed subset of
the public Yandex Yambda-50M sequential listening dataset. They retain anonymous
user and track identifiers, causal contexts, completion labels, and timestamps.

- Original source: https://huggingface.co/datasets/yandex/yambda
- Original file: sequential/50m/listens.parquet
- Source repository revision observed on 2026-10-25: dd6f3a19eef5866e346c3270e098baa641a44948
- Original file SHA-256: c07db8326739b23fd4fb0a25037304e78b2e916320ccf1b01dc904d2b1b376c7
- Original license declared by the dataset card: Apache License, Version 2.0.
- The authors state that the dataset is published for scientific and research purposes.
- Full license text: licenses/DATASET_APACHE_2_0.txt.

The modifications are chronological splitting, training-only catalog selection,
mapping IDs to tokens (with PAD/OOV), sampling causal contexts and held-out queries,
and capping the completion target at one. The full source file is not included.
No ownership of the original data is claimed. The MIT license in LICENSE applies
to the new experiment code, not to the original dataset or third-party libraries.
