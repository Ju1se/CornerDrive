"""CIFAR-10 dual-dataset rarity benchmark for CornerDrive.

A self-contained GPU benchmark (torch + numpy only) that re-runs the thesis's
central test on a harder public dataset:

    an uploaded client update is applied to a replica of the global model and
    its loss drift is measured on two server-held golden sets, a normal set
    D_main and a rare set D_corner. The pair (dL_main, dL_corner) decides
    whether the update is Fraud, Rarity, Honest or Noise.

Design and results: docs/benchmarks/CIFAR10_DUAL_DATASET_BENCHMARK.md

Modules
-------
cifar_golden   golden-dataset annotation and the long-tailed client federation
batched_cnn    model A: one CNN, many clients, a single grouped-convolution pass
batched_head   model B: an MLP head on a frozen embedding model
backend        the two model back-ends and the loops they share
detectors      L1 routing signals, the L2 dual verdict (thesis Table 4.2) and
               the embedding-space read-out
aggregators    FedAvg, GeoMed, Multi-Krum, FLTrust, Zeno, Zeno++, FedVal
engine         the federated round loop and per-update logging
analysis       unit files to tables
"""
