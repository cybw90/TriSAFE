## TriSAFE: A Gateway-Assisted IoT Federated Learning

**Paper Title: Transcript-Bound Verifiable Secure Aggregation with Differential Privacy and Timing Defenses for Gateway-Assisted IoT Federated Learning**

<p align="left">
  <img alt="Status" src="https://img.shields.io/badge/status-research%20artifact-blue">
  <img alt="Venue" src="https://img.shields.io/badge/IEEE%20Access-2026-00629B">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-3776AB">
  <img alt="Paillier" src="https://img.shields.io/badge/HE-python--paillier%20(phe)-orange">
</p>

TriSAFE is a **protocol composition** for federated learning across IoT / IoMT devices that simultaneously (1) keeps each device's update confidential, (2) bounds the influence of malicious participants, and (3) hides which devices participate from passive network observers — all under a **single coordinating server that holds no decryption key**, assisted by **three threshold helpers** (2-of-3 Paillier).

This repository contains the research prototype accompanying the paper. It implements the full three-layer pipeline (timing-private batching → verifiable robust screening → encrypted aggregation with distributed DP), the attack suite, and the figure-generation scripts used in the evaluation.

---

## Table of Contents

- [Key Contributions](#key-contributions)
- [Architecture & Code Map](#architecture--code-map)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data Setup](#data-setup)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Attacks](#attacks)
- [Outputs & Reproducing Figures](#outputs--reproducing-figures)
- [System & Threat Model](#system--threat-model)
- [Results Summary](#results-summary)
- [Citation](#citation)
---

## Key Contributions

1. **Transcript-bound proof-to-packed-ciphertext binding (PEP).** A folded, transcript-bound plaintext-equivalence protocol with a Schwartz–Zippel collision analysis that binds zero-knowledge-visible committed slot values to the exact packed Paillier plaintext blocks aggregated by the coordinator, under explicit no-wrap conditions, with soundness `2^(-λ) + d · 2^(-κ)`. Implemented in [`bulletproof_pep.py`](#repository-structure).

2. **Fixed-sensitivity, helper-noised encrypted release.** A *no-renormalization* weighted aggregation rule plus overflow conditions under which helper-side discrete-Gaussian noise is added before threshold decryption, preserving packed correctness while keeping the released sum's ℓ₂-sensitivity independent of post-screen acceptance outcomes. Implemented across [`phe_mechanism.py`](#repository-structure) and [`server_ops.py`](#repository-structure).

3. **End-to-end security evaluation in gateway-assisted IIoT FL.** Experiments on Edge-IIoTset, N-BaIoT, and MNIST showing the binding layer closes a concrete substitution gap, helper-noised release preserves the privacy/utility tradeoff, and fixed-window cover traffic reduces timing inference to near chance.

---

## Architecture & Code Map

The protocol is organized into three layers.

| Layer | Responsibility | Primary module(s) | Key methods / classes |
|---|---|---|---|
| **Client** | Clip, fixed-point encode, range-prove, pack, PEP, cover traffic | `worker_ops.py` | `WorkerOperations`, `_clip_gradients_global`, `_encode_fixed_point`, `_generate_bulletproof`, `_pack_for_encryption`, `_generate_pep_proof`, `prepare_cover_traffic` |
| **Layer 1** | Windowed intake, weight normalization over scheduled set, dummy/cover release | `server_ops.py` | `ServerOperations.layer1_time_sensitive_processing`, `_enforce_release_cadence`, `_emit_dummy_release` |
| **Layer 2** | Verify Bulletproof + PEP, threat scoring, acceptance | `server_ops.py` | `layer2_verifiable_validation`, `_verify_bulletproof`, `_verify_pep`, `_compute_threat_score` |
| **Layer 3** | Homomorphic sum, helper noise, threshold decrypt, RDP accounting | `server_ops.py` | `layer3_secure_aggregation`, `_add_encrypted_noise`, `_request_decryption_shares`, `RDPAccountant`, `_update_privacy_budget_rdp` |
| **Crypto core** | Threshold Paillier, packing, discrete Gaussian, apportionment, no-wrap | `phe_mechanism.py` | `ThresholdPaillier`, `EncryptedPackedValue`, `DiscreteGaussian`, `DistributedNoiseGenerator`, `ApportionmentRule`, `verify_no_wrap_condition` |
| **ZK binding** | Range proofs + plaintext-equivalence protocol | `bulletproof_pep.py` | `BulletproofRangeProof`, `PEPProtocol`, `aggregate_bulletproofs`, `generate_pedersen_parameters` |
| **Helpers** | Noise generation + transcript-bound partial decryption | `helper_service.py` | `HelperService.generate_noise_share`, `partial_decrypt` |

---

## Repository Structure

```text
TriSAFE/
├── main.py                     # Entry point: TriSAFESystem + CLI, orchestrates rounds
├── config.py                   # GlobalConfig dataclass + default/test/production factories
│
├── worker_ops.py               # Client side: clipping, encoding, Bulletproof, packing, PEP, cover traffic
├── server_ops.py               # Coordinator: Layer 1/2/3 + RDPAccountant
├── helper_service.py           # Threshold helper: noise share + partial decryption
│
├── phe_mechanism.py            # Threshold Paillier, discrete Gaussian, packing, apportionment, no-wrap checks
├── bulletproof_pep.py          # Bulletproof range proofs + Plaintext-Equivalence Protocol (PEP)
│
├── attack_trainer.py           # Attack suite: HIDRA Byzantine, label-flip, noise, time-delay
├── model_training.py           # Model (TinyMLNetwork) + ModelTrainer / FederatedTrainer
├── data_loader.py              # Unified multi-dataset loader (edge_iiot / nbaiot / mnist / cifar10), IID / non-IID
│
├── results_final.py            # Figure generation (full suite incl. FANG)
├── results_100.py              # Figure generation (N=100)
├── results_100_All.py          # Figure generation (N=100 incl. FANG)
├── mnist_comparison_final.py   # MNIST vs. MODEL baseline comparison figures
│
├── requirements.txt            # Python dependencies
│
├── data/                       # CSVs (LP.csv, extracted_features.csv) and torchvision downloads
├── experiments/<name>_<ts>/    # Per-run outputs: config.json, metadata.json, checkpoints, final_results.json
└── logs/                       # Run logs
```

---

## Installation

> **Heads-up:** `requirements.txt` pins the full stack — `phe`, `numpy`, `pandas`, `scikit-learn`, `scipy`, `torch`, `torchvision`, `tqdm`, `matplotlib`, and `seaborn`. `torchvision` is needed only for the MNIST / CIFAR-10 datasets.

```bash
# Clone
git clone https://github.com/cybw90/TriSAFE.git
cd TriSAFE

# (Recommended) virtual environment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Core cryptography uses **`phe` (python-paillier) 1.5.0** for additively-homomorphic encryption; threshold behavior, packing, and the PEP/Bulletproof layers are built on top in `phe_mechanism.py` and `bulletproof_pep.py`.

> The prototype is CPU-friendly. `--device cuda` is available; the device defaults to CUDA if available, otherwise CPU.

---

## Data Setup

A single unified loader (`data_loader.py`) handles all three datasets; select with `--dataset`. It imputes missing values, scales features, and partitions across workers (IID or Dirichlet non-IID) through the same pipeline.

| `--dataset` | Source | Input | Notes |
|---|---|---|---|
| `edge_iiot` | Edge-IIoTset CSV | `LP.csv` (binary `Attack_label`) | classes balanced deterministically |
| `nbaiot` | N-BaIoT CSV | `extracted_features.csv` | label column auto-detected (else last column); string labels factorized |
| `mnist` | torchvision | auto-download → `data_path` | flattened to 784-d vectors |
| `cifar10` | torchvision | auto-download → `data_path` | flattened to 3072-d vectors |

CSV datasets are searched for in `config.data_path` (file or directory) and then the current working directory:

```text
TriSAFE/
├── LP.csv                   # Edge-IIoTset  (--dataset edge_iiot)
├── extracted_features.csv   # N-BaIoT       (--dataset nbaiot)
└── main.py                  # MNIST/CIFAR-10 download automatically
```

If a required CSV is missing, the loader raises a `FileNotFoundError` listing the paths it searched. Image datasets download on first run.

> All datasets are publicly available and cited in the paper.
---

## Quick Start

```bash
# Smoke test: small, fast configuration (10 workers, 10 rounds, reduced crypto)
python main.py --mode test --dataset edge_iiot --distribution iid

# Default run (100 workers, 100 rounds) on Edge-IIoTset, IID, no attack
python main.py --mode default --dataset edge_iiot --distribution iid \
  --experiment_name trisafe_baseline

# Other datasets (same pipeline, one flag)
python main.py --mode default --dataset nbaiot --distribution iid   # needs extracted_features.csv
python main.py --mode default --dataset mnist  --distribution iid   # auto-downloads via torchvision

# Under an attack (fraction = β)
python main.py --mode default --dataset edge_iiot \
  --attack_type noise --attack_fraction 0.2

# Non-IID stress test
python main.py --mode default --dataset edge_iiot --distribution non_iid

# Resume / override from a saved config file
python main.py --config experiments/trisafe_20251001_144819/config.json
```

Results land in `experiments/<experiment_name>_<timestamp>/` and logs in `logs/`.

---

## Configuration

`config.py` defines `GlobalConfig` (a dataclass) with three presets: `create_default_config`, `create_test_config`, and `create_production_config`. Selected knobs and their **code defaults**:

| Field | Meaning | Default (code) |
|---|---|---|
| `num_workers`, `num_rounds` | Population / rounds | `100`, `100` |
| `threshold_t`, `threshold_n` | t-of-n threshold | `2`, `3` |
| `paillier_modulus_bits` | Paillier modulus | `3072` (test: 1024, production: 4096) |
| `slots_per_ciphertext` (L) | Packed slots per block | `64` |
| `packing_base_exp` (b) | Packing base exponent (B = 2^b) | `29` |
| `fixed_point_scale_exp` / `weight_scale_exp` | log2 of S_fp / S_α | `16` / `16` |
| `folding_weight_bits` (κ) | PEP folding weight bits | `32` |
| `security_margin_bits` (λ) | Security margin | `128` |
| `packing_overflow_prob_exp` | log2(δ_wrap) | `-80` |
| `max_grad_norm` (C) | Global clipping bound | `1.0` |
| `privacy_budget` (ε), `delta` (δ) | DP budget | `10.0`, `1e-5` |
| `noise_multiplier` (σ) | Gaussian noise multiplier | `0.1` |
| `time_window` | Submission window | `300.0` |
| `cover_traffic_ratio` (ρ) | Cover traffic | `0.5` |
| `dropout_tolerance` (drop) | Dropout gate | `0.3` |
| `development_skip_crypto` | Skip expensive crypto in dev | `True` |
| `production_mode` | Production safety checks | `False` |

> **Code defaults vs. paper Table 6.** Some defaults are development placeholders and differ from the reported experimental configuration in the paper (e.g., paper uses `b=42`, `κ=128`, `win=30 s`, `ρ=1.0`, and `σ_real=0.32` → noise multiplier `1.6` over `R=200`). Set these explicitly (via a JSON config or CLI overrides) to reproduce the reported runs.

A saved run's exact configuration is always written to `experiments/<run>/config.json`, so any result is reproducible from its own config.

---

## Attacks

Implemented in `attack_trainer.py` (`AttackTrainer`, `VALID_ATTACK_TYPES = ['byzantine', 'label_flip', 'gradient_inversion', 'noise', 'time_delay']`):

| Attack | Method | Notes |
|---|---|---|
| **Byzantine (HIDRA)** | `_apply_byzantine_attack` | High-dimensional perturbation; uses paper parameters (ε=0.2, k=√20) |
| **Label-flip** | `_apply_label_flip_attack` | Source→target class flipping; disproportionately harms minority classes |
| **Noise** | `_apply_noise_attack` | Additive gradient noise |
| **Time-delay** | `_apply_time_delay_attack` | Delays submission to/after the window boundary; tests Layer-1 enforcement |
| **FANG / LIE** | — | Adaptive optimization-based / evasive; reproduced via figure scripts from logged runs (paper §VII, §VII-I) |

**Metrics** (computed in `server_ops.py` / results scripts): test accuracy, **ASR** (attack success rate, paired noise seeds isolate the attack from DP variance), timing **AUC** (passive membership inference), per-round overhead, and the realized `(ε, δ)` from the `RDPAccountant`.

> The `main.py` CLI (`--attack_type`) selects the gradient/timing-space attacks `sign_flip`, `noise`, `byzantine`, and `time_delay`, applied per malicious worker. The broader suite above (HIDRA Byzantine, label-flip, FANG/LIE) is driven through `attack_trainer.py` / the `FederatedTrainer` pipeline and the figure scripts.

---

## Outputs & Reproducing Figures

Each run writes to `experiments/<experiment_name>_<timestamp>/`:

```text
experiments/trisafe_20251001_144819/
├── config.json            # full GlobalConfig snapshot
├── metadata.json          # run metadata (workers, rounds, dataset, attack, device)
├── checkpoint_round_*.pt  # periodic checkpoints
├── best_model.pt          # best validation model
└── final_results.json     # final_round, best_validation_accuracy, test_metrics,
                           # privacy_budget_used, total_rounds, data_dimensions,
                           # configuration, attack_config, metrics_history
```

Publication figures are regenerated from logged results:

```bash
python results_100.py              # core N=100 figures
python results_100_All.py          # N=100 incl. FANG
python results_final.py            # full figure suite
python mnist_comparison_final.py   # TriSAFE vs. MODEL baselines (MNIST)
```

> These scripts require `matplotlib` and `seaborn` and read from the logged experiment outputs / embedded result tables.

---

## System & Threat Model

- **Topology:** `N` IoT/IoMT devices, **1 coordinator (no decryption share)**, **3 threshold helpers** in a 2-of-3 Paillier configuration.
- **Confidentiality:** no collusion between the coordinator and **two** helpers. (Coordinator + 1 helper is tolerable; coordinator + 2 helpers degrades to a trusted-curator release.)
- **Noise:** DP is calibrated to the minimum variance of **two honest helpers**; one malicious helper cannot reduce noise below target.
- **Liveness:** remains live if any one helper fails (2-of-3).
- **Integrity:** Byzantine clients tolerated up to population fraction `β < 0.5` (bounded-influence model, **not** a Krum-style breakdown threshold).
- **Network adversary:** passive observer of timing/sizes/counts; active modification handled by authenticated channels, nonces, transcript binding; jamming/DoS out of scope.

| Adversary scenario | VRS | DP | Confidentiality | Timing |
|---|---|---|---|---|
| Baseline (design target) | ✓ | ✓ | ✓ | ✓ |
| 1 helper compromised | ✓ | ✓ | ✓ | ✓ |
| Coordinator + 1 helper | ✓ | ✓\* | ✓ | ✓ |
| Coordinator + 2 helpers | ✓ | † | ✗ | ✓ |
| Malicious coordinator (drops) | ✓ | ✓‡ | ✓ | ✓ |

<sub>\* if the two non-colluding helpers add honest noise. † noise-free sum exposed; DP reduces to trusted curator. ‡ no renormalization ⇒ sensitivity unaffected.</sub>

**Deployment target:** gateway-assisted / edge-class IIoT nodes with stable power and connectivity — *not* bare MCU-class sensor motes.

---

## Citation

If you use TriSAFE in your research, please cite:

```bibtex
@article{shah2026trisafe,
  author={Shah, Sajjad H. and Walker, Ian and Borowczak, Mike},
  journal={IEEE Access}, 
  title={TriSAFE: Transcript-Bound Verifiable Secure Aggregation With Differential Privacy and Timing Defenses for Gateway-Assisted IoT Federated Learning}, 
  year={2026},
  volume={14},
  number={},
  pages={89354-89379},
  keywords={Federated learning;Internet of Things;Labeling;Licenses;Modeling;Noise;Nuclear facility regulation;Privacy;Timing;Aggregates;Federated learning;Internet of Things;secure aggregation;differential privacy;zero-knowledge proofs;homomorphic encryption;verifiable computation},
  doi={10.1109/ACCESS.2026.3700048}}
```
---

## Acknowledgments

Artifacts (documented codebase, implementation specifications, library versions, default hyperparameters, baseline / privacy-sweep / attack-evaluation scripts, expected outputs, and acceptable variance ranges) are released with this repository to support reproduction. 

**Affiliations:** University of Wyoming (EECS) — Sajjad H. Shah, Ian Walker; University of Central Florida (ECE) — Mike Borowczak.
