# Optimising Reinforcement Learning methods for QoS aware Energy-Efficient Cell Switch-Off in Mobile Networks

MSc thesis project (UCL, ELEC0054 — Integrated Machine Learning Systems)

## Overview

This project studies **Cell Switch-Off (CSO)** — dynamically turning cells in a radio access network ON/OFF to save energy — through the lens of **offline reinforcement learning**. Five algorithms are trained and compared on a simulated 46-cell heterogeneous network (7 macro, 39 micro) built from real OpenCellID sites:

- **DQN** — online RL baseline
- **DQN + QoS Penalty** — DQN with three penalty terms (blocked switch-off attempts, overload-scaled penalty, flip-flop penalty)
- **CQL** — offline RL
- **IQL** — offline RL
- **QR-CQL** — novel distributional offline RL variant combining CQL's conservatism penalty with a quantile-regression Q-function (this project's primary contribution)

The central question the thesis asks is not just "how much energy can be saved," but **how much energy-saving efficiency can be retained while minimising QoS violation**.


.
├── .vscode/                                # IDE workspace settings (ignored)
│
├── Datasets/                               # RAW & INTERMEDIATE DATA + EDA notebooks
│   ├── Base/                               # Base station traffic metrics (per-cell)
│   │   ├── cell_assignment_log.csv         # File with all the traffic assignments to each cell
│   │   ├── cell_type.ipynb                 # Labels cells based on traffic
│   │   ├── cell_type_labelled_data_MR.csv  # MR labelled data
│   │   ├── cell_type_labelled_data_PRB.csv # PRB utilisation labelled data
│   │   ├── processed_cell_MR.csv           # Cleaned MR data for simulation
│   │   ├── processed_cell_PRB.csv          # Cleaned PRB data for simulation
│   │   ├── simulator_ready_traffic_MR.csv  # Final formatted MR traffic
│   │   ├── simulator_ready_traffic_PRB.csv # Final formatted PRB traffic
│   │   ├── traffic_DLPRB.csv               # Downlink PRB timeseries
│   │   ├── traffic_DLThpTime.csv           # Downlink throughput (time)
│   │   ├── traffic_DLThpVol.csv            # Downlink throughput (volume)
│   │   └── traffic_MR_number.csv           # Mobility report counts
│   │
│   ├── Opencellid/                         # Geographic & topology data (by MNC)
│   │   ├── 234.csv                         # MNC 234 topology
│   │   ├── 730.csv                         # MNC 730 topology
│   │   ├── 748.csv                         # MNC 748 topology
│   │   ├── combined_topology.csv           # Merged topology
│   │   ├── final_topology.csv              # Final cleaned topology
│   │   └── only_UCL_cells.csv              # Filtered UCL (unique cell list)
│   │
│   ├── cellidentification.ipynb            # Notebook: identify cells around UCL from data
│   ├── main_traffic_mapper.ipynb           # Notebook: map traffic to cells
│   ├── temporal.ipynb                      # Notebook: per-cell ON/OFF & PRB load over eval week
│   ├── trafficdata.ipynb                   # Notebook: traffic data EDA & visualisation
│   ├── topology_map.html                   # Interactive topology visualisation
│   ├── filtered_topology_map.html          # Topology map (filtered view)
│   ├── final_topology_map.html             # Topology map (final version)
│   └── final_topology_map_macro_blac...    # (truncated name – macro/black version)
│
├── dataset/                                # FINAL OFFLINE RL DATASET (generated)
│   └── offline_dataset_stats.json          # Summary stats (size, rewards, action dims)
│
├── models/                                 # Saved model checkpoints (DQN, CQL, IQL, QR-CQL)
├── results/                                # Experiment outputs (metrics, scores, CSV logs)
├── logs/                                   # Training logs (TensorBoard / stdout)
│
├── simulator.py                            # Custom Gymnasium Env (RANEnv) – state/action/reward
├── policies.py                             # Behaviour policies (random, heuristic, etc.) used for data collection
├── generate_dataset.py                     # Runs behaviour policy on simulator → creates offline dataset
├── generate_config.py                      # Helper script to create/modify network_config.json
├── main.py                                 # Master entrypoint for experiment orchestration
├── dqn_wrapper.py                          # Wrapper to integrate SB3 DQN with the custom env
├── train_dqn.py                            # Online DQN training (Stable-Baselines3)
├── train_cql.py                            # Offline CQL training (d3rlpy)
├── train_iql.py                            # Offline IQL training (d3rlpy)
├── train_qr_cql.py                         # Offline QR-CQL (Quantile Regression) training (d3rlpy)
├── plot_results.py                         # Aggregates runs & plots performance comparisons
├── temporal_analysis.ipynb                 # Post-hoc temporal evaluation (loading saved models)
├── network_config.json                     # RAN environment parameters (cells, bandwidth, etc.)
├── .gitignore
├── LICENSE.md
└── README.md

## Environment & Network Setup

- **Topology:** 46 real cells (7 macro, 39 micro) sourced from OpenCellID, with macro-to-micro parent assignment done manually via a Folium map and lat/lon bounding box (no optimisation algorithm used).
- **Traffic profiles:** Each cell is manually assigned to one of five archetypes — Rush, Shop, Transport, House, Hospital — via Google Maps POI inspection (`domain_map` dict). Profiles are sampled (seeded, without replacement) from matching-labelled columns of the Zindi Spatio-Temporal Beam-Level Traffic Forecasting dataset, then min-max scaled per-cell to [0, 1].
- **Key constants:**
  - `MICRO_CSO_LIMIT = 0.3`
  - `MACRO_CAPACITY_LIMIT = 0.8`
  - `MACRO_OVERLOAD_THRESHOLD = 0.75`
- **Power model:** Auer et al. (2011) EARTH linear power model (`auer_power()`), with an Al-Tahmeesschi-calibrated variant used for a sensitivity check.
- **Offline dataset:** 125 behaviour policies generating ~517k+ transitions.
- **Evaluation window:** timesteps 664–829 (held-out week).
- **Training budget:** all models standardised to 800,000 steps (with a 400k extended run confirming 250k as the optimal checkpoint for final models).

## Tech Stack

- Python, custom Gymnasium environment (`RANEnv`)
- [d3rlpy](https://github.com/takuseno/d3rlpy) 2.8.1 — CQL, IQL, QR-CQL
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) — DQN
- Zindi Spatio-Temporal Beam-Level Traffic Forecasting dataset
- OpenCellID topology data
- Miniconda environment: `amls2` (Windows)

## Setup

```bash
conda env create -f environment.yml
conda activate ran-rl
```

## Usage

```bash
# 1. Generate the offline dataset from behaviour policies
python generate_dataset.py

# 2. Train each model
python train_dqn.py
python train_cql.py
python train_iql.py
python train_qr_cql.py

# 3. Evaluate Each model

python main.py --policy dqn --split eval
python main.py --policy cql --split eval
python main.py --policy iql --split eval
python main.py --policy qr_cql --split eval

#Before running  the evaluation, ensure that the correct model is loaded in the elif in the main() function in main.py 

# 4. Aggregate and plot results
python plot_results.py
```

Final model configurations:
- `DQN_v_final`: faster epsilon-decay, lower learning rate with negative penatly for QOS violations
- `QR-CQL_v_final`: alpha = 0.3, lower learning rate

## Reproducibility Notes

- Multi-seed validation (5 seeds) is used  for the two final locked models (`DQN_final`, `QR-CQL_v_final`). 
- Multi-seed validation (5 seeds) is done with the traffic assignment to ensure our mdoels work with different traffics as well


