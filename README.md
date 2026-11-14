# Peer-Assisted Dissemination of Incremental Model Updates over Constrained IoT Networks

This repository contains the implementation and experimental code accompanying our research on incremental machine-learning model updates in resource-constrained IoT networks.

The project is organized into two main parts:

| Directory | Description |
| --- | --- |
| [`ml/`](ml/) | Machine-learning experiments and model-update preparation. **Documentation to be completed by the ML contributor.** |
| [`simulator/`](simulator/) | Network simulations of the proposed peer-assisted dissemination protocol and a Deluge-based baseline, including radio configurations, testbeds, and energy-report utilities. |

## Machine learning (`ml/`)

> **Documentation placeholder — to be completed by the ML contributor.**
>
> Describe the datasets, model architectures, training and update-generation workflow, dependencies, and commands for reproducing the ML experiments here. See `ml/README.md` when available.

## Network simulation (`simulator/`)

The simulation code evaluates the delivery of model updates across constrained wireless networks using different radio technologies and channel-access settings. It includes two independently runnable implementations:

- **Proposed protocol:** peer-assisted dissemination using advertisement, update requests, data transfer, missing-chunk reporting, and repair.
- **Deluge baseline:** page-based dissemination using advertisements, requests, and data transfers.

Each implementation has its **own configuration files, testbeds, batch runner, output directory, and energy-reporting utilities**. No shared testbed directory or combined campaign runner is required.

For setup, individual runs, batch experiments, and output descriptions, see the [simulator README](simulator/README.md).

