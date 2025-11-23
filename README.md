1)**Overview**

This project implements a lightweight Neural Architecture Search (NAS) framework using a Genetic Algorithm (GA). The system evolves CNN architectures for CIFAR-like image classification.

The GA performs the following operations:

Population initialization

Fitness evaluation (training + validation)

Roulette-Wheel (fitness-proportional) Selection (modified as per assignment requirement)

Crossover between candidate architectures

Mutation of architecture genes

Elitism for preserving top individuals

Logging into nas_run.log

Generation-wise snapshots into generation_<n>.jsonl

All results are written to:

AI/A2-16-11/nas-ga-basic-main/outputs/run_<n>/

2. **Modification Implemented (Q1A Requirement)**

The assignment required replacing the existing Tournament Selection with Roulette-Wheel Selection.

Changes made in model_ga.py:

The selection() function was rewritten to implement roulette-wheel selection.

Fitness values are clipped to non-negative values.

If all fitness values are zero, the algorithm falls back to uniform random sampling to avoid division by zero.

Selection uses:

random.choices(population, weights=fitness_scores, k=population_size)


Selected individuals are deep-copied to prevent unintended mutation propagation.

The run log now explicitly prints:

Performing roulette-wheel (fitness-proportional) selection...


No other part of the GA (fitness computation, mutation, crossover, or architecture encoding) was modified.

3. **How to Run the Project (WSL2 + CUDA Recommended)**
Prerequisites

Windows 11

WSL2 (Ubuntu 22.04)

NVIDIA GPU with WSL CUDA support (nvidia-smi must work inside WSL)

Miniconda installed in WSL

PyTorch with CUDA support (Nightly required for RTX 50-series GPUs)

Steps
Step 1 — Navigate to project directory
wsl -d Ubuntu-22.04 -- bash -lc "source ~/miniconda/bin/activate torchgpu; cd '/mnt/c/<YOUR_PATH>/nas-ga-basic-main'"

Step 2 — Install PyTorch Nightly (if required)
pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu130 torch torchvision torchaudio

Step 3 — Quick smoke test (fast)
export SMOKE=1
export EPOCHS=1
python nas_run.py

Step 4 — Full NAS run
unset SMOKE
unset EPOCHS
python nas_run.py

4. **Monitoring Progress**
Log files
outputs/run_<n>/nas_run.log

Monitor logs live
tail -n 200 -f outputs/run_1/nas_run.log

Monitor GPU usage
nvidia-smi -l 2

5. **Environment Used**
| Component     | Version                       |
| ------------- | ----------------------------- |
| OS            | WSL2 Ubuntu 22.04             |
| NVIDIA Driver | 580.88                        |
| Python        | 3.10 (conda env: `torchgpu`)  |
| PyTorch       | Nightly build (CUDA 13.0)     |
| GPU           | RTX series (sm_120 supported) |

6. **Results (Run-1 and Run-2)**

The following results come directly from the logs generated during execution and represent the true performance of the evolved CNN architectures.

**Run-1 Summary**

Location: outputs/run_1/
Device: CUDA
Generations: 5
Population: 10

Best accuracy per generation
Generation	Accuracy
| Generation | Accuracy   |
| ---------- | ---------- |
| G1         | 0.7020     |
| G2         | 0.7120     |
| G3         | 0.7080     |
| G4         | **0.7250** |
| G5         | 0.7060     |

Final Best Architecture (Run-1)
{
 "num_conv": 4,
 "conv_configs": [
   {"filters":128, "kernel_size":5},
   {"filters":128, "kernel_size":3},
   {"filters":128, "kernel_size":7},
   {"filters":16,  "kernel_size":5}
 ],
 "pool_type": "max",
 "activation": "relu",
 "fc_units": 64
}

Final Metrics (Run-1)
Metric	Value
Accuracy	0.7250
Fitness	0.7129
Parameters	1,073,900

**Run-2 Summary**

Location: outputs/run_2/
Device: CUDA

Best accuracy per generation
Generation	Accuracy

Final Best Architecture (Run-2)
{
 "num_conv": 4,
 "conv_configs": [
   {"filters":128, "kernel_size":3},
   {"filters":16,  "kernel_size":5},
   {"filters":64,  "kernel_size":7},
   {"filters":16,  "kernel_size":5}
 ],
 "pool_type": "avg",
 "activation": "leaky_relu",
 "fc_units": 64
}

Final Metrics (Run-2)
Metric	Value
Accuracy	0.7240
Fitness	0.7094
Parameters	205,120
7. **Output Files Created**

Each run generates:

generation_0.jsonl

generation_1.jsonl

generation_2.jsonl

generation_3.jsonl

generation_4.jsonl

nas_run.log

best_arch.pkl / best_arch.json

Each JSONL file contains 10 entries of the form:

{"id": 1, "genes": {...}, "fitness": ..., "accuracy": ..., "total_params": ...}

8. **Troubleshooting**
Issue	Cause	Resolution
CUDA not detected	PyTorch mismatch	Reinstall PyTorch Nightly
CIFAR download failed	WSL network issues	Test internet inside WSL
Slow training	CPU fallback	Ensure torch.cuda.is_available() is TRUE
Generated: November 2025