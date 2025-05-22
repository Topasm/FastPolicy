# 🤗 LeRobot Installation Guide

Follow these steps to install and set up 🤗 LeRobot.

---

## Repository

Clone the repository:

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
```

---

## Installation Steps

### 1. Create and Activate a Virtual Environment

Use Python 3.10 with (Mini)conda:

```bash
conda create -y -n lerobot python=3.10
conda activate lerobot
```

---

### 2. Install FFmpeg

#### Option 1: Recommended (via Conda)

```bash
conda install ffmpeg -c conda-forge
```

#### Option 2: Specific Version

```bash
conda install ffmpeg=7.1.1 -c conda-forge
```

---

### 3. Install 🤗 LeRobot

Install the package in editable mode:

```bash
pip install -e .
```

---

### 4. Install Optional Environments

To install simulation environments like `aloha`, `xarm`, or `pusht`:

```bash
pip install -e ".[aloha, pusht]"
```

Replace `aloha, pusht` with the environments you need.

---


python train_eval/train_modernbert_critic.py \
    --noise_schedule=cosine \
    --initial_noise_multiplier=3.0 \
    --final_noise_multiplier=0.1 \
    --steps=20000