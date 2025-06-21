# 🧵 Fashionably Late: Conditional GANs on Fashion-MNIST

This demo explores why **GANs are difficult to train** — and how classic stabilization tricks improve training dynamics and output quality. The project uses **Conditional GANs (cGANs)** trained on **Fashion-MNIST** and compares both **MLP** and **CNN (DCGAN)** architectures.

---

## 📦 Quick Start

### 🔧 Setup

```bash
git clone https://github.com/yourusername/fashionably-late-gan.git
cd fashionably-late-gan
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 🚀 Training a Model

Each model has its own training script under `train_configs/`. For example:

This saves generated `.png` images under:

Run any of the following:

| Command | Description |
| --- | --- |
| `train_baseline.py` | Basic MLP cGAN |
| `train_label_smooth.py` | Adds label smoothing |
| `train_feature_match.py` | Adds feature matching (Salimans et al.) |
| `train_minibatch_disc.py` | Adds minibatch discrimination trick |
| `train_cnn.py` | Full Conditional DCGAN |

---

### 📊 Visualizing Results

After training, open the notebook:

```bash
bashvisualize_results.ipynb
```

This will show side-by-side comparisons for each model’s output at **Epoch 1 and Epoch 10**. The notebook pulls images from each `samples/` subfolder.

---

## 🧠 Educational Goal

This project illustrates:

- How **different GAN tricks** affect learning dynamics
- How **model capacity** (MLP vs. CNN) influences visual quality
- Where GAN training fails — and how to **visually debug it**

All models are kept **CPU-friendly** and trainable in under ~10–15 minutes each.

---

## 🧵 Credits

Part of the *Understanding Deep Learning* seminar at Osnabrück University  
Developed by the **Fashionably Late** group.