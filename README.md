# 🧵 Fashionably Late: Conditional GANs on Fashion-MNIST

This demo was created for the Understanding Deep Learning seminar at the Institute of Cognitive Science, Osnabrück University. It explores why **GANs are difficult to train** — and how classic stabilization tricks improve training dynamics and output quality. The project uses **Conditional GANs (cGANs)** trained on **Fashion-MNIST** and compares both **MLP** and **CNN (DCGAN)** architectures.

---

## 📦 Quick Start

### 🔧 Setup

```bash
git clone https://github.com/acesmeci/fashionably-late-gan.git
cd fashionably-late-gan
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 🚀 Training a Model

Each model has its own training script under `train_configs/`. For example:

```bash
python -m train_configs.train_baseline
```

This saves generated `.png` images under:

```bash
samples/baseline/epoch_1.png
samples/baseline/epoch_2.png
...
```

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
visualize_results.ipynb
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

## 📚 References

- Prince, S. J. (2023). Understanding deep learning. MIT press.
- Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: A Novel Image Dataset for Benchmarking Machine Learning Algorithms. *arXiv preprint arXiv:1708.07747*.
- Mirza, M., & Osindero, S. (2014). Conditional Generative Adversarial Nets. *arXiv preprint arXiv:1411.1784*.
- Radford, A., Metz, L., & Chintala, S. (2016). Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks. *ICLR*.
- Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., & Chen, X. (2016). Improved Techniques for Training GANs. *NeurIPS*.
