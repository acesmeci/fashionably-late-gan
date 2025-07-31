# 🧵 Fashionably Late: Conditional GANs on Fashion-MNIST

This demo was created for the *Understanding Deep Learning* seminar at the Institute of Cognitive Science, Osnabrück University. It explores why **GANs are difficult to train** — and how classic stabilization tricks improve training dynamics and output quality. The project uses **Conditional GANs (cGANs)** trained on **Fashion-MNIST** and compares both **MLP** and **CNN (DCGAN)** architectures.

---

## 📦 Quick Start

### 🔧 Setup
Open a terminal and run the following to set up your environment:

```bash
git clone https://github.com/acesmeci/fashionably-late-gan.git
cd fashionably-late-gan
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 🚀 Training a Model

Each model has its own training script under `train_configs/`.
Run the following training scripts from your terminal. Each model saves generated `.png` images under `samples/`.

```bash
python -m train_configs.train_mlp_naive
python -m train_configs.train_mlp_stable
python -m train_configs.train_mlp_tricks
python -m train_configs.train_dcgan
```

This saves generated `.png` images under:

```bash
samples/mlp_naive/epoch_1.png
samples/mlp_naive/epoch_2.png
...
```

Model descriptions:

| Command | Description |
| --- | --- |
| `train_mlp_naive.py` | Basic MLP cGAN |
| `train_mlp_stable.py` | Adds BatchNorm, LeakyReLU, low-momentum Adam |
| `train_mlp_tricks.py` | Adds feature matching + minibatch discrimination |
| `train_dcgan.py` | Full Conditional DCGAN |

---

### 📊 Visualizing Results

After training, open the notebook:

```bash
visualize_results.ipynb
```

It displays side-by-side samples for each model at **Epoch 1** and **Epoch 10**, with clear headings. The notebook pulls images from each `samples/` subfolder.
**Note:** Fashion-MNIST class labels (0–9) are listed in the notebook for interpretability.

---

## 🧠 Educational Goal

This project illustrates:

- How different **GAN stabilization tricks** affect training and image quality
- That **more tricks ≠ always better** — failure cases included!
- The importance of **architecture choices** (MLP vs CNN)
- How to visually inspect and debug GAN outputs

All models are kept **CPU-friendly** and trainable in under ~10–15 minutes each.

---

## 📚 References

- Prince, S. J. (2023). Understanding deep learning. MIT press.
- Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: A Novel Image Dataset for Benchmarking Machine Learning Algorithms. *arXiv preprint arXiv:1708.07747*.
- Mirza, M., & Osindero, S. (2014). Conditional Generative Adversarial Nets. *arXiv preprint arXiv:1411.1784*.
- Radford, A., Metz, L., & Chintala, S. (2016). Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks. *ICLR*.
- Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., & Chen, X. (2016). Improved Techniques for Training GANs. *NeurIPS*.
