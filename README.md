# 🎩 Fashionably Late: Conditional GANs on Fashion-MNIST

This demo explores why GANs are hard to train — and how various stabilization techniques improve their performance.

## 🧠 Demo Overview

We train a Conditional GAN (cGAN) on the Fashion-MNIST dataset using an MLP architecture. Our goal is not just to generate images, but to **illustrate GAN failure modes** and how to address them.

## 💡 Variants Implemented

| Model                | Trick Used              | Notes |
|----------------------|--------------------------|-------|
| `train_baseline.py`        | None (standard GAN loss)     | Shows mode collapse, blurry outputs |
| `train_label_smooth.py`    | Label Smoothing             | Slight improvement, but limited effect |
| `train_feature_match.py`   | Feature Matching (Salimans) | More stable training and class structure |
| `train_minibatch_disc.py`  | Minibatch Stddev Trick      | Prevents collapse, enhances diversity |

## 🏁 How to Run

```bash
pip install -r requirements.txt
python -m train_configs.train_baseline  # or any other variant
