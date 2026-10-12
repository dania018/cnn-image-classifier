# CIFAR-10 Image Classification with a CNN

A Convolutional Neural Network built from scratch in PyTorch that classifies 32×32 colour images into 10 categories.

## Results

| Metric | Score |
|---|---|
| Test accuracy | **74.71%** |
| Final training loss | 0.113 |
| Trainable parameters | ~620K |

## Dataset

**[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)**, loaded with `torchvision.datasets.CIFAR10`

- 60,000 colour images: 50,000 for training and 10,000 for testing
- Each image is **32 × 32 × 3** (width × height × RGB channels)
- 10 classes: airplane · automobile · bird · cat · deer · dog · frog · horse · ship · truck

## Preprocessing

```python
transform = transforms.Compose([
    transforms.ToTensor(),                                   # pixels 0–255 → tensor in [0, 1]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))   # [0, 1] → [-1, 1] per channel
])
```

- `ToTensor()` converts each image to a PyTorch tensor (C × H × W) and scales the pixels to **[0, 1]**.
- `Normalize(mean, std)` applies `(x − 0.5) / 0.5` to each RGB channel, which shifts the values to **[−1, 1]**. Centred inputs help the network train faster and more stably.

## Model Architecture

```
Input (3×32×32)
 → [Conv 3→32  + ReLU → MaxPool]  → 32×16×16
 → [Conv 32→64 + ReLU → MaxPool]  → 64×8×8
 → [Conv 64→128 + ReLU → MaxPool] → 128×4×4
 → Flatten (2048)
 → Linear(2048, 256) + ReLU
 → Linear(256, 10) → class scores
```

### Design choices

- **Conv layers:** `kernel_size=3`, `padding=1`. The padding keeps the width and height the same after each convolution.
- **MaxPool:** `kernel_size=2`, `stride=2`. Each pool halves the width and height.
- **Filters double each block (32 → 64 → 128).** As the image gets spatially smaller, the network learns more and more feature maps.
- **No softmax in the model.** `nn.CrossEntropyLoss` applies log-softmax internally, so the last layer outputs raw scores (logits) through a linear layer.

## Training

| Setting | Value |
|---|---|
| Loss | `CrossEntropyLoss` (multi-class) |
| Optimizer | Adam (default lr = 0.001) |
| Epochs | 10 |
| Batch size | 64 (training data shuffled) |

The training loss fell steadily from **0.756 → 0.113** over 10 epochs.

## Evaluation

Predictions are made with `torch.max(outputs, 1)`, which takes the class with the highest score, and compared against the true labels on the 10,000 test images:

```
accuracy = 74.71%
```
