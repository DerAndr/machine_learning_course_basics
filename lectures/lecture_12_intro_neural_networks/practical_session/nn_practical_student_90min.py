# %% [markdown]
# # Introduction to Neural Networks: Practical Session — STUDENT VERSION (90 minutes)
# 
# **Goal**: Understand what happens inside a neural network (using NumPy) and then build models for regression and classification using PyTorch.
# 
# **Learning objectives:**
# 
# - implement a forward pass and manual back-propagation for a single-layer and a three-layer network in pure NumPy;
# - contrast manual gradient computation with PyTorch autograd;
# - build and train a regression model (California Housing) and a classification model (FashionMNIST) in PyTorch;
# - apply regularisation techniques (Dropout, BatchNorm) in practice;
# - evaluate model performance and visualise errors.
# 
# **Agenda:**
# 
# | Part | Topic | Time |
# |------|-------|------|
# | 1 | The "Guts" of a Neural Network (NumPy) | ~35 min |
# | 2 | Regression with PyTorch (California Housing) | ~25 min |
# | 3 | Classification with PyTorch (FashionMNIST) | ~25 min |
# | — | Debrief | ~5 min |

# %% [markdown]
# ## Setup
# 
# For local work in this repository, prefer:
# 
# ```bash
# uv sync --group neural_networks
# uv run python tools/check_notebook_environment.py
# ```
# 
# In Google Colab the cell below detects the environment and installs any missing packages automatically.

# %%
import importlib.util
import os

IN_COLAB = "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ

required_packages = ["torch", "torchvision", "torchinfo", "sklearn", "seaborn"]
package_status = {pkg: importlib.util.find_spec(pkg) is not None for pkg in required_packages}

if IN_COLAB:
    missing = [pkg for pkg, available in package_status.items() if not available]
    if missing:
        # Map import names to pip names where they differ
        pip_names = {"sklearn": "scikit-learn"}
        to_install = [pip_names.get(p, p) for p in missing]
        print(f"Colab detected: installing {', '.join(to_install)}")
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *to_install])
    else:
        print("Colab detected: all required packages are already available.")
else:
    missing = [pkg for pkg, available in package_status.items() if not available]
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Install locally with:  uv sync --group neural_networks")
    else:
        print("All required packages are available.")

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchinfo import summary

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sns.set_theme(
    style="whitegrid",
    context="talk",
    rc={
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "grid.color": "#DCEAF2",
        "axes.edgecolor": "#CCCCCC",
        "axes.labelcolor": "#17324D",
        "xtick.color": "#17324D",
        "ytick.color": "#17324D",
    },
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")

PLOT_COLORS = {
    "teal": "#22B8BD",
    "blue": "#149ECA",
    "orange": "#F28E2B",
    "rose": "#D1495B",
    "ink": "#17324D",
    "grid": "#DCEAF2",
    "panel": "white",
}

# NOTE: notebook magic commented for local script use: %matplotlib inline

# %%
def plot_loss_curve(losses, title="Training Loss"):
    """Plot a training loss curve."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, color=PLOT_COLORS["blue"], linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title, fontweight="bold", color=PLOT_COLORS["ink"])
    ax.grid(True, color=PLOT_COLORS["grid"])
    plt.tight_layout()
    plt.show()


def show_misclassified(images, true_labels, pred_labels, class_names, n=5):
    """Display n misclassified examples side by side."""
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 3))
    for i in range(n):
        img = images[i] / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        axes[i].imshow(np.transpose(npimg, (1, 2, 0)), cmap="gray")
        axes[i].set_title(
            f"Pred: {class_names[pred_labels[i]]}\nTrue: {class_names[true_labels[i]]}",
            fontsize=9,
        )
        axes[i].axis("off")
    plt.suptitle("Misclassified Examples", fontweight="bold", color=PLOT_COLORS["ink"])
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## How To Work In Teams
# 
# If you are working in pairs or small groups, a clean split is:
# 
# - **Group A**: Part 1 — The "Guts" (NumPy, Tasks 1.1–1.7) — ~35 min
# - **Group B**: Part 2 — Regression with PyTorch (Tasks 2.1–2.3) — ~25 min
# - **Group C**: Part 3 — Classification with PyTorch (Tasks 3.1–3.3) — ~25 min
# 
# Then regroup for the debrief. Each group presents:
# 1. what they built;
# 2. one thing that surprised them;
# 3. one question they still have.

# %% [markdown]
# ## Part 1: The "Guts" of a Neural Network (NumPy)
# 
# Before using a framework like PyTorch, let's understand the math of a single neuron.
# 
# $$z = w \cdot x + b$$
# $$y = \sigma(z)$$
# 
# ### Why not just use NumPy forever?
# 
# 1. **Autograd**: In NumPy you must manually derive and code every gradient. PyTorch does this with `loss.backward()`.
# 2. **GPU acceleration**: NumPy runs on CPU. PyTorch tensors can run on GPU (10×–100× faster for large networks).
# 3. **Pre-built layers**: `nn.Linear`, `nn.Conv2d`, etc. — no hand-typing `w * x + b` for 100 layers.

# %% [markdown]
# ### Task 1.1: Activation Functions
# 
# Implement the **sigmoid** function:
# 
# $$\sigma(x) = \frac{1}{1 + e^{-x}}$$
# 
# > **Note:** In modern deep learning, ReLU and its variants are preferred for hidden layers because they avoid the vanishing-gradient problem. We use sigmoid here because the derivative is straightforward to compute by hand — which is exactly what we need for manual back-propagation in the next tasks.

# %%
# TODO:
# 1. Implement the sigmoid function: σ(x) = 1 / (1 + e^(-x))
# Hint: use np.exp()
def sigmoid(x):
    ...


x = np.linspace(-10, 10, 100)
plt.plot(x, sigmoid(x), color=PLOT_COLORS["blue"], linewidth=2)
plt.title("Sigmoid Activation", fontweight="bold")
plt.xlabel("x")
plt.ylabel("σ(x)")
plt.grid(True, color=PLOT_COLORS["grid"])
plt.show()

# %% [markdown]
# ### Activation Function Comparison
# 
# The lecture covers several activation functions. The plot below compares the four most common ones.
# 
# Key take-aways:
# 
# - **Sigmoid** and **Tanh** saturate for large inputs — their gradients become very small (vanishing-gradient problem).
# - **ReLU** has a constant gradient of 1 for positive inputs but is zero for negative inputs (dead neurons).
# - **Leaky ReLU** fixes the dead-neuron issue with a small negative slope.
# 
# Remember: we use sigmoid in Part 1 only because its derivative $\sigma'(x) = \sigma(x)(1 - \sigma(x))$ is easy to compute by hand. In practice, prefer ReLU-family activations for hidden layers.

# %%
def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

x = np.linspace(-5, 5, 200)

fig, axes = plt.subplots(1, 4, figsize=(16, 3.5), sharey=False)

for ax, (name, fn, color) in zip(axes, [
    ("Sigmoid", sigmoid, PLOT_COLORS["blue"]),
    ("Tanh", np.tanh, PLOT_COLORS["teal"]),
    ("ReLU", relu, PLOT_COLORS["orange"]),
    ("Leaky ReLU", leaky_relu, PLOT_COLORS["rose"]),
]):
    ax.plot(x, fn(x), color=color, linewidth=2.5)
    ax.axhline(0, color="#999999", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="#999999", linewidth=0.8, linestyle="--")
    ax.set_title(name, fontweight="bold", color=PLOT_COLORS["ink"])
    ax.set_xlabel("x")
    ax.grid(True, color=PLOT_COLORS["grid"], alpha=0.7)

axes[0].set_ylabel("f(x)")
plt.suptitle("Activation Functions Comparison", fontweight="bold", color=PLOT_COLORS["ink"], fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Vanishing Gradients — Why Sigmoid Can Be Problematic
# 
# The sigmoid derivative is $\sigma'(x) = \sigma(x)(1 - \sigma(x))$, which peaks at **0.25** and approaches zero for large $|x|$.
# 
# In a deep network, backpropagation **multiplies** these small derivatives across layers. After many layers, the gradient reaching early weights can become negligibly small — this is the **vanishing-gradient problem**.
# 
# This is a major reason modern networks prefer **ReLU** (gradient = 1 for positive inputs) in hidden layers.

# %%
# Sigmoid and its derivative
x = np.linspace(-6, 6, 200)
sig = sigmoid(x)
sig_deriv = sig * (1 - sig)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(x, sig, color=PLOT_COLORS["blue"], linewidth=2, label="σ(x)")
ax1.plot(x, sig_deriv, color=PLOT_COLORS["rose"], linewidth=2, label="σ'(x)")
ax1.axhline(0.25, color="#999999", linewidth=0.8, linestyle="--", label="max σ' = 0.25")
ax1.set_title("Sigmoid and Its Derivative", fontweight="bold", color=PLOT_COLORS["ink"])
ax1.set_xlabel("x")
ax1.legend()
ax1.grid(True, color=PLOT_COLORS["grid"], alpha=0.7)

# Gradient decay across layers
layers = np.arange(1, 21)
grad_sigmoid = 0.25 ** layers  # worst-case: derivative = 0.25 at every layer
grad_relu = np.ones_like(layers, dtype=float)  # ReLU: derivative = 1

ax2.semilogy(layers, grad_sigmoid, "o-", color=PLOT_COLORS["rose"], linewidth=2, label="Sigmoid (0.25^L)")
ax2.semilogy(layers, grad_relu, "s--", color=PLOT_COLORS["orange"], linewidth=2, label="ReLU (1^L)")
ax2.set_title("Gradient Magnitude vs Depth", fontweight="bold", color=PLOT_COLORS["ink"])
ax2.set_xlabel("Number of layers")
ax2.set_ylabel("Gradient magnitude (log scale)")
ax2.legend()
ax2.grid(True, color=PLOT_COLORS["grid"], alpha=0.7)

plt.tight_layout()
plt.show()

print(f"After 10 sigmoid layers: gradient ≈ {0.25**10:.2e}")
print(f"After 20 sigmoid layers: gradient ≈ {0.25**20:.2e}")
print("ReLU gradient stays 1.0 regardless of depth (for positive inputs).")

# %% [markdown]
# ### Task 1.2: A Single Neuron — Forward Pass
# 
# Given inputs, weights, and bias, compute the output of a single neuron.
# 
# $$z = X \cdot W + b$$
# $$y = \sigma(z)$$

# %%
inputs = np.array([
    [2.0, 3.0],
    [1.0, 1.0],
    [5.0, 2.0],
])
weights = np.array([[0.5], [-0.5]])  # (2×1)
bias = 1.0

# TODO:
# 1. Compute z = X @ W + b  (use np.dot)
# 2. Apply sigmoid activation to get the output
z = ...
output = ...

print(f"Linear output (z):\n{z}")
print(f"Activated output (y):\n{output}")

# %% [markdown]
# ### Task 1.3: Compute the Loss (MSE)
# 
# The true labels are `[[1.0], [0.0], [1.0]]`. Calculate the Mean Squared Error:
# 
# $$L = \frac{1}{N} \sum (y_{pred} - y_{true})^2$$

# %%
y_true = np.array([[1.0], [0.0], [1.0]])

# TODO:
# 1. Compute the MSE loss between output and y_true
# Hint: np.mean((prediction - target) ** 2)
loss = ...
print(f"MSE Loss: {loss:.6f}")

# %% [markdown]
# ### Task 1.4: Manual Back-Propagation
# 
# Now let's compute gradients to update the weights. This is what PyTorch does for you!
# 
# $$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial z} \cdot \frac{\partial z}{\partial w}$$
# 
# where:
# 
# 1. $\frac{\partial L}{\partial y} = \frac{2}{N}(y_{pred} - y_{true})$
# 2. $\frac{\partial y}{\partial z} = \sigma(z)(1 - \sigma(z))$ — sigmoid derivative
# 3. $\frac{\partial z}{\partial w} = x$

# %%
N = len(inputs)

# TODO:
# 1. Compute dL/dy = (2/N) * (output - y_true)
# 2. Compute dy/dz = output * (1 - output)   (sigmoid derivative)
# 3. Combine into error_term = dL/dy * dy/dz
# 4. Compute grad_weights = inputs.T @ error_term
# 5. Compute grad_bias = sum of error_term
d_loss_d_output = ...
d_output_d_z = ...
error_term = ...

grad_weights = ...
grad_bias = ...

print(f"Gradient w.r.t. weights:\n{grad_weights}")
print(f"Gradient w.r.t. bias: {grad_bias:.6f}")

# Update weights (learning rate = 0.1)
lr = 0.1
new_weights = weights - lr * grad_weights
print(f"Updated weights:\n{new_weights}")

# %% [markdown]
# ### Task 1.5: The Learning Loop
# 
# Put it all together! Run forward pass + back-propagation in a loop for 350 epochs. You should see the loss decrease.

# %%
weights = np.array([[0.5], [-0.5]])
bias = 1.0
lr = 0.1
losses_np = []

# TODO:
# Build a training loop for 350 epochs:
#   1. Forward: z = inputs @ weights + bias, then output = sigmoid(z)
#   2. Loss: MSE between output and y_true
#   3. Backward: compute error_term, grad_weights, grad_bias
#   4. Update: weights -= lr * grad_weights, bias -= lr * grad_bias
#   5. Track loss in losses_np
for epoch in range(350):
    ...

plot_loss_curve(losses_np, "Single-Neuron Training Loss (NumPy)")
print(f"Final predictions:\n{output}")
print(f"Targets:\n{y_true}")

# %% [markdown]
# ### Task 1.6: The "Nightmare" — A 3-Layer Network
# 
# To see why we need PyTorch, let's look at a network with **one hidden layer**:
# 
# - Layer 1 (Hidden): $z_1 = x \cdot W_1 + b_1$, $a_1 = \sigma(z_1)$
# - Layer 2 (Output): $z_2 = a_1 \cdot W_2 + b_2$, $output = \sigma(z_2)$
# 
# Back-propagation now means chaining derivatives through the hidden layer:
# 
# $$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial output} \cdot \frac{\partial output}{\partial z_2} \cdot \frac{\partial z_2}{\partial a_1} \cdot \frac{\partial a_1}{\partial z_1} \cdot \frac{\partial z_1}{\partial W_1}$$
# 
# Study the code below to appreciate the complexity. You do not need to implement this from scratch.

# %%
d_in, d_hidden, d_out = 2, 3, 1

# Random initialisation
np.random.seed(RANDOM_STATE)
W1 = np.random.randn(d_in, d_hidden)
b1 = np.zeros((1, d_hidden))
W2 = np.random.randn(d_hidden, d_out)
b2 = np.zeros((1, d_out))

# Same data as before
x = np.array([[2.0, 3.0], [1.0, 1.0], [5.0, 2.0]])
y_true = np.array([[1.0], [0.0], [1.0]])

# ── Forward ──
z1 = np.dot(x, W1) + b1
a1 = sigmoid(z1)
z2 = np.dot(a1, W2) + b2
output = sigmoid(z2)
print(f"Output:\n{output}")

# ── Backward ──
error_term_output = 2 * (output - y_true) * (output * (1 - output))

grad_W2 = np.dot(a1.T, error_term_output)
grad_b2 = np.sum(error_term_output, axis=0, keepdims=True)

error_term_hidden = np.dot(error_term_output, W2.T) * (a1 * (1 - a1))

grad_W1 = np.dot(x.T, error_term_hidden)
grad_b1 = np.sum(error_term_hidden, axis=0, keepdims=True)

print(f"grad_W1 shape: {grad_W1.shape}")
print(f"grad_W2 shape: {grad_W2.shape}")
print("\nSee how messy this gets? PyTorch handles this automatically!")

# %% [markdown]
# ### Task 1.7: Training the 3-Layer Network
# 
# Wrap the forward + backward logic in a loop for 1 000 epochs. Even with this complexity, it *will* learn.

# %%
np.random.seed(RANDOM_STATE)
W1 = np.random.randn(d_in, d_hidden)
b1 = np.zeros((1, d_hidden))
W2 = np.random.randn(d_hidden, d_out)
b2 = np.zeros((1, d_out))

lr = 0.1
losses_3layer = []

# TODO:
# Build a training loop for 1000 epochs:
#   1. Forward: z1, a1, z2, output (two-layer forward pass from Task 1.6)
#   2. Loss: MSE
#   3. Backward: compute gradients for W2, b2, W1, b1 (chain rule through hidden layer)
#   4. Update all weights and biases
#   5. Track loss in losses_3layer
# Hint: reuse the forward and backward logic from Task 1.6
for epoch in range(1000):
    ...

print(f"\nFinal predictions:\n{output}")
print(f"Targets:\n{y_true}")
plot_loss_curve(losses_3layer, "3-Layer Network Training Loss (NumPy)")

# %% [markdown]
# ### Part 1 — Reflection
# 
# Before moving on, think about:
# 
# - How many lines of gradient code did the 3-layer network require compared to the single neuron?
# - What would happen if you needed 10 layers, or different activation functions per layer?
# - This is exactly the problem that PyTorch's **autograd** solves.

# %% [markdown]
# ## Part 2: Regression with PyTorch (California Housing)
# 
# We will predict house prices based on features like room count, population, and location.
# 
# Notice the workflow change: we no longer write gradient code by hand.

# %%
# 1. Load data
data = fetch_california_housing()
X, y = data.data, data.target

# 2. Split and scale (standard scaling is crucial for neural networks!)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE,
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Convert to PyTorch tensors and move to device
X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)
y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1).to(device)

print(f"Training shape: {X_train_tensor.shape} on {X_train_tensor.device}")
print(f"Test shape:     {X_test_tensor.shape} on {X_test_tensor.device}")

# %% [markdown]
# ### Why PyTorch here?
# 
# We converted NumPy arrays to `torch.FloatTensor`. This lets PyTorch:
# 
# - track operations in a **computation graph**;
# - compute gradients automatically via `loss.backward()`;
# - update parameters in one call via `optimizer.step()`.
# 
# Compare this with the manual gradient code from Part 1.

# %% [markdown]
# ### Task 2.1: Define the Model
# 
# Build a regression model with **Dropout** for regularisation. In NumPy, implementing dropout properly (scaling, turning it off at test time) is tedious. In PyTorch it is one line.

# %%
# TODO:
# Define a HousingModel class inheriting from nn.Module.
# Architecture: Linear(input_dim, 64) → ReLU → Dropout(0.2)
#             → Linear(64, 32)        → ReLU → Dropout(0.2)
#             → Linear(32, 1)
# Hint: use nn.Sequential inside __init__
class HousingModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            ...
        )

    def forward(self, x):
        return self.layers(x)


model_reg = HousingModel(input_dim=8).to(device)
summary(model_reg, input_size=(1, 8), device=device)

# %% [markdown]
# ### Task 2.1.1: Visualise the Computation Graph (Optional)
# 
# PyTorch builds a dynamic computation graph. We can visualise it with `torchviz`.
# 
# > This step is optional. It requires `torchviz` and system `graphviz` to be installed. If they are missing, the cell will print a note and continue.

# %%
try:
    from torchviz import make_dot

    dummy_input = torch.randn(1, 8).to(device)
    dummy_output = model_reg(dummy_input)
    graph = make_dot(dummy_output, params=dict(model_reg.named_parameters()))
    display(graph)
except ImportError:
    print("torchviz or graphviz not installed — skipping graph visualisation.")
    print("Install with: pip install torchviz  (and brew install graphviz / apt install graphviz)")

# %% [markdown]
# ### Task 2.2: Training Loop
# 
# Write the training loop using `MSELoss` and `Adam`.
# 
# ### How To Read The Loss Curve
# 
# - The **x-axis** is the epoch number.
# - The **y-axis** is the training loss.
# - A healthy curve should decrease quickly at first, then flatten.
# - If it oscillates or diverges, the learning rate may be too high.

# %%
criterion = nn.MSELoss()
optimizer = optim.Adam(model_reg.parameters(), lr=0.01)

epochs = 100
losses_reg = []

# TODO:
# Write a training loop for 100 epochs:
#   1. optimizer.zero_grad()
#   2. Forward pass: outputs = model_reg(X_train_tensor)
#   3. Compute loss: criterion(outputs, y_train_tensor)
#   4. loss.backward()
#   5. optimizer.step()
#   6. Append loss.item() to losses_reg
for epoch in range(epochs):
    ...

plot_loss_curve(losses_reg, "Housing Model — Training Loss")

# %% [markdown]
# ### Task 2.3: Inference and Evaluation
# 
# Use the trained model to predict on the test set and calculate RMSE.

# %%
# TODO:
# 1. Set model to eval mode: model_reg.eval()
# 2. Inside torch.no_grad(): predict on X_test_tensor
# 3. Compute MSE, then RMSE = sqrt(MSE)
# Hint: torch.sqrt()
...

print(f"Test RMSE: {rmse.item():.4f}")

print("\nFirst 5 predictions vs. true values:")
for i in range(5):
    print(f"  Predicted: {predictions[i].item():.2f}  |  True: {y_test_tensor[i].item():.2f}")

# %% [markdown]
# ### Part 2 — Reflection
# 
# - How many lines of gradient code did you write? (Answer: zero — `loss.backward()` did it all.)
# - What does `model_reg.eval()` do? Why is it important when using Dropout?
# - How does the RMSE compare to mean house price (~2.07 in units of $100k)?

# %% [markdown]
# ## Part 3: Classification with PyTorch (FashionMNIST)
# 
# Classifying 28×28 grayscale images of clothing into 10 categories.

# %%
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

trainset = torchvision.datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform,
)
testset = torchvision.datasets.FashionMNIST(
    root="./data", train=False, download=True, transform=transform,
)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

CLASSES = (
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
)


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap="gray")
    plt.axis("off")
    plt.show()


# Show a sample batch
dataiter = iter(trainloader)
images, labels = next(dataiter)
print(" | ".join(f"{CLASSES[labels[j]]}" for j in range(8)))
imshow(torchvision.utils.make_grid(images[:8]))

# %% [markdown]
# ### Task 3.1: Build a Robust Classifier
# 
# Define a deeper model with **BatchNorm** and **Dropout** to prevent overfitting.
# 
# - `nn.BatchNorm1d` normalises activations within each mini-batch, stabilising training.
# - `nn.Dropout` randomly zeroes neurons during training, forcing redundancy.

# %%
# TODO:
# Define a FashionClassifier with:
#   Flatten → Linear(784, 256) → BatchNorm1d(256) → ReLU → Dropout(0.2)
#           → Linear(256, 128) → BatchNorm1d(128) → ReLU → Dropout(0.2)
#           → Linear(128, 10)
class FashionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            ...
        )

    def forward(self, x):
        return self.layers(x)


model_cls = FashionClassifier().to(device)
criterion_cls = nn.CrossEntropyLoss()
optimizer_cls = optim.Adam(model_cls.parameters(), lr=0.001)
summary(model_cls, input_size=(1, 1, 28, 28), device=device)

# %% [markdown]
# ### Task 3.2: Train and Evaluate
# 
# Train for 5 epochs, then evaluate accuracy on the test set.
# 
# **Note:** Unlike Part 2 (where we passed the full dataset at once), here we iterate over **mini-batches** from `trainloader`. This is standard for larger datasets.

# %%
epochs = 5

# TODO:
# Training loop (iterate over trainloader for each epoch):
#   1. model_cls.train()
#   2. For each (images, labels) batch:
#      — Move to device: images, labels = images.to(device), labels.to(device)
#      a. optimizer_cls.zero_grad()
#      b. Forward: outputs = model_cls(images)
#      c. Loss: criterion_cls(outputs, labels)
#      d. loss.backward()
#      e. optimizer_cls.step()
#   3. Print average loss per epoch
for epoch in range(epochs):
    ...

print("Finished training.\n")

# TODO:
# Evaluation (iterate over testloader):
#   1. model_cls.eval()
#   2. Inside torch.no_grad(): move batches to device, count correct predictions
#   3. Print accuracy as a percentage
correct = 0
total = 0
...

print(f"Test accuracy: {100 * correct / total:.2f}%")

# %% [markdown]
# ### Task 3.3: Visualise Errors
# 
# It is important to see *what* the model gets wrong. Collect and display 5 misclassified images.

# %%
# TODO:
# 1. Iterate over testloader and collect 5 misclassified examples
# 2. For each batch: move to device, predict, find indices where predicted != labels
# 3. Store the images (.cpu() for display), predicted labels, and true labels
# 4. Call show_misclassified() to display them
incorrect_images = []
incorrect_preds = []
incorrect_labels = []

...

show_misclassified(incorrect_images, incorrect_labels, incorrect_preds, CLASSES, n=5)

# %% [markdown]
# ### Part 3 — Reflection
# 
# - Which clothing categories does the model confuse most often?
# - Would you expect a CNN (convolutional neural network) to do better? Why?
# - What role did BatchNorm play during training?

# %% [markdown]
# ## Bonus: CPU vs GPU Speed Comparison
# 
# Neural-network training involves many matrix multiplications — exactly the kind of work GPUs are designed for. The cell below trains the **same FashionClassifier** on CPU and (if available) on GPU, then compares the wall-clock time.
# 
# > **Google Colab:** go to *Runtime → Change runtime type → T4 GPU* to get a free GPU.
# > On a laptop without a dedicated GPU, the cell will report "GPU not available" and skip the comparison.

# %%
import time


def train_on_device(device, trainloader, epochs=3):
    """Train a fresh FashionClassifier on *device* and return elapsed seconds."""
    torch.manual_seed(RANDOM_STATE)
    model = FashionClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    start = time.perf_counter()
    for epoch in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    # Make sure all GPU ops finish before stopping the timer
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed


BENCH_EPOCHS = 3

# --- CPU benchmark ---
cpu_time = train_on_device(torch.device("cpu"), trainloader, epochs=BENCH_EPOCHS)
print(f"CPU  — {BENCH_EPOCHS} epochs: {cpu_time:.2f} s")

# --- GPU benchmark ---
if torch.cuda.is_available():
    gpu_device = torch.device("cuda")
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")

    # Warm-up run (first CUDA call allocates context)
    _ = train_on_device(gpu_device, trainloader, epochs=1)

    gpu_time = train_on_device(gpu_device, trainloader, epochs=BENCH_EPOCHS)
    print(f"GPU  — {BENCH_EPOCHS} epochs: {gpu_time:.2f} s")
    print(f"\nSpeedup: {cpu_time / gpu_time:.1f}×")
elif torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    print("Apple MPS detected")

    _ = train_on_device(mps_device, trainloader, epochs=1)

    mps_time = train_on_device(mps_device, trainloader, epochs=BENCH_EPOCHS)
    print(f"MPS  — {BENCH_EPOCHS} epochs: {mps_time:.2f} s")
    print(f"\nSpeedup: {cpu_time / mps_time:.1f}×")
else:
    print("\nNo GPU available — run this notebook in Google Colab with a T4 GPU to see the comparison.")

# %% [markdown]
# ## Debrief
# 
# Wrap-up discussion prompts:
# 
# 1. **Manual vs. automatic gradients** — How many lines of gradient code did Part 1 require compared to Parts 2–3? What does this tell you about scaling to deeper networks?
# 2. **Regularisation** — Where did you use Dropout and BatchNorm? How would you know if the model is overfitting?
# 3. **Evaluation** — Is test RMSE (Part 2) or test accuracy (Part 3) enough to judge the model? What else would you check?
# 4. **Next steps** — If you had 30 more minutes, would you try a CNN for FashionMNIST, tune hyperparameters, or add more data augmentation? Why?
# 5. **Big picture** — When would you prefer a simpler model (e.g., logistic regression, random forest) over a neural network?
