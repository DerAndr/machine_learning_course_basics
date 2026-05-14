# Auto-generated companion script for computer_vision_practical_student_90min.ipynb
# Keep the notebook as the source of truth.

# %% [markdown]
# # Computer Vision Basics: CNNs and Multimodal Embeddings (90 minutes)
#
# **Goal:** build intuition for modern computer vision by moving from pixels and hand-written convolution filters to trainable CNN features, then to detection, segmentation, image embeddings, and multimodal image-text embeddings.
#
# **Learning objectives:**
#
# - read images as tensors with shape `height x width x channels`;
# - explain what a convolution kernel does and why CNNs reuse the same filter across an image;
# - train a small CNN on Fashion-MNIST and inspect its feature maps;
# - use a pretrained CNN as an embedding extractor;
# - distinguish image classification, object detection, and segmentation tasks;
# - explain the YOLO one-stage detection idea;
# - build a small visual similarity search system with cosine similarity;
# - connect image embeddings to multimodal embeddings such as CLIP, where images and text live in one shared vector space;
# - name a few current CV directions: promptable segmentation, real-time detection, self-supervised dense features, and prompt-driven vision-language models.
#
# **Practical note:** some cells are guided demos, and some contain TODOs for you to complete during class.
#
# **Agenda:**
#
# | Part | Topic | Time |
# |------|-------|------|
# | 1 | Historical note and images as tensors | ~15 min |
# | 2 | Convolution filters and feature maps | ~20 min |
# | 3 | Tiny CNN on Fashion-MNIST | ~30 min |
# | 4 | Detection, segmentation, and YOLO | ~15 min |
# | 5 | Image embeddings and similarity search | ~15 min |
# | 6 | Bridge to multimodal embeddings | ~5 min |
# | 7 | Modern CV snapshot | optional / debrief |

# %% [markdown]
# ## Setup
#
# The notebook is designed for Google Colab. It also runs locally in this repository if the neural-network dependencies are installed.
#
# Local setup:
#
# ```bash
# uv sync --group neural_networks
# uv run python -m pip install scikit-image
# ```
#
# Colab setup is automatic: the first cell checks for required packages and installs missing ones. We use `skimage.data` for `camera`, `astronaut`, `chelsea`, and `coffee`; Lenna is downloaded from Wikimedia as a historical test image.

# %%
import importlib.util
import os
import subprocess
import sys

IN_COLAB = "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ

required_imports = [
    "numpy", "matplotlib", "PIL", "requests", "scipy", "skimage",
    "sklearn", "torch", "torchvision", "torchinfo",
]
pip_names = {"PIL": "pillow", "skimage": "scikit-image", "sklearn": "scikit-learn"}
missing = [pkg for pkg in required_imports if importlib.util.find_spec(pkg) is None]

if missing and IN_COLAB:
    to_install = [pip_names.get(pkg, pkg) for pkg in missing]
    print("Colab detected. Installing:", ", ".join(to_install))
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *to_install])
elif missing:
    print("Missing packages:", ", ".join(missing))
    print("Install locally with the course environment, or install missing packages manually.")
else:
    print("All required packages are available.")

# %%
import math
import random
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from scipy.signal import correlate2d
from skimage import color, data, img_as_float
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader, Subset
from torchinfo import summary
from torchvision.models import ResNet18_Weights, resnet18
from matplotlib.patches import Rectangle

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

def get_best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_best_device()
print(f"Using device: {DEVICE}")

plt.rcParams.update({
    "figure.figsize": (8, 5),
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})
PLOT_COLORS = {"blue": "#1F77B4", "green": "#2CA02C", "red": "#D62728"}

# %% [markdown]
# ## Helper functions
#
# These helpers keep the lesson cells readable. In the student notebook, most helpers can stay provided while model definition, training, and embedding extraction become TODOs.

# %%
def ensure_rgb(image: np.ndarray) -> np.ndarray:
    """Return an RGB image as a float array in [0, 1]."""
    arr = img_as_float(image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return np.clip(arr, 0, 1)


def pil_from_array(image: np.ndarray) -> Image.Image:
    """Convert a NumPy image array to RGB PIL.Image."""
    arr = ensure_rgb(image)
    return Image.fromarray((arr * 255).astype(np.uint8))


def to_gray(image: np.ndarray) -> np.ndarray:
    """Return a grayscale float image in [0, 1]."""
    arr = img_as_float(image)
    if arr.ndim == 3:
        arr = color.rgb2gray(arr[..., :3])
    return np.clip(arr, 0, 1)


def show_images(images, titles=None, cols=4, cmap=None, figsize=None):
    """Display a list or dict of images.

    Grayscale arrays are shown with a grayscale colormap by default. This avoids
    Matplotlib's default viridis colors, which can make grayscale images look
    artificially colored.
    """
    if isinstance(images, dict):
        titles = list(images.keys())
        images = list(images.values())
    titles = titles or [""] * len(images)
    rows = math.ceil(len(images) / cols)
    figsize = figsize or (3.4 * cols, 3.2 * rows)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)
    for ax, img, title in zip(axes, images, titles):
        arr = np.asarray(img)
        image_cmap = cmap if cmap is not None else ("gray" if arr.ndim == 2 else None)
        if arr.ndim == 2:
            ax.imshow(arr, cmap=image_cmap, vmin=float(np.nanmin(arr)), vmax=float(np.nanmax(arr)))
        else:
            ax.imshow(arr, cmap=image_cmap)
        ax.set_title(title)
        ax.axis("off")
    for ax in axes[len(images):]:
        ax.axis("off")
    plt.tight_layout()
    return fig


def show_tensor_shape(name: str, image: np.ndarray) -> None:
    arr = np.asarray(image)
    print(f"{name:10s} shape={arr.shape}, dtype={arr.dtype}, min={arr.min():.3f}, max={arr.max():.3f}")

# %% [markdown]
# # Part 1. Historical note and images as tensors
#
# Computer vision starts with a deceptively simple idea: an image is a grid of numbers. A grayscale image has one channel. A color image normally has three channels: red, green, and blue.
#
# We begin with **Lenna** because it is a famous historical test image in image processing. It also has a debated cultural history: the image comes from a 1972 Playboy photograph and was reused for decades outside its original context. In class, say this briefly, then move to neutral sample images for the rest of the practical.

# %%
LENNA_URLS = [
    "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png",
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
]


def load_lenna(urls=LENNA_URLS) -> tuple[np.ndarray, str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ml-course-notebook/1.0; educational use)"
    }
    errors = []

    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
            return np.asarray(image), url
        except Exception as exc:
            errors.append(f"{url}: {exc}")

    print("Could not download Lenna from the configured sources.")
    print("Using skimage.data.astronaut() as a placeholder so the notebook can continue.")
    print("Download errors:")
    for error in errors:
        print(" -", error)
    return data.astronaut(), "placeholder: skimage.data.astronaut"


lenna, lenna_source = load_lenna()
print("Lenna source:", lenna_source)
show_images({"Lenna (historical test image)": lenna}, cols=1, figsize=(4, 4))
show_tensor_shape("Lenna", lenna)

# %% [markdown]
# Now switch to neutral sample images from `skimage.data`:
#
# - `camera`: grayscale image, useful for edges and filters;
# - `astronaut`: RGB portrait, useful for channels and textures;
# - `chelsea`: cat image, useful for crops and embeddings;
# - `coffee`: object scene, useful for color and semantic similarity.

# %%
# TODO: load the four built-in scikit-image examples.
# Hint: use data.camera(), data.astronaut(), data.chelsea(), and data.coffee().
skimage_images = {
    "camera": ...,
    "astronaut": ...,
    "chelsea": ...,
    "coffee": ...,
}

show_images(skimage_images, cols=4, figsize=(12, 3.2))
for name, image in skimage_images.items():
    show_tensor_shape(name, image)

# %% [markdown]
# ### RGB channels
#
# A color image is not one matrix, but three aligned matrices. CNNs exploit this: the first convolution layer receives all channels at once and learns filters that can combine color and spatial patterns.

# %%
astronaut = ensure_rgb(skimage_images["astronaut"])
zeros = np.zeros_like(astronaut[..., 0])

# TODO: split RGB image into three visual channel images.
# Each result should still have 3 channels, but only one channel should keep its original values.
channels = {
    "original": astronaut,
    "red channel": ...,
    "green channel": ...,
    "blue channel": ...,
}
show_images(channels, cols=4, figsize=(12, 3.2))

# %% [markdown]
# # Part 2. Convolutions: hand-written filters before learned filters
#
# A convolution kernel is a small matrix that slides over an image. At each position, it multiplies the local patch by the kernel values and sums the result.
#
# This is the bridge from classical CV to CNNs:
#
# - classical CV: humans design filters by hand;
# - CNNs: the model learns useful filters from data.

# %%
def apply_kernel_gray(image_gray: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply the CNN-style cross-correlation operation often called convolution."""
    # TODO: use correlate2d with mode="same" and boundary="symm".
    return ...


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    """Normalize any numeric image to [0, 1] for visualization."""
    arr = np.asarray(image, dtype=float)
    lo, hi = np.percentile(arr, [1, 99])
    if hi <= lo:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0, 1)

# TODO: define five 3x3 kernels: identity, blur, sharpen, sobel_x, sobel_y.
kernels = {
    "identity": ...,
    "blur": ...,
    "sharpen": ...,
    "sobel_x": ...,
    "sobel_y": ...,
}

camera_gray = to_gray(skimage_images["camera"])
filtered = {"original": camera_gray}
for name, kernel in kernels.items():
    if name != "identity":
        filtered[name] = normalize_for_display(apply_kernel_gray(camera_gray, kernel))
show_images(filtered, cols=5, cmap="gray", figsize=(15, 3.2))

# %% [markdown]
# ### One convolution step under a microscope
#
# The cell below visualizes one local operation: a `3 x 3` patch, a `3 x 3` kernel, their element-wise product, and the final sum. That sum becomes one pixel in the feature map.

# %%
def show_convolution_step(image_gray, kernel, row=180, col=180):
    patch = image_gray[row - 1:row + 2, col - 1:col + 2]
    product = patch * kernel
    value = product.sum()

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    panels = [patch, kernel, product, np.array([[value]])]
    titles = ["local image patch", "kernel", "patch * kernel", f"sum = {value:.3f}"]

    for ax, panel, title in zip(axes, panels, titles):
        ax.imshow(panel, cmap="gray")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        for i in range(panel.shape[0]):
            for j in range(panel.shape[1]):
                ax.text(j, i, f"{panel[i, j]:.2f}", ha="center", va="center", color="tab:red", fontsize=9)
    plt.tight_layout()
    return fig

show_convolution_step(camera_gray, kernels["sobel_x"], row=180, col=180)

# %% [markdown]
# ### Receptive field, channels, and feature maps
#
# A CNN layer repeats the same operation thousands of times:
#
# 1. choose a small kernel;
# 2. slide it over the image;
# 3. produce one feature map;
# 4. repeat with many kernels to produce many feature maps.
#
# The key difference from the hand-written kernels above is that CNN kernels are trainable parameters. During training, the network discovers filters that reduce classification loss.

# %%
def plot_kernel_bank(kernels):
    fig, axes = plt.subplots(1, len(kernels), figsize=(3 * len(kernels), 3))
    for ax, (name, kernel) in zip(axes, kernels.items()):
        limit = np.abs(kernel).max()
        # Grayscale keeps the lesson about numeric weights, not false color.
        # Black = negative, mid-gray = zero, white = positive.
        display_limit = limit if limit > 0 else 1.0
        ax.imshow(kernel, cmap="gray", vmin=-display_limit, vmax=display_limit)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        for i in range(kernel.shape[0]):
            for j in range(kernel.shape[1]):
                value = kernel[i, j]
                text_color = "white" if value < -0.25 * display_limit else "black"
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", color=text_color, fontsize=9)
    plt.tight_layout()
    return fig

plot_kernel_bank(kernels)
plt.show()

# %% [markdown]
# # Part 3. Tiny CNN on Fashion-MNIST
#
# The neutral `skimage` images are perfect for explanation, but too few for training a CNN. For training, we use Fashion-MNIST:
#
# - 28 x 28 grayscale images;
# - 10 clothing classes;
# - small enough for CPU, fast enough for Colab;
# - visually interpretable errors.
#
# This notebook trains on a subset to keep the classroom loop short. Increase `TRAIN_LIMIT` and `TEST_LIMIT` if you have more time or a GPU.

# %%
BATCH_SIZE = 128
TRAIN_LIMIT = 12_000
TEST_LIMIT = 3_000
NUM_WORKERS = 0  # Most reliable setting for Colab, Windows, and classroom notebooks.

fashion_transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,)),
])

train_full = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=fashion_transform)
test_full = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=fashion_transform)

train_dataset = Subset(train_full, range(TRAIN_LIMIT))
test_dataset = Subset(test_full, range(TEST_LIMIT))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

class_names = train_full.classes
print(class_names)
print(f"Train subset: {len(train_dataset):,} images")
print(f"Test subset:  {len(test_dataset):,} images")

# %%
def denormalize_fashion(batch):
    return (batch * 0.5 + 0.5).clamp(0, 1)

images, labels = next(iter(train_loader))
preview = denormalize_fashion(images[:16]).squeeze(1).numpy()
preview_titles = [class_names[i] for i in labels[:16].tolist()]
show_images(list(preview), preview_titles, cols=8, cmap="gray", figsize=(14, 4))

# %% [markdown]
# ## Model: two convolution blocks and one classifier head
#
# The model has three conceptual parts:
#
# - **convolution layers** learn local visual features;
# - **pooling** compresses spatial resolution and adds some translation tolerance;
# - **linear layers** use extracted features for classification.
#
# In a student notebook, good TODOs are `conv1`, `conv2`, the `forward` method, and the training loop.

# %%
class TinyFashionCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # TODO: build feature extractor:
            # Conv2d(1 -> 16, kernel_size=3, padding=1), ReLU, MaxPool2d(2),
            # Conv2d(16 -> 32, kernel_size=3, padding=1), ReLU, MaxPool2d(2).
            ...,
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.25),
            # TODO: after two 2x2 pools, Fashion-MNIST is 32 x 7 x 7.
            # Add Linear -> ReLU -> Linear classifier.
            ...,
        )

    def forward(self, x, return_features=False):
        feature_maps = self.features(x)
        logits = self.classifier(feature_maps)
        if return_features:
            return logits, feature_maps
        return logits

model = TinyFashionCNN(num_classes=len(class_names)).to(DEVICE)
print(model)

# %% [markdown]
# ### How many layers does this network have?
#
# This question has more than one valid answer, so always say what you are counting.
#
# For this tiny CNN, three common conventions are useful:
#
# - **Weighted layers**: layers with learned parameters. Here: `Conv2d`, `Conv2d`, `Linear`, `Linear` -> **4 weighted layers**. This is often what people mean when they say a network has N layers.
# - **Leaf operation modules**: all final modules that actually run during `forward`, including activations, pooling, flattening, and dropout -> more than 4.
# - **Container modules**: wrappers such as `Sequential`; useful for code organization, but usually not counted as neural-network layers in model descriptions.
#
# Parameter counting is less ambiguous: for every trainable tensor, multiply its dimensions and sum the results. Bias terms count too.

# %%
def describe_layer_counts(model: nn.Module) -> None:
    weighted_types = (nn.Conv2d, nn.Linear)
    weighted_layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, weighted_types)
    ]
    leaf_modules = [
        (name, module)
        for name, module in model.named_modules()
        if name and not list(module.children())
    ]

    print(f"Weighted layers: {len(weighted_layers)}")
    for name, module in weighted_layers:
        print(f"  {name:18s} {module.__class__.__name__}")

    print(f"\nLeaf operation modules: {len(leaf_modules)}")
    for name, module in leaf_modules:
        print(f"  {name:18s} {module.__class__.__name__}")


def describe_parameters(model: nn.Module) -> None:
    total_params = 0
    trainable_params = 0

    print(f"{'Parameter tensor':36s} {'Shape':20s} {'Params':>12s} {'Trainable':>10s}")
    print("-" * 84)
    for name, parameter in model.named_parameters():
        # TODO: number of scalar parameters in this tensor.
        count = ...
        total_params += count
        if parameter.requires_grad:
            trainable_params += count
        shape = " x ".join(str(dim) for dim in parameter.shape)
        print(f"{name:36s} {shape:20s} {count:12,d} {str(parameter.requires_grad):>10s}")

    print("-" * 84)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


describe_layer_counts(model)
print()
describe_parameters(model)

# %% [markdown]
# ### Manual parameter check
#
# A few examples make the table less mysterious:
#
# - First convolution: `16` filters, each sees `1` input channel and has a `3 x 3` kernel, plus one bias per filter: `16 * (1 * 3 * 3 + 1) = 160`.
# - Second convolution: `32` filters, each sees `16` input channels and has a `3 x 3` kernel, plus bias: `32 * (16 * 3 * 3 + 1) = 4,640`.
# - First linear layer: after two `2 x 2` poolings, `28 x 28` becomes `7 x 7`; with `32` channels, the flattened vector has `32 * 7 * 7 = 1,568` values. The layer has `64` outputs: `1,568 * 64 + 64 = 100,416`.
# - Final linear layer: `64 * 10 + 10 = 650`.
#
# Total: `160 + 4,640 + 100,416 + 650 = 105,866` parameters.

# %% [markdown]
# ### How to print the neural-network structure
#
# There are several levels of detail:
#
# - `print(model)` shows the module tree.
# - `named_modules()` gives programmatic access to every nested module.
# - `torchinfo.summary(...)` runs a dummy input through the model and reports output shapes and parameter counts.
#
# For CNNs, the output shapes are especially useful: they show how spatial dimensions change after convolutions and pooling.

# %%
print("Module tree from print(model):")
print(model)

print("\nNamed leaf modules:")
for name, module in model.named_modules():
    if name and not list(module.children()):
        print(f"{name:18s} -> {module}")

print("\ntorchinfo summary:")
summary(
    model,
    input_size=(BATCH_SIZE, 1, 28, 28),
    col_names=("input_size", "output_size", "num_params", "trainable"),
    depth=3,
    device=str(DEVICE),
)

# %% [markdown]
# ### Visualizing activations across CNN layers
#
# The model structure tells us what operations exist. Activations show what actually happens to one image as it passes through the network.
#
# For a CNN, each convolution layer produces a stack of **feature maps**. Early maps are still close to local edges and strokes. After pooling and deeper convolution, maps become smaller and more abstract: they no longer look like the original image, but they highlight patterns useful for the classifier.
#
# We will use PyTorch forward hooks to capture intermediate outputs without rewriting the model.

# %%
def capture_activations(model: nn.Module, image_tensor: torch.Tensor, layer_names: list[str]):
    activations = {}
    hooks = []
    modules = dict(model.named_modules())

    def make_hook(name):
        def hook(module, inputs, output):
            # TODO: save a detached CPU copy of the layer output.
            activations[name] = ...
        return hook

    for name in layer_names:
        hooks.append(modules[name].register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        _ = model(image_tensor.unsqueeze(0).to(DEVICE))

    for hook in hooks:
        hook.remove()

    return activations


def show_activation_grid(activation: torch.Tensor, title: str, max_maps=12, cmap="gray"):
    # Activation shape is usually [batch, channels, height, width].
    activation = activation.squeeze(0)
    if activation.ndim != 3:
        raise ValueError(f"Expected [channels, height, width], got shape {tuple(activation.shape)}")

    maps_to_show = min(max_maps, activation.shape[0])
    cols = min(6, maps_to_show)
    rows = math.ceil(maps_to_show / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.1 * rows))
    axes = np.array(axes).reshape(-1)

    for idx, ax in enumerate(axes):
        ax.axis("off")
        if idx < maps_to_show:
            feature_map = activation[idx].numpy()
            ax.imshow(feature_map, cmap=cmap, vmin=float(np.nanmin(feature_map)), vmax=float(np.nanmax(feature_map)))
            ax.set_title(f"map {idx}")

    fig.suptitle(f"{title} | shape: {tuple(activation.shape)}", y=1.02)
    plt.tight_layout()
    return fig

# %%
activation_layer_names = [
    "features.0",  # first Conv2d: 1 x 28 x 28 -> 16 x 28 x 28
    "features.1",  # ReLU after first convolution
    "features.2",  # first MaxPool: 16 x 28 x 28 -> 16 x 14 x 14
    "features.3",  # second Conv2d: 16 x 14 x 14 -> 32 x 14 x 14
    "features.5",  # second MaxPool: 32 x 14 x 14 -> 32 x 7 x 7
]

sample_image, sample_label = test_full[0]
activations = capture_activations(model, sample_image, activation_layer_names)

print("Input image shape:", tuple(sample_image.shape), "class:", class_names[sample_label])
for name, activation in activations.items():
    print(f"{name:10s} {dict(model.named_modules())[name].__class__.__name__:10s} -> {tuple(activation.shape)}")

show_images({"input image": denormalize_fashion(sample_image).squeeze(0)}, cols=1, cmap="gray", figsize=(3, 3))

# %% [markdown]
# The next plots show the first few channels from different stages.
#
# Teaching prompts:
#
# - What changes after ReLU? Negative values disappear, so some regions become inactive.
# - What changes after pooling? The spatial grid becomes smaller, but strong responses remain.
# - Why do deeper maps look less like clothing? They are no longer meant to be human-readable pictures; they are intermediate signals for the classifier.

# %%
show_activation_grid(activations["features.0"], "After first convolution", max_maps=12)
plt.show()
show_activation_grid(activations["features.1"], "After first ReLU", max_maps=12)
plt.show()
show_activation_grid(activations["features.2"], "After first max-pooling", max_maps=12)
plt.show()
show_activation_grid(activations["features.3"], "After second convolution", max_maps=12)
plt.show()
show_activation_grid(activations["features.5"], "After second max-pooling", max_maps=12)
plt.show()

# %% [markdown]
# ### Activation strength by layer
#
# A compact numeric view is also useful. The bars below summarize activation magnitudes after each captured layer. This is not an evaluation metric; it is a diagnostic view that helps students see that each layer transforms the distribution of signals.

# %%
activation_stats = []
for name, activation in activations.items():
    values = activation.numpy().ravel()
    activation_stats.append({
        "layer": name,
        "mean_abs": float(np.mean(np.abs(values))),
        "max_abs": float(np.max(np.abs(values))),
        "zero_fraction": float(np.mean(np.isclose(values, 0.0))),
    })

labels = [item["layer"] for item in activation_stats]
mean_abs = [item["mean_abs"] for item in activation_stats]
zero_fraction = [item["zero_fraction"] for item in activation_stats]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(labels, mean_abs, color=PLOT_COLORS["blue"])
axes[0].set_title("Mean absolute activation")
axes[0].set_ylabel("Mean |value|")
axes[0].tick_params(axis="x", rotation=30)

axes[1].bar(labels, zero_fraction, color=PLOT_COLORS["green"])
axes[1].set_title("Fraction of exact zeros")
axes[1].set_ylabel("Zero fraction")
axes[1].set_ylim(0, 1)
axes[1].tick_params(axis="x", rotation=30)

plt.tight_layout()
activation_stats

# %%
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_examples = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # TODO: forward pass, loss, backward pass, optimizer step.
        logits = ...
        loss = ...
        ...
        ...
        total_loss += loss.item() * images.size(0)
        total_examples += images.size(0)
    return total_loss / total_examples


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    all_logits, all_labels, all_images = [], [], []
    for images, labels in loader:
        # TODO: run the model on the current batch and move logits back to CPU.
        logits = ...
        all_logits.append(logits)
        all_labels.append(labels)
        all_images.append(images)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    images = torch.cat(all_images)
    preds = logits.argmax(dim=1)
    return images, labels, preds, logits


def evaluate_accuracy(model, loader, device):
    _, labels, preds, _ = predict_loader(model, loader, device)
    return accuracy_score(labels.numpy(), preds.numpy())

# %%
EPOCHS = 3
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
history = {"train_loss": [], "test_accuracy": []}

for epoch in range(1, EPOCHS + 1):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
    test_accuracy = evaluate_accuracy(model, test_loader, DEVICE)
    history["train_loss"].append(train_loss)
    history["test_accuracy"].append(test_accuracy)
    print(f"Epoch {epoch:02d} | train loss={train_loss:.4f} | test accuracy={test_accuracy:.3f}")

fig, ax1 = plt.subplots(figsize=(8, 4))
epochs = np.arange(1, EPOCHS + 1)
ax1.plot(epochs, history["train_loss"], marker="o", color=PLOT_COLORS["blue"], label="Train loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax2 = ax1.twinx()
ax2.plot(epochs, history["test_accuracy"], marker="s", color=PLOT_COLORS["green"], label="Test accuracy")
ax2.set_ylabel("Accuracy")
lines = ax1.get_lines() + ax2.get_lines()
ax1.legend(lines, [line.get_label() for line in lines], loc="center right")
ax1.set_title("Tiny CNN training progress")
plt.tight_layout()

# %% [markdown]
# ### What did the CNN learn?
#
# The first convolution layer usually learns simple local patterns: dark-to-light edges, light-to-dark edges, dots, strokes, and texture fragments. The exact filters depend on initialization and data, so do not overinterpret one filter. Use them as evidence that the model learns feature detectors rather than memorizing class names directly.

# %%
def visualize_first_layer_filters(model):
    # TODO: take first convolution weights and remove the single input-channel dimension.
    weights = ...
    cols = 8
    rows = math.ceil(weights.shape[0] / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2 * rows))
    axes = np.array(axes).reshape(-1)
    for idx, ax in enumerate(axes):
        ax.axis("off")
        if idx < weights.shape[0]:
            kernel = weights[idx].numpy()
            limit = np.abs(kernel).max()
            display_limit = limit if limit > 0 else 1.0
            ax.imshow(kernel, cmap="gray", vmin=-display_limit, vmax=display_limit)
            ax.set_title(f"filter {idx}")
    plt.tight_layout()
    return fig

visualize_first_layer_filters(model)
plt.show()

# %%
@torch.no_grad()
def visualize_feature_maps(model, image_tensor, max_maps=16):
    model.eval()
    image = image_tensor.unsqueeze(0).to(DEVICE)
    _, feature_maps = model(image, return_features=True)
    feature_maps = feature_maps.squeeze(0).cpu()
    maps_to_show = min(max_maps, feature_maps.shape[0])

    fig, axes = plt.subplots(2, math.ceil((maps_to_show + 1) / 2), figsize=(14, 5))
    axes = np.array(axes).reshape(-1)
    axes[0].imshow(denormalize_fashion(image_tensor).squeeze(0), cmap="gray")
    axes[0].set_title("input image")
    axes[0].axis("off")

    for idx in range(maps_to_show):
        ax = axes[idx + 1]
        ax.imshow(feature_maps[idx], cmap="gray", vmin=float(feature_maps[idx].min()), vmax=float(feature_maps[idx].max()))
        ax.set_title(f"map {idx}")
        ax.axis("off")
    for ax in axes[maps_to_show + 1:]:
        ax.axis("off")
    plt.tight_layout()
    return fig

sample_image, sample_label = test_full[0]
print("True class:", class_names[sample_label])
visualize_feature_maps(model, sample_image, max_maps=11)

# %% [markdown]
# ### Error analysis
#
# Metrics tell us whether the model is working. Misclassified examples tell us how it fails. In an introductory class, this is often where students stop treating models as magic and start treating them as systems with observable behavior.

# %%
test_images, test_labels, test_preds, test_logits = predict_loader(model, test_loader, DEVICE)
print(f"Test subset accuracy: {accuracy_score(test_labels.numpy(), test_preds.numpy()):.3f}")

cm = confusion_matrix(test_labels.numpy(), test_preds.numpy(), labels=list(range(10)))
fig, ax = plt.subplots(figsize=(8, 8))
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, xticks_rotation=45, colorbar=False)
ax.set_title("Tiny CNN confusion matrix")
plt.tight_layout()

wrong = torch.where(test_preds != test_labels)[0]
if len(wrong) > 0:
    chosen = wrong[:16]
    imgs = denormalize_fashion(test_images[chosen]).squeeze(1).numpy()
    titles = [
        f"true: {class_names[int(test_labels[i])]}\npred: {class_names[int(test_preds[i])]}"
        for i in chosen.tolist()
    ]
    show_images(list(imgs), titles, cols=8, cmap="gray", figsize=(15, 5))

# %% [markdown]
# # Part 4. Detection, segmentation, and YOLO
#
# So far our CNN has solved **image classification**: one image goes in, one class label comes out. Many computer-vision tasks need richer outputs.
#
# | Task | Question | Typical output | Example |
# |------|----------|----------------|---------|
# | Classification | What is the main object or class? | one label per image | `sneaker` |
# | Object detection | What objects are present, and where are they? | bounding boxes + labels + confidence scores | `person: 0.93` at box `(x1, y1, x2, y2)` |
# | Semantic segmentation | What class is each pixel? | one class map over the image | every pixel is `road`, `person`, `sky`, ... |
# | Instance segmentation | Which object instance owns each pixel? | separate masks per object | `person #1`, `person #2`, `cat #1` |
#
# A useful classroom phrasing:
#
# > Classification names the image. Detection draws boxes. Segmentation colors pixels.

# %%
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
example = ensure_rgb(skimage_images["astronaut"])

# Classification sketch
axes[0].imshow(example)
axes[0].set_title("Classification\nlabel: astronaut/person")
axes[0].axis("off")

# Detection sketch
axes[1].imshow(example)
axes[1].add_patch(Rectangle((175, 35), 165, 310, fill=False, edgecolor="lime", linewidth=3))
axes[1].text(175, 28, "person 0.93", color="black", backgroundcolor="lime", fontsize=10)
axes[1].set_title("Detection\nboxes + labels + scores")
axes[1].axis("off")

# Segmentation sketch. This is an illustrative mask, not a model prediction.
axes[2].imshow(example)
yy, xx = np.mgrid[:example.shape[0], :example.shape[1]]
head = ((xx - 250) / 72) ** 2 + ((yy - 115) / 82) ** 2 < 1
body = ((xx - 255) / 105) ** 2 + ((yy - 285) / 175) ** 2 < 1
helmet = ((xx - 285) / 130) ** 2 + ((yy - 360) / 115) ** 2 < 1
mask = head | body | helmet
overlay = np.zeros_like(example)
overlay[..., 0] = 1.0
mask_3d = np.repeat(mask[..., None], 3, axis=2)
axes[2].imshow(np.ma.masked_where(~mask_3d, overlay), alpha=0.35)
axes[2].contour(mask, levels=[0.5], colors=["white"], linewidths=1.5)
axes[2].set_title("Segmentation\nrough mask overlay")
axes[2].axis("off")

plt.tight_layout()

# %% [markdown]
# ## Real YOLO detection demo
#
# **YOLO** means **You Only Look Once**. The core idea is a one-stage detector: instead of first proposing candidate regions and then classifying them, the network predicts boxes and classes in one forward pass.
#
# For the practical, we should show a real detector, not hand-drawn boxes. The cell below runs an Ultralytics YOLO-family model on the `astronaut` image and displays actual predicted boxes, labels, and confidence scores.
#
# Teaching angle:
#
# - the output is no longer one class label for the whole image;
# - each detection has a box, class name, and confidence score;
# - predictions can be imperfect, and that is useful to discuss.

# %%
RUN_REAL_YOLO_DEMO = True


def install_ultralytics_if_needed():
    if importlib.util.find_spec("ultralytics") is None:
        print("Installing ultralytics for the real YOLO demo...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])


def load_first_available_yolo(model_candidates=("yolo26n.pt", "yolo11n.pt", "yolov8n.pt")):
    from ultralytics import YOLO

    errors = []
    for model_name in model_candidates:
        try:
            model = YOLO(model_name)
            return model, model_name
        except Exception as exc:
            errors.append(f"{model_name}: {exc}")

    raise RuntimeError("Could not load any YOLO model. Errors:\n" + "\n".join(errors))


if RUN_REAL_YOLO_DEMO:
    install_ultralytics_if_needed()
    yolo_model, yolo_model_name = load_first_available_yolo()

    yolo_image = Image.fromarray((ensure_rgb(skimage_images["astronaut"]) * 255).astype(np.uint8))
    yolo_results = yolo_model(yolo_image, conf=0.25, verbose=False)[0]

    print(f"YOLO model used: {yolo_model_name}")
    print(f"Detections: {len(yolo_results.boxes)}")
    for box in yolo_results.boxes:
        class_id = int(box.cls.item())
        class_name = yolo_results.names[class_id]
        confidence = float(box.conf.item())
        x1, y1, x2, y2 = box.xyxy.squeeze().tolist()
        print(f"- {class_name:12s} conf={confidence:.2f} box=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")

    annotated = yolo_results.plot()
    plt.figure(figsize=(7, 7))
    plt.imshow(annotated)
    plt.title(f"Real YOLO detections: {yolo_model_name}")
    plt.axis("off")
else:
    print("Real YOLO demo is off. Set RUN_REAL_YOLO_DEMO = True to run it.")

# %% [markdown]
# ## Optional: real object detection with TorchVision
#
# This cell runs by default and downloads a pretrained detection model; it can take a few minutes on CPU. It uses Faster R-CNN rather than YOLO because TorchVision ships the full inference pipeline with the dependencies we already use.
#
# Teaching angle: compare the output format with the sketch above. A detector returns `boxes`, `labels`, and `scores`. YOLO-style models return the same kind of final objects, even though the internal architecture is different.

# %%
RUN_TORCHVISION_DETECTION_DEMO = True

if RUN_TORCHVISION_DETECTION_DEMO:
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn

    detection_weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    detection_model = fasterrcnn_resnet50_fpn(weights=detection_weights).to(DEVICE).eval()
    detection_transform = detection_weights.transforms()
    detection_categories = detection_weights.meta["categories"]

    pil_image = Image.fromarray((ensure_rgb(skimage_images["astronaut"]) * 255).astype(np.uint8))
    image_tensor = detection_transform(pil_image).to(DEVICE)

    with torch.no_grad():
        prediction = detection_model([image_tensor])[0]

    score_threshold = 0.55
    keep = prediction["scores"].cpu() >= score_threshold
    boxes = prediction["boxes"].cpu()[keep]
    labels = prediction["labels"].cpu()[keep]
    scores = prediction["scores"].cpu()[keep]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(pil_image)
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.tolist()
        class_name = detection_categories[int(label)]
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="lime", linewidth=2))
        ax.text(x1, y1 - 4, f"{class_name} {float(score):.2f}", color="black", backgroundcolor="lime", fontsize=9)
    ax.set_title("TorchVision detection output: boxes + labels + scores")
    ax.axis("off")
    plt.tight_layout()
else:
    print("Detection demo is off. Set RUN_TORCHVISION_DETECTION_DEMO = True to run it.")

# %% [markdown]
# ## Optional: real semantic segmentation with TorchVision
#
# Segmentation predicts a class for each pixel. The model output is a tensor shaped like:
#
# `[batch, classes, height, width]`
#
# For every pixel, we take the class with the largest score. The result is a class map that can be shown as a colored overlay.

# %%
RUN_TORCHVISION_SEGMENTATION_DEMO = True

if RUN_TORCHVISION_SEGMENTATION_DEMO:
    from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50

    segmentation_weights = DeepLabV3_ResNet50_Weights.DEFAULT
    segmentation_model = deeplabv3_resnet50(weights=segmentation_weights).to(DEVICE).eval()
    segmentation_transform = segmentation_weights.transforms()
    segmentation_categories = segmentation_weights.meta["categories"]

    def predict_class_map(pil_image: Image.Image) -> np.ndarray:
        image_tensor = segmentation_transform(pil_image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = segmentation_model(image_tensor)["out"].cpu()
        return output.argmax(dim=1).squeeze(0).numpy()

    segmentation_candidates = [
        ("chelsea", pil_from_array(skimage_images["chelsea"]), "cat"),
        ("astronaut", pil_from_array(skimage_images["astronaut"]), "person"),
    ]

    candidate_results = []
    for image_name, pil_image, target_class in segmentation_candidates:
        if target_class not in segmentation_categories:
            continue
        class_map = predict_class_map(pil_image)
        target_index = segmentation_categories.index(target_class)
        raw_mask = class_map == target_index
        # The model output may be resized by the preprocessing pipeline, so resize
        # the mask back to the displayed image size before overlaying it.
        mask_image = Image.fromarray((raw_mask.astype(np.uint8) * 255))
        mask = np.asarray(mask_image.resize(pil_image.size, Image.Resampling.NEAREST)) > 0
        foreground_fraction = float(mask.mean())
        candidate_results.append((image_name, pil_image, target_class, mask, foreground_fraction))

    if not candidate_results:
        raise RuntimeError("No supported segmentation target classes were found in the model metadata.")

    good_results = [item for item in candidate_results if 0.02 <= item[4] <= 0.75]
    if good_results:
        image_name, pil_image, target_class, mask, foreground_fraction = good_results[0]
    else:
        image_name, pil_image, target_class, mask, foreground_fraction = min(
            candidate_results,
            key=lambda item: abs(item[4] - 0.35),
        )
        print("Warning: no candidate produced a clean foreground fraction; showing the least-bad mask.")

    print(f"Segmentation example: image={image_name}, target={target_class}, foreground={foreground_fraction:.1%}")

    red_overlay = np.zeros((*mask.shape, 3), dtype=float)
    red_overlay[..., 0] = 1.0
    mask_3d = np.repeat(mask[..., None], 3, axis=2)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(pil_image)
    axes[0].set_title("Original image")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title(f"Predicted {target_class} mask")
    axes[1].axis("off")

    axes[2].imshow(pil_image)
    axes[2].imshow(np.ma.masked_where(~mask_3d, red_overlay), alpha=0.45)
    axes[2].contour(mask, levels=[0.5], colors=["white"], linewidths=1.2)
    axes[2].set_title("Semantic segmentation overlay")
    axes[2].axis("off")

    plt.tight_layout()
else:
    print("Segmentation demo is off. Set RUN_TORCHVISION_SEGMENTATION_DEMO = True to run it.")

# %% [markdown]
# ## YOLO note
#
# The real YOLO demo above is the canonical detection demo for this notebook. If the newest `yolo26n.pt` weights are unavailable in a particular Colab runtime, the loader falls back to `yolo11n.pt` and then `yolov8n.pt`, so the cell still shows genuine YOLO-family inference.

# %% [markdown]
# # Part 5. Image embeddings with a pretrained CNN
#
# Training a CNN from scratch is useful for learning, but many real workflows use a pretrained model as a feature extractor.
#
# An **image embedding** is a vector representation of an image. Instead of comparing raw pixels, we compare embeddings. A good embedding space has a useful geometry: similar images should have high cosine similarity and nearby vector positions.
#
# Here we use pretrained ResNet-18 from `torchvision` and remove its final classification layer. The model was trained on ImageNet, so its intermediate features are often useful for edges, textures, parts, and object-level visual patterns. For the embedding gallery, we use a small balanced CIFAR-10 sample so students see more than the four `skimage` demo images.

# %%
def center_crop_fraction(image: Image.Image, fraction=0.72) -> Image.Image:
    width, height = image.size
    crop_w = int(width * fraction)
    crop_h = int(height * fraction)
    left = (width - crop_w) // 2
    top = (height - crop_h) // 2
    return image.crop((left, top, left + crop_w, top + crop_h)).resize((width, height))


def brighten(image: Image.Image, factor=1.25) -> Image.Image:
    return ImageEnhance.Brightness(image).enhance(factor)


# Use a small real dataset sample for embeddings, not only skimage demo images.
# CIFAR-10 is tiny, downloads reliably in Colab, and gives semantic variety.
cifar_for_embeddings = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=None,
)
cifar_class_names = cifar_for_embeddings.classes

# Pick a few examples per class for a balanced 30-image gallery.
examples_per_class = 3
selected_by_class = {class_idx: [] for class_idx in range(len(cifar_class_names))}
for image, label in cifar_for_embeddings:
    if len(selected_by_class[label]) < examples_per_class:
        selected_by_class[label].append(image.convert("RGB"))
    if all(len(items) == examples_per_class for items in selected_by_class.values()):
        break

embedding_images = {}
for class_idx, images_for_class in selected_by_class.items():
    class_name = cifar_class_names[class_idx]
    for item_idx, image in enumerate(images_for_class, start=1):
        embedding_images[f"{class_name}_{item_idx}"] = image

# Add a few transformed duplicates as sanity checks: near-duplicates should be close.
anchor_name = "cat_1" if "cat_1" in embedding_images else next(iter(embedding_images))
anchor_image = embedding_images[anchor_name]
embedding_images[f"{anchor_name}_flip"] = ImageOps.mirror(anchor_image)
embedding_images[f"{anchor_name}_crop"] = center_crop_fraction(anchor_image, 0.75)
embedding_images[f"{anchor_name}_bright"] = brighten(anchor_image, 1.25)

print(f"Embedding image set size: {len(embedding_images)}")
print("CIFAR-10 classes:", cifar_class_names)
show_images({name: np.asarray(img.resize((96, 96), Image.Resampling.NEAREST)) for name, img in embedding_images.items()}, cols=6, figsize=(16, 14))

# %%
weights = ResNet18_Weights.DEFAULT
embedding_transform = weights.transforms()

resnet = resnet18(weights=weights)
# TODO: replace the classification head with identity to get 512-dimensional image embeddings.
resnet.fc = ...
resnet = resnet.to(DEVICE).eval()

@torch.no_grad()
def extract_resnet_embeddings(pil_images: dict[str, Image.Image]) -> tuple[list[str], np.ndarray]:
    names = list(pil_images.keys())
    batch = torch.stack([embedding_transform(pil_images[name]) for name in names]).to(DEVICE)
    embeddings = resnet(batch).cpu().numpy()
    # TODO: L2-normalize embeddings row-wise for cosine similarity.
    embeddings = ...
    return names, embeddings

embedding_names, image_embeddings = extract_resnet_embeddings(embedding_images)
print("Embedding matrix shape:", image_embeddings.shape)
print("One image is now represented by a vector of length", image_embeddings.shape[1])

# %% [markdown]
# ### Similarity matrix and nearest neighbors
#
# Cosine similarity is the dot product after vectors are normalized. The matrix is symmetric, so the plot below shows only the lower triangle and hides the duplicate upper half plus the trivial diagonal.
#
# Discussion prompt: where does the model behave as expected, and where does it surprise you? Pretrained embeddings are powerful, but they are not perfect universal truth.

# %%
# TODO: compute pairwise cosine similarity between all image embeddings.
similarity = ...
mask_upper = np.triu(np.ones_like(similarity, dtype=bool), k=0)
similarity_lower = np.ma.array(similarity, mask=mask_upper)
color_min = float(similarity_lower.min())

fig, ax = plt.subplots(figsize=(12, 11))
im = ax.imshow(similarity_lower, cmap="coolwarm", vmin=color_min, vmax=1)
ax.set_xticks(range(len(embedding_names)))
ax.set_yticks(range(len(embedding_names)))
ax.set_xticklabels(embedding_names, rotation=45, ha="right")
ax.set_yticklabels(embedding_names)

for i in range(len(embedding_names)):
    for j in range(len(embedding_names)):
        if i > j:
            value = similarity[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="black", fontsize=7)

ax.set_title("Cosine similarity between image embeddings (lower triangle)")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()

# %%
def show_nearest_neighbors(query_name, names, embeddings, pil_images, top_k=5):
    name_to_index = {name: idx for idx, name in enumerate(names)}
    query_idx = name_to_index[query_name]
    # TODO: compare one query embedding against all embeddings.
    scores = ...
    order = np.argsort(scores)[::-1][:top_k]
    result_images = {f"{names[idx]}\ncos={scores[idx]:.2f}": np.asarray(pil_images[names[idx]]) for idx in order}
    show_images(result_images, cols=top_k, figsize=(3 * top_k, 3.5))
    return [(names[idx], float(scores[idx])) for idx in order]

# Query a real CIFAR-10 image. The transformed anchor variants give us a sanity check:
# they should usually appear close to the original anchor.
query_name = anchor_name
show_nearest_neighbors(query_name, embedding_names, image_embeddings, embedding_images, top_k=6)

# %%
# TODO: project embeddings to 2D with PCA for visualization.
pca = ...
coords = ...
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(coords[:, 0], coords[:, 1], s=90, color=PLOT_COLORS["blue"])
for name, (x, y) in zip(embedding_names, coords):
    ax.annotate(name, (x, y), xytext=(6, 4), textcoords="offset points")
ax.axhline(0, color="#DDDDDD", linewidth=1)
ax.axvline(0, color="#DDDDDD", linewidth=1)
ax.set_title("2D PCA projection of ResNet image embeddings")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
plt.tight_layout()

# %% [markdown]
# # Part 6. From image embeddings to multimodal embeddings
#
# Image embeddings compare images with images. Multimodal embeddings go one step further: they put different modalities into a shared vector space.
#
# A CLIP-style model has two encoders:
#
# - an image encoder maps images to vectors;
# - a text encoder maps captions/prompts to vectors;
# - training pulls matching image-text pairs together and pushes mismatched pairs apart.
#
# After that, cosine similarity can compare an image vector with a text vector. This enables zero-shot classification, text-to-image search, image-to-text retrieval, and many modern retrieval-augmented AI workflows.
#
# Classroom phrasing:
#
# > With ResNet embeddings, `chelsea` can be close to `chelsea_flip`. With multimodal embeddings, the same image can also be close to the text prompt `a photo of a cat`.

# %%
# Optional Colab demo: CLIP-style multimodal embeddings.
# This runs by default in the student notebook. Turn it off if the runtime has limited internet, time, or disk.
# The rest of the practical does not depend on this cell.

RUN_MULTIMODAL_DEMO = True

if RUN_MULTIMODAL_DEMO:
    if importlib.util.find_spec("open_clip") is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "open_clip_torch"])
    import open_clip

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=DEVICE,
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    clip_model.eval()

    # TODO idea after the first run: edit these prompts and observe how the scores change.
    prompts = [
        "a photo of a cat",
        "a photo of a person in a space suit",
        "a photo of a cup of coffee",
        "a black and white photograph of a camera operator",
        "a historical portrait test image",
    ]
    clip_demo_images = {
        "chelsea": pil_from_array(skimage_images["chelsea"]),
        "astronaut": pil_from_array(skimage_images["astronaut"]),
        "coffee": pil_from_array(skimage_images["coffee"]),
        "camera": pil_from_array(skimage_images["camera"]),
        "lenna": pil_from_array(lenna),
    }
    clip_names = list(clip_demo_images)
    clip_image_batch = torch.stack([clip_preprocess(clip_demo_images[name]) for name in clip_names]).to(DEVICE)
    clip_text_tokens = tokenizer(prompts).to(DEVICE)

    with torch.no_grad():
        image_features = clip_model.encode_image(clip_image_batch)
        text_features = clip_model.encode_text(clip_text_tokens)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_text_similarity = (image_features @ text_features.T).cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(image_text_similarity, cmap="coolwarm", vmin=float(image_text_similarity.min()), vmax=float(image_text_similarity.max()))
    ax.set_xticks(range(len(prompts)))
    ax.set_yticks(range(len(clip_names)))
    ax.set_xticklabels(prompts, rotation=35, ha="right")
    ax.set_yticklabels(clip_names)
    for i in range(len(clip_names)):
        for j in range(len(prompts)):
            ax.text(j, i, f"{image_text_similarity[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    ax.set_title("Image-text similarity in a multimodal embedding space")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
else:
    print("Optional multimodal demo is off. Set RUN_MULTIMODAL_DEMO = True to run CLIP in Colab.")

# %% [markdown]
# # Part 7. Modern CV snapshot: what feels current now?
#
# This practical teaches fundamentals, but the field has moved from single-task CNNs toward **foundation-style visual systems**. The goal of this final block is not to run every new model. It is to give students a mental map of what changed recently.
#
# | Direction | Example models | Why it matters | How it connects to today |
# |-----------|----------------|----------------|--------------------------|
# | Promptable segmentation | SAM 2 | Segment objects from clicks, boxes, masks, and video prompts | Extends segmentation from fixed classes to interactive masks |
# | Real-time detection | YOLO26, RF-DETR | Detection is still being optimized for speed, edge deployment, and fine-tuning | Extends our detection sketch into production systems |
# | Self-supervised dense features | DINOv3 | Useful visual features can be learned from images without human labels | Extends image embeddings beyond supervised ResNet features |
# | Multilingual vision-language encoders | SigLIP 2 | Images and text can share a stronger multilingual embedding space | Extends the CLIP-style multimodal idea |
# | Prompt-driven vision foundation models | Florence-2 | One model can respond to task prompts for captioning, detection, grounding, OCR, and segmentation | Connects CV tasks to instruction-style interfaces |
#
# Key message: CNNs are still foundational. What changed is how those visual features are trained, prompted, reused, and connected to language.

# %% [markdown]
# ## SAM 2: promptable segmentation for images and video
#
# **Segment Anything Model 2 (SAM 2)** is a good modern example because it makes segmentation interactive. Instead of training a task-specific segmentation model for one fixed label set, a user can provide prompts such as points or boxes, and the model predicts object masks. SAM 2 also extends this idea to video, tracking object masks across frames.
#
# Classroom framing:
#
# > Classical segmentation asks: what class is each pixel? Promptable segmentation asks: which pixels belong to the object I just indicated?
#
# Useful links for students:
#
# - Meta SAM 2 page: https://ai.meta.com/sam2/
# - Meta announcement: https://about.fb.com/news/2024/07/our-new-ai-model-can-segment-video/
# - Paper: https://arxiv.org/abs/2408.00714

# %%
# SAM 2 is best treated as a demo pointer in this 90-minute class.
# The official demos and notebooks change faster than this course repository.
# Suggested live activity: open the SAM 2 demo/page and ask students:
# 1. What is the prompt?
# 2. What is the output mask?
# 3. How is this different from semantic segmentation with fixed classes?

print("SAM 2 demo/reference: https://ai.meta.com/sam2/")

# %% [markdown]
# ## YOLO26 and RF-DETR: detection is still moving
#
# Our YOLO explanation used the classic one-stage idea: grid-like spatial predictions, boxes, class scores, then duplicate removal. Modern real-time detectors are still evolving.
#
# Two useful current examples:
#
# - **YOLO26**: Ultralytics positions it as a newer YOLO family model with end-to-end, NMS-free inference for streamlined deployment.
# - **RF-DETR**: a real-time detection-transformer family from Roboflow, designed for fine-tuning and also extended toward instance segmentation.
#
# Classroom framing:
#
# > Detection is not a solved historical topic; teams still compete on the accuracy-latency-deployment tradeoff.
#
# Links:
#
# - YOLO26 docs: https://docs.ultralytics.com/models/yolo26/
# - YOLO26 end-to-end detection guide: https://docs.ultralytics.com/guides/end2end-detection/
# - RF-DETR GitHub: https://github.com/roboflow/rf-detr
# - RF-DETR model overview: https://roboflow.com/model/rf-detr

# %%
print("YOLO/RF-DETR note: the real YOLO-family inference demo already ran in Part 4.")
print("For RF-DETR, use the project docs when you want a detector-transformer fine-tuning demo: https://github.com/roboflow/rf-detr")

# %% [markdown]
# ## DINOv3: self-supervised dense visual features
#
# ResNet embeddings in this notebook came from supervised ImageNet training. A major modern direction is **self-supervised visual representation learning**: train on large image collections without manual labels, then reuse the backbone for many tasks.
#
# DINOv3 is a useful example because it emphasizes high-resolution dense features from a general visual backbone. Dense features matter for tasks like detection, segmentation, depth estimation, and tracking, not just one-vector-per-image retrieval.
#
# Classroom framing:
#
# > Image embeddings are not only final vectors. Modern backbones can produce dense feature maps where every patch has a useful representation.
#
# Links:
#
# - DINOv3 blog: https://ai.meta.com/blog/dinov3-self-supervised-vision-model/
# - DINOv3 page: https://ai.meta.com/dinov3/

# %% [markdown]
# ## SigLIP 2 and Florence-2: CV becomes multimodal and prompt-driven
#
# CLIP is still a great teaching anchor, but current systems are often stronger, more multilingual, and more task-flexible.
#
# - **SigLIP 2**: a family of multilingual vision-language encoders with improved semantic understanding, localization, and dense features. It is a natural update to mention after CLIP-style embeddings.
# - **Florence-2**: a prompt-based vision foundation model. It can use task prompts for captioning, detection, grounding, OCR, and segmentation.
#
# Classroom framing:
#
# > A modern CV model may not expose a separate API for every task. Instead, it may accept an image plus a task prompt and return structured text, boxes, or masks.
#
# Links:
#
# - SigLIP 2 blog: https://huggingface.co/blog/siglip2
# - SigLIP 2 paper: https://arxiv.org/abs/2502.14786
# - Florence-2 Microsoft Research: https://www.microsoft.com/en-us/research/publication/florence-2-advancing-a-unified-representation-for-a-variety-of-vision-tasks/
# - Florence-2 Hugging Face docs: https://huggingface.co/docs/transformers/model_doc/florence2

# %%
modern_cv_map = {
    "CNN basics": "learned filters and feature maps",
    "Detection": "boxes + labels + scores",
    "Segmentation": "pixel masks",
    "SAM 2": "promptable masks for images and video",
    "DINOv3": "self-supervised dense visual features",
    "SigLIP 2": "multilingual image-text embedding space",
    "Florence-2": "prompt-driven vision tasks",
}

for topic, idea in modern_cv_map.items():
    print(f"{topic:14s} -> {idea}")

# %% [markdown]
# ## Wrap-up
#
# The core narrative of this practical is:
#
# 1. pixels are numbers;
# 2. classical CV uses hand-written filters;
# 3. CNNs learn filters and compose them into useful visual features;
# 4. detection localizes objects and segmentation labels pixels;
# 5. pretrained CNNs can turn images into reusable embeddings;
# 6. multimodal models extend the same idea across modalities, so images and text can be compared in one shared space;
# 7. current CV systems are increasingly promptable, real-time, self-supervised, and multimodal.
#
# Suggested debrief questions:
#
# - What was the most concrete evidence that a CNN learns features rather than using raw pixels directly?
# - Which misclassifications made sense visually?
# - Why are embeddings useful even when we do not train a new model?
# - What new tasks become possible once text and images share the same embedding space?
# - Which modern direction feels most useful for real applications: SAM-style prompting, YOLO/RF-DETR detection, DINO-style features, or prompt-driven VLMs?

