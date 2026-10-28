"""Build the notebook for Practical 5 (Training a small Convolutional Neural Network for a Safety Detector).

Run:  python3 build_p5.py
Writes: MEEN41490_P5_Overalls_Detector_STUDENT.ipynb
"""
import json, sys

TEACHER = "--teacher" in sys.argv
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text, title=None, hidden=False):
    src = text.strip("\n")
    if title:
        src = f"# @title {title}\n" + src
    meta = {"cellView": "form"} if hidden else {}
    cells.append({"cell_type": "code", "metadata": meta, "execution_count": None,
                  "outputs": [], "source": src})


# =========================================================================== intro
md(r"""
# Practical 5 — Training a small Convolutional Neural Network for a Safety Detector

**AI in Engineering (MEEN41490)**

In the demonstration, the Safety Monitor finds every person in a video with YOLOv8, a pretrained detector. Then, it decides who is authorised to stand in the restricted zone. That decision uses a colour rule. A person counts as "authorised staff" when more than a set percentage of the pixels in their box fall inside a hand-tuned range of red.

The rule works on the video it was tuned for. However, the workshop has many cameras and the light changes from one camera to the next. Under a darker camera, the red overalls fall outside the tuned range. The monitor then reports an authorised worker as an intruder.

In this practical, you replace the colour rule with a **trained model**. Instead of writing the rule yourself, you give the computer examples of workers in overalls and people without them. The model adjusts its internal numbers until it separates the two groups. Your challenge is to design and calibrate that model. At the end, you plug your best model into the safety monitor and run it on the scenes and the three videos from the demonstration.

**How this notebook works.** Run the cells in order. To run a cell, click the play button on its left. In Part 3 you edit code directly, so read Parts 1 and 2 with care. Part 0 of the practical page refreshes the ideas that the challenge relies on, i.e., images as arrays, convolutions, pooling and training.

### The data of the practice

The notebook uses ten data files, which sit on the practical page under "The data of the session". They are the crops (`overalls_crops.zip`), six camera snapshots (`scene_*.jpg`) and the three videos of the demonstration (`workshop_walk.mp4`, `workshop_walk2.mp4` and `multiple_workers.mp4`).

You do not need to download or upload anything. The setup cell below installs the libraries, downloads the ten files into Colab, unpacks the crops and prepares the helper functions. When it finishes, the files appear in the Files panel of Colab (the folder icon on the left). If Colab restarts, run the setup cell again, and it downloads the files again.
""")

# =========================================================================== setup
code(r"""
# ---- Libraries -------------------------------------------------------------
# ultralytics gives you YOLOv8, the person detector used in the demonstration.
# The other libraries are preinstalled in Colab.
import importlib.util, subprocess, sys
if importlib.util.find_spec("ultralytics") is None:
    # Keep the preinstalled versions of the core libraries (Colab breaks if pip upgrades them).
    import numpy, matplotlib, torch
    with open("/tmp/constraints.txt", "w") as f:
        f.write(f"numpy=={numpy.__version__}\nmatplotlib=={matplotlib.__version__}\ntorch=={torch.__version__}\n")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-c", "/tmp/constraints.txt", "ultralytics"], check=True)

import os, glob, zipfile, random, time, urllib.request
import numpy as np                 # NumPy: arrays of numbers (an image is an array)
import pandas as pd                # pandas: tables
import cv2                         # OpenCV: reading images and video, colour conversions
import matplotlib.pyplot as plt    # Matplotlib: charts and image grids
import matplotlib.patches as patches
import torch                       # PyTorch: the library that builds and trains the model
import torch.nn as nn

# ---- Data files ------------------------------------------------------------
# The cell downloads the data files from the course website into Colab. It skips any
# file that is already in the Files panel, e.g., when you run this cell a second time.
DATA_URL = "https://juanbmedina.github.io/ai-engineering-practicals/practicals/uniform_detector/assets/data/"
FILES = ["overalls_crops.zip",
         "workshop_walk.mp4", "workshop_walk2.mp4", "multiple_workers.mp4",
         "scene_002.jpg", "scene_021.jpg", "scene_032.jpg", "scene_033.jpg", "scene_047.jpg", "scene_099.jpg"]
for name in FILES:
    if not os.path.exists(name):
        try:
            urllib.request.urlretrieve(DATA_URL + name, name + ".part")
            os.rename(name + ".part", name)          # only complete downloads get the final name
        except Exception as e:
            print(f"Could not download {name}: {e}")
missing = [n for n in FILES if not os.path.exists(n)]
if missing:
    raise SystemExit(f"Could not download {missing}. Run this cell again. If the error repeats, download "
                     "the files from the practical page, upload them in the Files panel (folder icon on the left) "
                     "and run this cell once more.")
if not os.path.isdir("dataset"):
    zipfile.ZipFile("overalls_crops.zip").extractall(".")

random.seed(0); np.random.seed(0); torch.manual_seed(0)
print("Setup done. Files in the dataset folder:")
for split in ["train", "test"]:
    for label in ["overalls", "no_overalls"]:
        n = len(glob.glob(f"dataset/{split}/{label}/*.jpg"))
        print(f"  {split:5s} / {label:11s}: {n:3d} images")

# ---- Helper functions ------------------------------------------------------
# These functions draw the figures and print the tables of the notebook. You call some of
# them later (show_grid, show_architecture), and the training functions of Part 2 call the others.

def show_grid(images, titles, columns=6, size=1.9, colours=None):
    # A grid of images with one title above each image.
    rows = (len(images) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(columns * size, rows * (size + 0.35)))
    for i, ax in enumerate(np.array(axes).flatten()):
        ax.axis("off")
        if i < len(images):
            ax.imshow(images[i])
            ax.set_title(titles[i], fontsize=9, color=(colours[i] if colours else "black"))
    plt.tight_layout(); plt.show()

def show_architecture(model, title="model"):
    # Draws a model as a chain of boxes, with the shape of the data after each layer.
    # It passes one dummy crop through the layers and records the shape after each one.
    x = torch.zeros(1, 3, 64, 64)
    stages = [("input", tuple(x.shape[1:]))]
    with torch.no_grad():
        for layer in model:
            x = layer(x)
            name = type(layer).__name__
            if isinstance(layer, nn.Conv2d):
                name = f"Conv2d\n{layer.out_channels} kernels\n{layer.kernel_size[0]} x {layer.kernel_size[1]}"
            elif isinstance(layer, nn.MaxPool2d):
                name = f"MaxPool2d\n{layer.kernel_size} x {layer.kernel_size}"
            elif isinstance(layer, nn.Linear):
                name = f"Linear\n{layer.in_features} -> {layer.out_features}"
            stages.append((name, tuple(x.shape[1:])))
    colours = {"Conv2d": "#f4a261", "ReLU": "#e9ecef", "MaxPool2d": "#90be6d", "Flatten": "#e9ecef", "Linear": "#8ecae6", "input": "#ffd166"}
    fig, ax = plt.subplots(figsize=(1.6 * len(stages) + 1, 3.2))
    for i, (name, shape) in enumerate(stages):
        colour = colours.get(name.split("\n")[0], "#e9ecef")
        ax.add_patch(patches.FancyBboxPatch((i * 1.6, 0.9), 1.3, 1.5, boxstyle="round,pad=0.05", facecolor=colour, edgecolor="black"))
        ax.text(i * 1.6 + 0.65, 1.65, name, ha="center", va="center", fontsize=8.5)
        shape_text = " x ".join(str(s) for s in shape) if len(shape) > 1 else (f"{shape[0]} numbers" if shape[0] > 1 else "1 number")
        ax.text(i * 1.6 + 0.65, 0.45, shape_text, ha="center", va="center", fontsize=8.5, color="#333")
        if i < len(stages) - 1:
            ax.annotate("", xy=((i + 1) * 1.6, 1.65), xytext=(i * 1.6 + 1.3, 1.65), arrowprops=dict(arrowstyle="->"))
    n_params = sum(p.numel() for p in model.parameters())
    ax.set_xlim(-0.2, len(stages) * 1.6); ax.set_ylim(0, 3); ax.axis("off")
    ax.set_title(f"{title}: {n_params:,} parameters   (orange = convolution, green = pooling, blue = linear)", fontsize=10)
    plt.show()

def predict_probability(model, X):
    # The sigmoid turns the score of the model into a number between 0 and 1.
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(X).squeeze(1))

def accuracy(model, X, y):
    # Share of crops whose label the model gets right.
    pred = (predict_probability(model, X) > 0.5).long()
    return (pred == y).float().mean().item()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def plot_history(history, title):
    # The loss curve and the two accuracy curves of one training run.
    epochs = range(1, len(history["loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.2))
    ax1.plot(epochs, history["loss"], marker="o"); ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.set_title("Loss on the training set")
    ax1.set_ylim(0, max(0.75, max(history["loss"]) * 1.05))
    for ax in (ax1, ax2): ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax2.plot(epochs, history["train_acc"], marker="o", label="training set")
    ax2.plot(epochs, history["test_acc"], marker="s", label="test set")
    ax2.axhline(0.95, color="gray", linestyle="--", linewidth=1); ax2.text(1, 0.955, "target 0.95", fontsize=8, color="gray")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("accuracy"); ax2.set_ylim(0.4, 1.02); ax2.set_title("Accuracy"); ax2.legend(loc="lower right")
    fig.suptitle(title, fontsize=10); plt.tight_layout(); plt.show()

def show_log():
    # Prints the experiment log of Part 2 as a table, with the "target reached" column of the challenge.
    table = pd.DataFrame([{k: v for k, v in r.items() if k != "model"} for r in experiment_log])
    table["target reached"] = np.where((table["test_accuracy"] >= 0.95) & (table["parameters"] < 50000) & (table["seconds"] <= 60), "yes", "")
    print(table.to_string(index=False))

print("Helper functions ready.")
""", title="Setup: libraries, data and helper functions (about 1 minute)", hidden=True)

# =========================================================================== Part 1
md(r"""
## Part 1 — The data

### 1.1 Loading the crops

Each crop is a 96 × 96 image of one person, cut by YOLOv8 from a workshop camera snapshot. A person labelled every crop as `overalls` or `no_overalls`. The cell reads them into arrays, counts them, and shows twelve random examples.
""")
code(r"""
LABELS = ["no_overalls", "overalls"]      # label 0 and label 1

def load_split(split):
    images, labels = [], []
    for label_id, label in enumerate(LABELS):
        for path in sorted(glob.glob(f"dataset/{split}/{label}/*.jpg")):
            bgr = cv2.imread(path)                          # OpenCV reads colours as BGR
            images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            labels.append(label_id)
    return np.array(images), np.array(labels)

train_images, train_labels = load_split("train")
test_images,  test_labels  = load_split("test")
print("Training set:", train_images.shape, "-> images, height, width, colour channels")
print("Test set:    ", test_images.shape)
print("Training labels:", np.bincount(train_labels)[0], "no_overalls,", np.bincount(train_labels)[1], "overalls")
print("Test labels:    ", np.bincount(test_labels)[0], "no_overalls,", np.bincount(test_labels)[1], "overalls")

idx = np.random.choice(len(train_images), 12, replace=False)
show_grid(train_images[idx], [LABELS[train_labels[i]] for i in idx])      # show_grid is one of the helper functions
""", title="Load the crops and show twelve examples", hidden=True)

md(r"""
### 1.2 The colour rule from the demonstration

The demonstration decides "red overalls" with a rule. It converts the crop to the HSV colour space, counts the pixels whose hue sits near red, and says `overalls` when those pixels cover more than 15% of the box, which is the default threshold of the demonstration. The cell applies the same rule, with the default range and threshold, to the test set. This is the score to beat.
""")
code(r"""
def demo_colour_rule(rgb, threshold=0.15):
    # Same thresholds as the Safety Monitor: hue 0-10 and 170-179, saturation >= 20, value >= 20.
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (0, 20, 20), (10, 255, 255)) + cv2.inRange(hsv, (170, 20, 20), (179, 255, 255))
    red_share = cv2.countNonZero(mask) / mask.size
    return int(red_share > threshold), red_share

rule_pred = np.array([demo_colour_rule(img)[0] for img in test_images])
rule_accuracy = (rule_pred == test_labels).mean()
print(f"Accuracy of the colour rule on the test set: {rule_accuracy:.2f}")
print("Workers in overalls that the rule missed:   ", int(((rule_pred == 0) & (test_labels == 1)).sum()), "of", int((test_labels == 1).sum()))
print("People without overalls that it accepted:   ", int(((rule_pred == 1) & (test_labels == 0)).sum()), "of", int((test_labels == 0).sum()))

wrong = np.where(rule_pred != test_labels)[0][:6]
print("\nSome workers that the rule missed (their overalls look dark under these cameras):")
show_grid(test_images[wrong], [f"red share {demo_colour_rule(test_images[i])[1]:.2f}" for i in wrong], columns=6)
""", title="Measure the colour rule of the demonstration on the test set", hidden=True)

# =========================================================================== Part 2
md(r"""
## Part 2 — The training loop and a first model

From this part on, the notebook uses **PyTorch**, a free Python library that builds and trains neural networks. PyTorch provides the layers of a model as ready-made blocks, e.g., `nn.Conv2d` and `nn.Linear`. It also computes the gradient steps of training for you. In the code, `torch` is the library and `nn` is its collection of layers.

### 2.1 Preparing the arrays

The model takes crops of 64 × 64 pixels, with the values scaled from 0–255 to −0.5–0.5. The cell builds the training and test tensors. A **tensor** is PyTorch's name for an array.
""")
code(r"""
def to_tensor(images, size=64):
    small = np.array([cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA) for img in images])
    return torch.tensor(small, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0 - 0.5

X_train, y_train = to_tensor(train_images), torch.tensor(train_labels)
X_test,  y_test  = to_tensor(test_images),  torch.tensor(test_labels)
print("Training tensor:", tuple(X_train.shape), "-> images, colour channels, height, width")
print("Test tensor:    ", tuple(X_test.shape))
""", title="Prepare the training and test tensors", hidden=True)

md(r"""
### 2.2 The training loop

The cell below defines the two functions that you call in the challenge. You need to know what each function does, because the challenge asks you to read what they print. Double-click the title if you want to read the code and its comments.

- `train_model` runs the gradient steps for a number of epochs and records the loss and the accuracies after each epoch. This is the training loop of Part 0.5 of the practical page, written in code.
- `run_experiment` trains a model from fresh random parameters, measures its test accuracy, its number of parameters and its training time, draws the curves, and adds one row to an **experiment log**. The log is a table with every model you have tried. Keeping such a log is how engineers tune a model without losing track.

The curves and the table come from the helper functions `plot_history` and `show_log`, which the setup cell prepared.
""")
code(r"""
def train_model(model, X, y, epochs, learning_rate, batch_size):
    loss_function = nn.BCEWithLogitsLoss()                         # error between score and label
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = {"loss": [], "train_acc": [], "test_acc": []}
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(X))                            # shuffle the training crops
        losses = []
        for start in range(0, len(X), batch_size):
            batch = order[start:start + batch_size]
            optimizer.zero_grad()
            score = model(X[batch]).squeeze(1)
            loss = loss_function(score, y[batch].float())
            loss.backward()                                        # direction that lowers the loss, for each parameter
            optimizer.step()                                       # move every parameter one step
            losses.append(loss.item())
        history["loss"].append(float(np.mean(losses)))
        history["train_acc"].append(accuracy(model, X, y))
        history["test_acc"].append(accuracy(model, X_test, y_test))
    return history

experiment_log = []      # one row per model you train

def run_experiment(name, model, epochs, learning_rate, batch_size=32, seed=0):
    torch.manual_seed(seed)
    for layer in model:                                            # fresh random parameters for each run
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
    t0 = time.time()
    history = train_model(model, X_train, y_train, epochs, learning_rate, batch_size)
    seconds = time.time() - t0
    row = {"run": len(experiment_log) + 1, "name": name, "parameters": count_parameters(model),
           "epochs": epochs, "learning_rate": learning_rate, "batch_size": batch_size,
           "seconds": round(seconds, 1), "train_accuracy": round(history["train_acc"][-1], 3),
           "test_accuracy": round(accuracy(model, X_test, y_test), 3), "model": model}
    experiment_log.append(row)
    plot_history(history, f"run {row['run']}: {name}")                # draw the curves (helper function)
    show_log()                                                         # print the experiment log (helper function)
    return model

print("Training functions ready.")
""", title="Training functions: train_model and run_experiment", hidden=True)

md(r"""
### 2.3 A first model

The cell defines a small model and trains it. This is your starting point for the challenge. In PyTorch, a model is a chain of layers inside `nn.Sequential`. The crop enters the first layer, and the output of each layer feeds the next one. The first model uses five layers.

- `nn.Conv2d(3, 4, kernel_size=3, padding=1)` is a **convolutional layer**. It slides 4 kernels of 3 × 3 over the 3 colour channels of the crop and produces 4 feature maps of 64 × 64. The first argument is the number of channels that arrive, and the second is the number of kernels.
- `nn.ReLU()` replaces every negative number of the feature maps by 0.
- `nn.MaxPool2d(8)` is a **pooling layer**. It divides each feature map into squares of 8 × 8 and keeps only the largest number of each square. The maps shrink from 64 × 64 to 8 × 8, and the model keeps the strongest responses without their exact position.
- `nn.Flatten()` lays all the numbers of the 4 maps of 8 × 8 in one long row of 4 × 8 × 8 = 256 numbers. Flatten has no parameters. It only changes the shape, so that a linear layer can read the numbers.
- `nn.Linear(256, 1)` is a **linear layer**. It multiplies each of the 256 numbers by a weight, adds the products, and adds a bias. The result is one score. This is the neuron of Practical 2, applied to 256 inputs instead of four. The first argument must equal the number of numbers that arrive, and the second is the number of outputs.

The sigmoid, which turns the score into a confidence between 0 and 1, sits inside the helper `predict_probability`. Part 0 of the practical page explains each of these layers with a figure.

The function `show_architecture` draws the model as a chain of boxes, with the shape of the data after each layer and the number of parameters. The model trains for 5 epochs with a learning rate of 0.001. Run the cell and read the output. The log table shows the test accuracy, the number of parameters and the training time. The curves show how the loss and the accuracies changed over the epochs.
""")
code(r"""
first_model = nn.Sequential(
    nn.Conv2d(3, 4, kernel_size=3, padding=1),   # convolution: 4 kernels of 3 x 3 -> 4 feature maps of 64 x 64
    nn.ReLU(),                                    # negative numbers become 0
    nn.MaxPool2d(8),                              # pooling: keep the largest number of each 8 x 8 square -> 8 x 8
    nn.Flatten(),                                 # lay the 4 x 8 x 8 = 256 numbers in one row
    nn.Linear(256, 1),                            # linear layer: 256 numbers x 256 weights -> one score
)
show_architecture(first_model, "first model")
run_experiment("first model", first_model, epochs=5, learning_rate=0.001)
""")
md(r"""
The first model beats the colour rule, but not by much. The loss is still going down when the training stops. The challenge starts here.
""")

# =========================================================================== Part 3
md(r"""
## Part 3 — The challenge: calibrate the model (30 minutes)

Your goal is a model that reaches a **test accuracy of 0.95 or more**, with **fewer than 50,000 parameters** and **at most 60 seconds of training**. Among the models that reach the target, the one with the fewest parameters wins. The log marks every run that meets the three conditions with "yes" in the last column.

You can change two kinds of things.

**The training.** The number of epochs, the learning rate, and the batch size are the arguments of `run_experiment`.

**The architecture.** Inside `nn.Sequential` you can change the number of kernels of a convolution (its second argument), the kernel size (keep `padding = kernel_size // 2` so that the map keeps its size), the size of the pooling, the number of convolutional layers, and the number of linear layers. The rules of shape are simple. A `Conv2d` with `in` and `out` channels needs `in` to match the number of maps that arrive. A `MaxPool2d(k)` divides the height and the width by `k`. The first `Linear` layer needs `in_features` equal to channels × height × width of the maps that reach `Flatten`. If you get a shape wrong, the error message tells you the numbers that did not match, and `show_architecture` shows the shapes after each layer.

**How to work.**

1. Copy the cell below (or edit it in place), change **one thing**, give the run a short name that says what you changed, and run it. Every run adds a row to the log, so you can always compare.
2. Look at the curves before you decide the next change. A loss that is still going down asks for more epochs. A loss that jumps up and down asks for a smaller learning rate. A loss that stays flat near 0.69 asks for a larger learning rate or a different architecture. A training accuracy near 1.00 with a lower test accuracy means that the model memorises the training crops. More data or a smaller model help there.
3. When a run reaches the target, run it again with `seed=1`. A good configuration reaches the target with both seeds.
4. Then, try to reach the target with fewer parameters.

Keep each training under a minute. Training takes a few seconds for small models, so you have time for many runs.
""")
code(r"""
# Edit this cell. Change one thing at a time and give each run a name that says what you changed.
my_model = nn.Sequential(
    nn.Conv2d(3, 4, kernel_size=3, padding=1),   # 4 kernels of 3 x 3   (try more kernels, or kernel_size=5 with padding=2)
    nn.ReLU(),
    nn.MaxPool2d(8),                              # 64 x 64 -> 8 x 8     (try MaxPool2d(4), or add a second Conv2d + ReLU + MaxPool2d block)
    nn.Flatten(),
    nn.Linear(256, 1),                            # 256 = 4 kernels x 8 x 8  (update this number when you change the layers above)
)
show_architecture(my_model, "my model")
run_experiment("attempt 1: same as first model", my_model, epochs=5, learning_rate=0.001, batch_size=32, seed=0)
""")
if TEACHER:
    md(r"""
**Teacher notes — a worked path through the challenge.** The cells below show one sequence of edits with the result observed on the CPU (seed 0). The first model stops too early and steps too small. Raising the learning rate alone reaches the target with 369 parameters, which is the smallest model we found. A second convolutional block also reaches it, but with more parameters and a result that depends on the seed. Expected timings are in the log.
""")
    code(r"""
my_model = nn.Sequential(nn.Conv2d(3, 4, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(8), nn.Flatten(), nn.Linear(256, 1))
run_experiment("attempt 2: 15 epochs", my_model, epochs=15, learning_rate=0.001)
""", title="Teacher: attempt 2 — more epochs (loss still falling)")
    code(r"""
run_experiment("attempt 3: learning rate 0.01", my_model, epochs=15, learning_rate=0.01)
""", title="Teacher: attempt 3 — larger learning rate (reaches the target with 369 parameters)")
    code(r"""
run_experiment("attempt 4: same, seed 1", my_model, epochs=15, learning_rate=0.01, seed=1)
""", title="Teacher: attempt 4 — check with another seed")
    code(r"""
my_model = nn.Sequential(
    nn.Conv2d(3, 8, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(4),
    nn.Flatten(), nn.Linear(1024, 1))
show_architecture(my_model, "two convolutional blocks")
run_experiment("attempt 5: two conv blocks, lr 0.001", my_model, epochs=15, learning_rate=0.001)
run_experiment("attempt 6: two conv blocks, lr 0.001, seed 1", my_model, epochs=15, learning_rate=0.001, seed=1)
""", title="Teacher: attempts 5 and 6 — a second block (more parameters, seed-dependent at lr 0.001)")
    code(r"""
run_experiment("attempt 7: two conv blocks, lr 0.01", my_model, epochs=15, learning_rate=0.01)
run_experiment("attempt 8: two conv blocks, lr 0.1 (too large)", my_model, epochs=15, learning_rate=0.1)
""", title="Teacher: attempts 7 and 8 — learning rate on the two-block model")

md(r"""
**Two examples of valid edits**, to show the shape rules at work. Do not run them as they are. Use them as a guide.

A second convolutional block:

```python
nn.Sequential(
    nn.Conv2d(3, 8, kernel_size=3, padding=1),   # 8 maps of 64 x 64
    nn.ReLU(),
    nn.MaxPool2d(2),                              # 8 maps of 32 x 32
    nn.Conv2d(8, 16, kernel_size=3, padding=1),  # 16 maps of 32 x 32   (in = 8, the maps that arrive)
    nn.ReLU(),
    nn.MaxPool2d(4),                              # 16 maps of 8 x 8
    nn.Flatten(),                                 # 16 x 8 x 8 = 1024 numbers
    nn.Linear(1024, 1),
)
```

A hidden linear layer before the score:

```python
nn.Sequential(
    nn.Conv2d(3, 4, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(8),
    nn.Flatten(),                                 # 256 numbers
    nn.Linear(256, 16),                           # 16 intermediate numbers
    nn.ReLU(),
    nn.Linear(16, 1),
)
```

When you have finished, run the cell below. It picks the best run of your log and keeps its model for Part 4.
""")
code(r"""
reached = [r for r in experiment_log if r["test_accuracy"] >= 0.95 and r["parameters"] < 50000 and r["seconds"] <= 60]
if reached:
    best = min(reached, key=lambda r: r["parameters"])
    print(f"Best run that reached the target: run {best['run']} ({best['name']}), "
          f"{best['parameters']:,} parameters, test accuracy {best['test_accuracy']:.3f}")
else:
    best = max(experiment_log, key=lambda r: r["test_accuracy"])
    print(f"No run reached the target yet. Using the most accurate run: run {best['run']} ({best['name']}), "
          f"test accuracy {best['test_accuracy']:.3f}")
best_model = best["model"]

test_prob = predict_probability(best_model, X_test).numpy()
model_pred = (test_prob > 0.5).astype(int)
tp = int(((model_pred == 1) & (test_labels == 1)).sum()); fn = int(((model_pred == 0) & (test_labels == 1)).sum())
fp = int(((model_pred == 1) & (test_labels == 0)).sum()); tn = int(((model_pred == 0) & (test_labels == 0)).sum())
print(f"\nColour rule accuracy on the test set: {rule_accuracy:.2f}")
print(f"Best model accuracy on the test set:  {(model_pred == test_labels).mean():.2f}")
print("\nConfusion matrix of the best model")
print("                          model says overalls   model says no_overalls")
print(f"  label overalls                 {tp:3d}                    {fn:3d}")
print(f"  label no_overalls              {fp:3d}                    {tn:3d}")
wrong = np.where(model_pred != test_labels)[0]
if len(wrong):
    print(f"\nThe {len(wrong)} mistakes of the best model, with its confidence in 'overalls':")
    show_grid(test_images[wrong[:12]], [f"label {LABELS[test_labels[i]]}\nconfidence {test_prob[i]:.2f}" for i in wrong[:12]], columns=6)
""", title="Pick the best model", hidden=True)

# =========================================================================== Part 4
md(r"""
## Part 4 — Your model inside the safety monitor

The model only knows crops. The safety monitor needs a decision for every person in a full camera snapshot. This part repeats the pipeline of the demonstration. YOLOv8 finds the people, the notebook cuts a square crop around each person, and your best model decides. A green box means "authorised" and a red box means "not authorised". The tag under each box shows what the colour rule would have said.

The first cell loads YOLOv8 and the four functions of the pipeline, with its code collapsed behind the title. `square_crop` cuts a square around a box, and `classify_crop` asks your model for its confidence on that square. `annotate` runs the detector on one frame and draws the boxes, and `annotate_video` does the same on one frame out of three of a video. The practical page explains the lines that matter. Double-click the title if you want to read the whole code.
""")
code(r"""
from ultralytics import YOLO
detector = YOLO("yolov8s.pt")      # downloads the pretrained weights the first time (about 22 MB)

def square_crop(bgr, box, margin=1.15):
    # Cuts a square around the box, with a margin of 15%, padded with black at the edges of the frame.
    x1, y1, x2, y2 = box
    cx, cy, s = (x1 + x2) / 2, (y1 + y2) / 2, int(max(x2 - x1, y2 - y1) * margin)
    padded = cv2.copyMakeBorder(bgr, s, s, s, s, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    sx, sy = int(cx - s / 2) + s, int(cy - s / 2) + s
    return padded[sy:sy + s, sx:sx + s]

def classify_crop(bgr_crop):
    # Resizes the square to 96 x 96 and returns the confidence of your best model and the answer of the colour rule.
    rgb = cv2.cvtColor(cv2.resize(bgr_crop, (96, 96), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    prob = predict_probability(best_model, to_tensor(rgb[None]))[0].item()
    return prob, demo_colour_rule(rgb)[0]

def annotate(bgr, imgsz=1280):
    # Runs YOLOv8 on one frame (class 0 = person), classifies each box and draws it.
    result = detector.predict(bgr, classes=[0], conf=0.35, imgsz=imgsz, verbose=False)[0]
    out = bgr.copy()
    for b in result.boxes:
        box = [int(v) for v in b.xyxy[0].tolist()]
        prob, rule = classify_crop(square_crop(bgr, box))
        colour = (0, 180, 0) if prob > 0.5 else (0, 0, 230)
        x1, y1, x2, y2 = box
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 3)
        cv2.putText(out, f"model {prob:.2f}", (x1, max(y1 - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
        cv2.putText(out, f"rule: {'overalls' if rule else 'no overalls'}", (x1, y2 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out

def annotate_video(filename, every=3, show=6):
    # Runs annotate() on one frame out of `every`, shows `show` of them, and writes <name>_annotated.mp4.
    out_name = os.path.splitext(filename)[0] + "_annotated.mp4"
    cap = cv2.VideoCapture(filename)
    fps, W, H = cap.get(cv2.CAP_PROP_FPS), int(cap.get(3)), int(cap.get(4))
    writer = cv2.VideoWriter("_tmp.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps / every, (W, H))
    frames, n = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if n % every == 0:
            out = annotate(frame, imgsz=960)
            writer.write(out); frames.append(out)
        n += 1
    cap.release(); writer.release()
    os.system(f"ffmpeg -y -loglevel error -i _tmp.mp4 -vcodec libx264 -pix_fmt yuv420p {out_name}")
    print(f"{filename}: processed {len(frames)} frames of {W} x {H}. Annotated video saved as {out_name}")
    pick = np.linspace(0, len(frames) - 1, show).astype(int)
    show_grid([cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB) for i in pick],
              [f"{filename}, frame {every * i}" for i in pick], columns=3, size=4.5)
    return frames

print("YOLOv8 and the safety monitor pipeline ready.")
""", title="Load YOLOv8 and the pipeline of the safety monitor (about 30 seconds)", hidden=True)

md(r"""
### 4.1 Camera snapshots

The cell runs the pipeline on two snapshots of the workshop. Compare the colour of each box with the tag under it.
""")
code(r"""
for name in ["scene_021.jpg", "scene_047.jpg"]:
    out = annotate(cv2.imread(name))
    plt.figure(figsize=(13, 8)); plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)); plt.axis("off"); plt.title(name); plt.show()
""", title="Run the pipeline on two camera snapshots", hidden=True)

md(r"""
### 4.2 The video from the demonstration

The cell runs the same pipeline on the first video of the demonstration. The function `annotate_video` processes one frame out of every three and shows six of them. It also writes the annotated video to `workshop_walk_annotated.mp4`, which you can download from the Files panel. Watch the worker who walks into the yellow zone. Does your model keep a green box on him from frame to frame?
""")
code(r"""
frames = annotate_video("workshop_walk.mp4")
""")

md(r"""
### 4.3 Try your model on the three videos

In Section 1 you calibrated the mask on three videos. Your model deserves the same test. The cell runs it on the other two videos. `workshop_walk2.mp4` comes from the same camera as the first video. `multiple_workers.mp4` comes from another camera, with several workers inside the zone and other people walking past it. Each run takes about a minute on the CPU.

Look at the frames and answer these questions with your team. Does the model keep the same answer for one person from frame to frame? Does it accept every worker in overalls, and nobody else? Which video gives it the most trouble, and why? If the model fails on the second camera, think about the crops it trained on. Then, go back to Part 3, try to improve the model, run the cell that picks the best model, and run this cell again.
""")
code(r"""
for name in ["workshop_walk2.mp4", "multiple_workers.mp4"]:
    frames = annotate_video(name)
""")

# =========================================================================== finish
md(r"""
## You have finished the practical

Discuss these questions with your team.

1. Look at your log. Which change gave the largest gain in test accuracy? Which change cost the most parameters for the smallest gain?
2. Which errors matter more for a safety monitor, a missed intruder or a false alarm on a worker? Look at the confusion matrix of your best model.
3. You compared your models on the test set and then reported the accuracy of the best one on the same test set. Why is that number optimistic? What would you change in a real project?
4. The crops come from one simulated workshop. What would you expect on a video from a different building, with a different floor colour? What data would you need to collect?
5. The overalls could change colour next year. Compare the effort of updating the colour rule with the effort of updating your model.

## Advanced 1 — What the first layer learned

Run the cell below. It shows the kernels of the first convolutional layer of your best model and the feature maps they produce on one crop. Compare them with the hand-written kernels of Part 0.2 of the practical page.
""")
code(r"""
first_conv = [layer for layer in best_model if isinstance(layer, nn.Conv2d)][0]
with torch.no_grad():
    maps = torch.relu(first_conv(X_test[:1]))[0]
n = min(first_conv.out_channels, 8)
fig, axes = plt.subplots(2, n + 1, figsize=(1.8 * (n + 1), 3.8))
axes[0, 0].imshow(test_images[0]); axes[0, 0].set_title("crop", fontsize=9)
axes[1, 0].axis("off")
for k in range(n):
    kernel = first_conv.weight[k].detach().numpy()                 # 3 x kh x kw: one 2-D kernel per colour channel
    rgb_kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min() + 1e-6)
    axes[0, k + 1].imshow(np.transpose(rgb_kernel, (1, 2, 0))); axes[0, k + 1].set_title(f"kernel {k + 1}", fontsize=9)
    axes[1, k + 1].imshow(maps[k], cmap="gray"); axes[1, k + 1].set_title(f"feature map {k + 1}", fontsize=9)
for ax in axes.flatten(): ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.show()
""", title="Advanced 1 — Kernels and feature maps of the first layer", hidden=True)

md(r"""
## Advanced 2 — A model that sees one pixel at a time

A convolution with `kernel_size=1` looks at a single pixel, so it can only combine the three colour values of that pixel. Such a model is a learned colour rule. Train one with the cell below and compare it with your best model. Then, think about what a 3 × 3 kernel can see that a 1 × 1 kernel cannot.
""")
code(r"""
pixel_model = nn.Sequential(
    nn.Conv2d(3, 4, kernel_size=1),    # one pixel at a time
    nn.ReLU(),
    nn.MaxPool2d(8),
    nn.Flatten(),
    nn.Linear(256, 1),
)
run_experiment("advanced 2: 1 x 1 kernels", pixel_model, epochs=15, learning_rate=0.01)
""")

md(r"""
## Advanced 3 — Less data

Overfitting appears when the training set is small. The cell trains your best architecture on only 40 crops. Compare the two accuracy curves. The training accuracy climbs towards 1.00 and the test accuracy stays behind. The model memorises the 40 crops instead of learning the overalls. The gap grows with the size of the model.
""")
code(r"""
import copy
small_model = copy.deepcopy(best_model)
torch.manual_seed(0)
for layer in small_model:                     # start again from random parameters
    if hasattr(layer, "reset_parameters"):
        layer.reset_parameters()
few = torch.randperm(len(X_train))[:40]
history = train_model(small_model, X_train[few], y_train[few], epochs=40, learning_rate=0.003, batch_size=8)
plot_history(history, "best architecture trained on 40 crops")
""", title="Advanced 3 — Train the best architecture on 40 crops", hidden=True)

md(r"""
**Something went wrong?**

- If the setup cell cannot download the data files, run it again. If the error repeats, download the ten files from the practical page, upload them in the Files panel (folder icon on the left), and run the setup cell once more.
- `RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x1024 and 256x1)`. The first `Linear` layer expects 256 numbers but receives 1024. Set its first argument to the number that `show_architecture` prints before `Linear`.
- `RuntimeError: ... expected input[32, 8, 32, 32] to have 4 channels`. A `Conv2d` expects 4 maps but receives 8. Set its first argument to the number of kernels of the convolution before it.
- If training is slow, reduce the number of kernels or add pooling earlier. Models under 50,000 parameters train in well under a minute on the CPU.
- If Part 4 shows no boxes, run the first cell of Part 4 again so that the YOLOv8 weights download.
""")

nb = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4, "nbformat_minor": 0,
}
out_name = "MEEN41490_P5_Overalls_Detector_TEACHER.ipynb" if TEACHER else "MEEN41490_P5_Overalls_Detector_STUDENT.ipynb"
with open(out_name, "w") as f:
    json.dump(nb, f, indent=1)
print("written", len(cells), "cells")
