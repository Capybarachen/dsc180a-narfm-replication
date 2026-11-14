#!/usr/bin/env python
# coding: utf-8

# # Replicating our paper's CIFAR-10 experiment: the starter
# 
# Our paper (Wang, Belkin and Wang, 2026, [arXiv 2602.11395](https://arxiv.org/abs/2602.11395)) steers a pretrained
# diffusion model toward a class without retraining it, in two stages:
# - **noise alignment (NA)** in the early, high-noise steps: the CFG recipe with two Gaussian denoisers;
# - **RFM steering** in the later steps: edit the features of one block of the network along a class direction
#   $v_{\mathrm{RFM}}$, then apply the CFG recipe again.
# 
# Table 2 of the paper measures both on CIFAR-10. This notebook installs the paper's code and runs every step once,
# for one class: cars (CIFAR-10's "automobile", class 1), the running example from the Nov 4 slides. The replication
# itself, with all ten classes and five rows of Table 2, is yours; section 5 lists it.
# 
# ## Coding assignment, due Friday, November 13
# 
# First, finish your checkpoint, due this Friday, November 6. Then run this notebook on a GPU (DSMLP or Colab) and
# do the replication in section 5: a table next to the paper's Table 2, sample grids, and a paragraph on what matches
# and what does not. `Nov_4_assignment.pdf` has the rest of the assignment, for this week and next, and what to
# submit.
# 
# **Notation, as on the slides:** $x_1$ is a clean image, $x_0$ is noise and $\sigma$ the noise level, so a noisy
# image is $x = x_1 + \sigma x_0$. The paper writes the clean image as $x_0$ and the noise as $\epsilon$. Its model
# works with $x_t = \sqrt{\bar\alpha_t}\, x_1 + \sqrt{1-\bar\alpha_t}\, x_0$; dividing by $\sqrt{\bar\alpha_t}$ gives
# $x = x_1 + \sigma x_0$ with $\sigma = \sqrt{(1-\bar\alpha_t)/\bar\alpha_t}$, the $\sigma$ in the paper's
# Appendix H.

# ## 0. Setup
# 
# The setup downloads:
# - the paper's code, [na-rfm](https://github.com/isotrivial/na-rfm), with the
#   [TFG code](https://github.com/YWolfeee/Training-Free-Guidance) it builds on, each at a fixed version;
# - the CIFAR-10 diffusion model the paper uses, the U-Net of OpenAI's improved DDPM (210 MB);
# - CIFAR-10, from Hugging Face (140 MB).
# 
# `run` prints a command, runs it in the code's folder, and stops the notebook if it fails.
# 
# Two lines need a fix for current library versions, and the cell makes both:
# - The paper's FID script loads CIFAR-10 from Hugging Face under the dataset's old name, `cifar10`, which current
#   versions of the `datasets` library reject. The fix uses the current name, `uoft-cs/cifar10`.
# - TFG's FID code imports a private part of SciPy that, in SciPy 1.16, fails unless another part loads first. The
#   fix adds that import.

# In[1]:


import hashlib, json, os, subprocess, sys, time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

os.environ["PYTHONUTF8"] = "1"   # Windows: run the paper's scripts in UTF-8

print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none: switch to a GPU runtime")
HOME = Path("/content") if Path("/content").exists() else Path.cwd()   # Colab, or the folder you start in
REPO = HOME / "na-rfm"
TFG = REPO / "baselines" / "tfg"
NARFM_COMMIT = "8e9619bc6735b03d73056bfdc13dd3659b96e127"   # the paper's public code, Oct 27, 2026
TFG_COMMIT = "f8d17f3ec2f0e7377dedf7b7bc62fad15f36cb77"     # Training-Free-Guidance, Oct 29, 2025


def run(cmd, cwd=None):
    print("$", cmd, flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, shell=True, cwd=cwd or REPO, text=True, encoding="utf-8", errors="replace", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = [line for line in proc.stdout.splitlines() if "it/s]" not in line and "s/it]" not in line]   # no progress bars
    print("\n".join(lines[-25:]))
    if proc.returncode != 0:
        raise RuntimeError(f"failed with exit code {proc.returncode}: {cmd}")
    print(f"[{time.time() - t0:.0f} s]")


def fix(path, old, new):
    text = path.read_text(encoding="utf-8")
    if new not in text:
        path.write_text(text.replace(old, new), encoding="utf-8")


if not REPO.exists():
    run(f"git clone -q https://github.com/isotrivial/na-rfm.git {REPO}", cwd=HOME)
run(f"git checkout -q {NARFM_COMMIT}")
if not TFG.exists():
    run(f"git clone -q https://github.com/YWolfeee/Training-Free-Guidance.git {TFG}")
run(f"git checkout -q {TFG_COMMIT}", cwd=TFG)
run(f"{sys.executable} -m pip install -q datasets scikit-learn scipy transformers gdown")

fix(REPO / "repro" / "evaluate_saved_samples.py", 'load_dataset("cifar10")', 'load_dataset("uoft-cs/cifar10")')
sqrtm_import = "from scipy.linalg._matfuncs_sqrtm_triu import within_block_loop"
fix(TFG / "evaluations" / "utils" / "torch_sqrtm.py", sqrtm_import, "import scipy.linalg._matfuncs_sqrtm\n" + sqrtm_import)

os.chdir(REPO)
os.environ["PYTHONPATH"] = f"{REPO}{os.pathsep}{TFG}"   # for the scripts this notebook runs
sys.path[:0] = [str(REPO), str(TFG)]         # for this notebook itself


# The model: na-rfm loads the checkpoint file that TFG distributes, `openai_cifar10.pt`, from TFG's shared
# folder. The check below makes sure you have exactly the file the paper used. If `gdown` fails (Google Drive limits
# popular files), download `openai_cifar10.pt` by hand from
# [TFG's folder](https://drive.google.com/drive/folders/1fS7dKpO4O-FjaLwuRXuHBxEOlkqMMTGh) into
# `na-rfm/data/checkpoints/` and run the cell again.

# In[2]:


CKPT = REPO / "data" / "checkpoints" / "openai_cifar10.pt"
if not CKPT.exists():
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    run(f"gdown 1eInndCjx389Tl45kfmntVo23f_-82bHl -O {CKPT}")
md5 = hashlib.md5(CKPT.read_bytes()).hexdigest()
assert md5 == "2754c3e3c778ef6e0ab98e8aacaefaf9", f"not the paper's checkpoint (md5 {md5}): delete it and download again"
print(f"checkpoint OK: {CKPT.stat().st_size / 1e6:.0f} MB")


# CIFAR-10's 50,000 training images, one folder per class (`label_0` to `label_9`), which is the layout the
# paper's PCA script reads. Class 1 is the cars.

# In[3]:


from datasets import load_dataset
from PIL import Image

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
CAR = 1
BY_CLASS = HOME / "cifar10" / "train_by_class"
cifar = load_dataset("uoft-cs/cifar10", split="train")
assert cifar.features["label"].names == CLASSES
labels = np.array(cifar["label"])
pixels = np.stack([np.asarray(img) for img in cifar["img"]])   # 50,000 x 32 x 32 x 3, values 0 to 255
if not (BY_CLASS / "label_9").exists():
    for i, (img, y) in enumerate(zip(pixels, labels)):
        folder = BY_CLASS / f"label_{y}"
        folder.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(folder / f"{i:05d}.png")
print({CLASSES[k]: len(list((BY_CLASS / f"label_{k}").glob("*.png"))) for k in range(10)})


# ## 1. Noise alignment's two Gaussian denoisers
# 
# Noise alignment needs two Gaussian denoisers, one from the 5,000 training cars ($D_{\mathrm{car}}$) and one from all
# 50,000 training images ($D_{\mathrm{all}}$). Each keeps a mean image $\mu$ and the top $k = 1{,}000$ PCA directions
# $u_k$ with their variances $\nu_k$ (the paper's Table 16), and denoises with the formula from the slides:
# 
# $$D(x;\sigma) = \mu + \sum_k \frac{\nu_k}{\nu_k + \sigma^2}\, \langle x - \mu,\, u_k\rangle\, u_k .$$
# 
# `pca/build_filters.py` computes $\mu$, $u_k$ and $\nu_k$, and saves them as `class_1.pt` and `full_dataset.pt`.
# 
# $D_{\mathrm{all}}$ must see every class, so it is built with `--num-images 0`, which means all 50,000 images. With
# `--full-dataset`, the script keeps the first N files in folder order, so N = 5,000 would be airplanes only.
# 
# This cell takes a few minutes, mostly for the PCA of all 50,000 images.

# In[4]:


FILTERS = REPO / "repro" / "assets" / "cifar10_filters"
build = (f"{sys.executable} pca/build_filters.py --image-root {BY_CLASS} --output-dir {FILTERS} "
         f"--image-size 32 --components 1000 --backend cpu")
if not (FILTERS / "full_dataset.pt").exists():
    run(f"{build} --full-dataset --num-images 0")          # D_all: all 50,000 training images
if not (FILTERS / f"class_{CAR}.pt").exists():
    run(f"{build} --classes {CAR} --num-images 5000")      # D_car: the 5,000 training cars


# What the two Gaussian denoisers keep: the mean image and the first PCA directions, for cars and for all
# images. Grey is zero; a direction is added to or subtracted from the mean.

# In[5]:


def as_image(v):
    return np.clip(v.reshape(3, 32, 32).transpose(1, 2, 0) / 2 + 0.5, 0, 1)   # [-1, 1] -> [0, 1]


def as_direction(u):
    u = u.reshape(3, 32, 32).transpose(1, 2, 0)
    return np.clip(0.5 + 0.5 * u / np.abs(u).max(), 0, 1)


fig, axes = plt.subplots(2, 9, figsize=(13, 3.3))
for row, (name, file) in enumerate([("cars", f"class_{CAR}.pt"), ("all images", "full_dataset.pt")]):
    f = torch.load(FILTERS / file, weights_only=False)
    axes[row, 0].imshow(as_image(f["mean_patch"].numpy()))
    axes[row, 0].set_title(f"mean, {name}", fontsize=9)
    for k in range(8):
        axes[row, k + 1].imshow(as_direction(f["eigenvectors"][:, k].numpy()))
        axes[row, k + 1].set_title(f"direction {k + 1}", fontsize=9)
    print(f"{name}: {f['num_images']} images, {f['eigenvectors'].shape[1]} directions")
for ax in axes.flat:
    ax.axis("off")
plt.show()


# ## 2. The RFM direction for cars
# 
# RFM steering edits the features $h$ of one block of the network: input block 9, with 256 numbers at each of
# 8 × 8 positions, 16,384 in all. To find the car direction $v_{\mathrm{RFM}}$, the paper (Table 16):
# 1. takes 1,000 training images of each class and adds a little noise, $\sigma \approx 0.21$, which is the model's
#    step $t = 60$;
# 2. records $h$ for each image;
# 3. fits an RFM classifier, cars against the other nine classes, with a Laplace kernel, bandwidth 100,
#    regularization $10^{-3}$ and 5 iterations;
# 4. combines the top three eigenvectors of what RFM learned into $v_{\mathrm{RFM}}$.
# 
# The classifier is checked on 20% of the images it did not see; AUC is the probability that it scores a random car
# above a random non-car.

# In[6]:


from torch.utils.data import DataLoader, TensorDataset

from na_rfm.activation_collection import collect_activations
from na_rfm.model_loaders import load_openai_unet
from na_rfm.rfm_training import RFMTrainConfig, get_direction, train_binary_rfm

SIGMA = 0.21
first = np.concatenate([np.flatnonzero(labels == k)[:1000] for k in range(10)])   # the first 1,000 of each class
images = torch.from_numpy(pixels[first]).permute(0, 3, 1, 2).float() / 127.5 - 1   # in [-1, 1]
loader = DataLoader(TensorDataset(images, torch.from_numpy(labels[first])), batch_size=250)

unet = load_openai_unet(str(CKPT), image_size=32, device="cuda")
torch.manual_seed(0)
H, H_labels = collect_activations(unet, loader, block_name="input_blocks_9",
                                  alpha_prod_t=1 / (1 + SIGMA**2), timestep=60, device="cuda")
print("features h:", tuple(H.shape))
del unet
torch.cuda.empty_cache()


# In[7]:


CONCEPTS = REPO / "repro" / "assets" / "cifar10_concepts"
CONCEPTS.mkdir(parents=True, exist_ok=True)

t0 = time.time()
config = RFMTrainConfig(bandwidth=100.0, reg=1e-3, n_iters=5, top_k=3, device="cuda")
result = train_binary_rfm(H, H_labels, target_value=CAR, config=config)
v_rfm = get_direction(result, topk=3).cpu()
torch.save({"direction": v_rfm, "eigenvectors": result.eigenvectors.cpu(), "eigenvalues": result.eigenvalues.cpu(),
            "sign": int(result.sign), "auc": float(result.best_auc), "block_name": "input_blocks_9",
            "timestep": 60, "sigma": SIGMA}, CONCEPTS / f"class_{CAR}.pt")
print(f"cars against the rest: AUC {result.best_auc:.3f} on held-out images ({time.time() - t0:.0f} s)")


# ## 3. The paper's smoke test
# 
# `cifar10.sh` is the paper's launcher. With `MODE=smoke` it draws 4 images of one class twice: with noise
# alignment only ($\lambda = 8$), and with NA-RFM. It measures accuracy with the paper's evaluator, a ConvNeXt-Tiny
# classifier fine-tuned on CIFAR-10. If this cell passes, your setup works.

# In[8]:


# Skipped on Windows: cifar10.sh needs bash. Section 4 checks the setup instead.
# run(f"MODE=smoke TARGETS={CAR} bash cifar10.sh")

# smoke = REPO / "results" / "camera_ready_repro" / "cifar10"
# fig, axes = plt.subplots(1, 2, figsize=(6, 1.8))
# for ax, (name, path) in zip(axes, [("noise alignment only", next((smoke / "na_only").glob("automobile_*.png"))),
#                                    ("NA-RFM", smoke / "na_rfm" / "automobile" / "grid.png")]):
#     ax.imshow(Image.open(path), interpolation="nearest")
#     ax.set_title(name, fontsize=9)
#     ax.axis("off")
# plt.show()


# ## 4. A first measurement: 64 cars, unguided and with NA-RFM
# 
# Both rows start from the same 64 noises (seeds 0 to 63) and differ only in the steering:
# - **unguided:** $\lambda = 0$ and $w_{\mathrm{RFM}} = 0$, so nothing is steered;
# - **NA-RFM:** the paper's setting, which is the script's default: $\lambda = 3$ while $\sigma \ge 3.33$, then the
#   edit $h \leftarrow h + w_{\mathrm{RFM}}\,\|h\|\,v_{\mathrm{RFM}}$ with $w_{\mathrm{RFM}} = 1$ and amplification
#   $s = 2$ while $\sigma \in [0.01, 11.69]$.
# 
# Two numbers for each row:
# - **accuracy:** the percentage of images the evaluator calls a car;
# - **FID:** against all 6,000 CIFAR-10 cars, train and test.
# 
# With only 64 images FID is far higher than the paper's, because FID is biased upward when there are few
# images. Compare rows only at the same number of images.

# In[9]:


FIRST = REPO / "results" / "first"
ROWS = {"unguided": "--gaussian-strength 0 --rfm-strength 0", "NA-RFM": ""}
scores = {}
for name, flags in ROWS.items():
    out = FIRST / name
    run(f"{sys.executable} repro/run_cifar10_na_rfm.py --classes {CAR} --num-samples 64 --batch-size 64 "
        f"--output-dir {out} {flags}")
    run(f"{sys.executable} repro/evaluate_saved_samples.py --dataset cifar10 --samples-root {out} --targets {CAR} "
        f"--output {out / 'quality.json'} --batch-size 64")
    scores[name] = json.loads((out / "quality.json").read_text())

fig, axes = plt.subplots(1, 2, figsize=(10, 5.5))
for ax, name in zip(axes, ROWS):
    q = scores[name]
    ax.imshow(Image.open(FIRST / name / "automobile" / "grid.png"), interpolation="nearest")
    ax.set_title(f"{name}: accuracy {q['average_accuracy']:.0f}%, FID {q['average_fid']:.0f} (64 images)", fontsize=10)
    ax.axis("off")
plt.show()


# Table 2 also reports sampling time, for a batch of 16 images with 100 DDIM steps on an A100. We time the
# batch size the scripts use, 64. `seconds_per_image` draws two batches of 64 cars after a warm-up batch and returns
# the seconds per image. Pass it the same flags as the row's script.

# In[10]:


from na_rfm.ddim_sampling import build_ddim_schedule
from repro import run_cifar10_na_only as na_only
from repro.run_cifar10_na_rfm import parse_args, sample_class

unet = load_openai_unet(str(CKPT), image_size=32, device="cuda")


def seconds_per_image(flags="", na_only_strength=None, batch=64, repeats=2):
    # flags: those of run_cifar10_na_rfm.py; or na_only_strength=lambda, for run_cifar10_na_only.py --mode diff
    if na_only_strength is None:
        args = parse_args(f"--classes {CAR} --num-samples {batch} --batch-size {batch} {flags}".split())
        schedule = build_ddim_schedule(1000, args.inference_steps, 0.0001, 0.02, "cuda")
        draw = lambda: sample_class(unet=unet, schedule=schedule, class_idx=CAR, args=args)
    else:
        schedule = na_only.build_ddim_schedule(1000, 100, 0.0001, 0.02, "cuda")
        guidance = na_only.WienerGuidance(str(FILTERS / f"class_{CAR}.pt"), str(FILTERS / "full_dataset.pt"),
                                          mode="diff", strength=na_only_strength, start_step=0, end_step=50,
                                          num_components=1000, device="cuda")
        draw = lambda: na_only.sample_with_wiener_guidance(unet, schedule, guidance, batch,
                                                           na_only.StackedRandomGenerator("cuda", list(range(batch))))
    times = []
    for _ in range(repeats + 1):   # the first batch warms up the GPU
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            draw()
        torch.cuda.synchronize()
        times.append(time.time() - t0)
    return min(times[1:]) / batch


print(f"NA-RFM: {seconds_per_image():.2f} s per image at batch size 64 on {torch.cuda.get_device_name(0)}")
print("(the paper's Table 2: 6.2 s for a batch of 16 on an A100)")


# ## 5. Your turn: the replication (due Friday, November 13)
# 
# **a. The offline parts for all ten classes.**
# - Build `class_k.pt` for every class: section 1's command with `--classes 0 2 3 4 5 6 7 8 9`.
# - Fit an RFM direction for every class: section 2's last cell with `target_value=k`, saved as `class_k.pt`.
#   Report each class's AUC.
# 
# **b. Five rows, 256 images per class**, with seeds 0 to 255 in every row and 100 DDIM steps. All settings are
# in the paper's Tables 16 and 17:
# 
# | Row | Script | Noise alignment | RFM steering |
# |---|---|---|---|
# | Unguided (not in Table 2) | `run_cifar10_na_rfm.py` | none, $\lambda = 0$ | none, $w_{\mathrm{RFM}} = 0$ |
# | Noise alignment only, $\lambda = 3$ | `run_cifar10_na_only.py --mode diff` | $\lambda = 3$ while $\sigma \ge 3.33$ (steps 0 to 50) | none |
# | Noise alignment only, $\lambda = 8$ | `run_cifar10_na_only.py --mode diff` | $\lambda = 8$ while $\sigma \ge 3.33$ (steps 0 to 50) | none |
# | RFM only (all steps) | `run_cifar10_na_rfm.py` | none, $\lambda = 0$ | $w_{\mathrm{RFM}} = 1$, $s = 2$, at every step |
# | NA-RFM | `run_cifar10_na_rfm.py` | $\lambda = 3$ while $\sigma \ge 3.33$ | $w_{\mathrm{RFM}} = 1$, $s = 2$, while $\sigma \in [0.01, 11.69]$ |
# 
# Each script's `--help` lists its flags.
# - In `run_cifar10_na_rfm.py`, `--gaussian-strength` is $\lambda$, `--rfm-strength` is $w_{\mathrm{RFM}}$,
#   `--cfg-scale` is $s$, and `--rfm-sigma-lo` and `--rfm-sigma-hi` set the RFM window. "Every step" means a window
#   wider than every $\sigma$ the sampler visits; the largest is about 157.
# - In `run_cifar10_na_only.py`, `--strength` is $\lambda$, `--num-seeds` is the number of images, and
#   `--start-step 0 --end-step 50` is its window. Add `--eval-mode hf_convnext` so that it uses the paper's evaluator.
# 
# **c. Measure each row.**
# - **Accuracy:** from the ConvNeXt evaluator, averaged over the ten classes.
# - **FID:** for each class, against that class's 6,000 CIFAR-10 images with `repro/evaluate_saved_samples.py`,
#   then averaged over the ten classes, as in the paper.
# - **Time:** seconds per image at batch size 64, with `seconds_per_image` above. Table 2 timed a batch of 16 on an
#   A100, so compare your settings with each other rather than your seconds with the paper's.
# 
# **d. Show.**
# - A table like Table 2, with your numbers next to the paper's.
# - For each row, the first 16 images of every class, like the paper's Figures 9 and 10.
# - A paragraph on what matches the paper, what does not, and why.
# 
# **Encouraged:**
# - NA-RFM at the paper's 2,048 images per class, to compare FID with Table 2 directly.
# - A noise-alignment-only sweep over $\lambda$, plotting accuracy against FID, like the paper's Figure 4.
# 
# **Good to know:**
# - At 256 images per class, expect FIDs well above Table 2's, which used 2,048 per class. Accuracy has no such
#   bias. At 256 per class the paper reports 97.9% for NA-RFM (Figure 7) and 94.3% for RFM only (Figure 8).
# - `evaluate_saved_samples.py` reads `<folder>/<class name>/samples.pt`. The NA-only script saves one file per
#   class, `<class name>_mode=diff_str=...pt`, with the images under `samples`. Copy it into that layout before you
#   measure FID.
# - On a Colab GPU, one row at 256 images per class takes 15 to 30 minutes, and its ten FIDs about 10 more, so all
#   five rows take about three hours. Save your offline parts and samples (for example to Google Drive) so that a
#   disconnected session costs you nothing.

# In[11]:


# 5a-1: build the Gaussian denoiser (PCA) for the other nine classes
for i in range(10):
    if i == CAR:
        continue

    file = FILTERS / f"class_{i}.pt"
    if not file.exists():
        run(f"{build} --classes {i} --num-images 5000")

print(sorted(p.name for p in FILTERS.glob("*.pt")))


# In[12]:


# 5a-2:
aucs = {}

for i in range(10):
    start = time.time()

    result = train_binary_rfm(H, H_labels, target_value=i, config=config)
    direction = get_direction(result, topk=3).cpu()

    data = {
        "direction": direction,
        "eigenvectors": result.eigenvectors.cpu(),
        "eigenvalues": result.eigenvalues.cpu(),
        "sign": int(result.sign),
        "auc": float(result.best_auc),
        "block_name": "input_blocks_9",
        "timestep": 60,
        "sigma": SIGMA
    }

    torch.save(data, CONCEPTS / f"class_{i}.pt")

    aucs[CLASSES[i]] = float(result.best_auc)
    elapsed = time.time() - start
    print(f"{CLASSES[i]}: AUC {result.best_auc:.3f}, {elapsed:.0f} seconds")


# In[13]:


results_dir = REPO / "results"
results_dir.mkdir(exist_ok=True)
file_path = results_dir / "rfm_auc.json"
file_path.write_text(json.dumps(aucs, indent=2))

print(aucs)


# In[14]:


run(f"{sys.executable} repro/run_cifar10_na_rfm.py --help")
run(f"{sys.executable} repro/run_cifar10_na_only.py --help")


# In[15]:


# Generate 256 images for each class
RESULTS = REPO / "results" / "replication"
N = 256

methods = {
    "unguided": "--gaussian-strength 0 --rfm-strength 0",
    "RFM-only": "--gaussian-strength 0 --rfm-sigma-lo 0 --rfm-sigma-hi 200",
    "NA-RFM": ""
}

for method in methods:
    output_dir = RESULTS / method
    flags = methods[method]

    for i in range(10):
        sample_file = output_dir / CLASSES[i] / "samples.pt"

        if not sample_file.exists():
            run(
                f"{sys.executable} repro/run_cifar10_na_rfm.py "
                f"--classes {i} --num-samples {N} "
                f"--seed-start 0 --batch-size 64 --inference-steps 100 "
                f"--output-dir {output_dir} {flags}"
            )


# In[16]:


# 5b-2: noise alignment only, lambda = 3 and 8, steps 0 to 50
for strength in [3, 8]:
    output_dir = RESULTS / f"NA-only-lambda{strength}"

    run(
        f"{sys.executable} repro/run_cifar10_na_only.py "
        f"--all-classes --mode diff --strength {strength} "
        f"--num-seeds {N} --start-step 0 --end-step 50 "
        f"--inference-steps 100 --batch-size 64 "
        f"--eval-mode hf_convnext --output-dir {output_dir}"
    )


# In[17]:


# Check 
sample_file = RESULTS / "NA-RFM" / "automobile" / "samples.pt"
data = torch.load(sample_file, weights_only=False)

print("NA-RFM samples.pt:", type(data))

if isinstance(data, dict):
    print("Keys:", data.keys())
else:
    print("Shape:", data.shape)

na_only_dir = RESULTS / "NA-only-lambda3"
files = sorted(na_only_dir.rglob("*.pt"))

print("NA-only files:")
for file in files[:3]:
    print(file.name)


# In[18]:


# 5c-1: 
for strength in [3, 8]:
    method_dir = RESULTS / f"NA-only-lambda{strength}"

    for i in range(10):
        class_name = CLASSES[i]
        target_file = method_dir / class_name / "samples.pt"

        if not target_file.exists():
            source_file = next(
                method_dir.glob(f"{class_name}_mode=diff_str=*.pt")
            )

            data = torch.load(source_file, weights_only=False)
            samples = data["samples"]

            target_file.parent.mkdir(exist_ok=True)
            torch.save(samples, target_file)

# Check the image shape and value range
for method in ["NA-RFM", "NA-only-lambda3"]:
    sample_file = RESULTS / method / "automobile" / "samples.pt"
    samples = torch.load(sample_file, weights_only=False)

    print(method, samples.shape)
    print("Min:", samples.min().item())
    print("Max:", samples.max().item())


# In[21]:


# 5c-2: 
# Calculate accuracy and FID for each class
methods = [
    "unguided",
    "NA-only-lambda3",
    "NA-only-lambda8",
    "RFM-only",
    "NA-RFM"
]

quality = {}

# Get accuracy from the NA-only file
def na_only_accuracy(method_dir, class_name):
    source_file = next(
        method_dir.glob(f"{class_name}_mode=diff_str=*.pt")
    )
    data = torch.load(source_file, weights_only=False)

    accuracy_keys = []
    for key in data:
        if "acc" in key.lower():
            accuracy_keys.append(key)

    if len(accuracy_keys) == 0:
        raise KeyError(f"No accuracy found in {source_file.name}. Keys: {list(data)}")

    accuracy = float(data[accuracy_keys[0]])

    # Convert to a percentage if needed
    if accuracy <= 1:
        accuracy = accuracy * 100

    return accuracy


for method in methods:
    output_dir = RESULTS / method
    accuracies = []
    fids = []

    for i in range(10):
        class_name = CLASSES[i]
        quality_file = output_dir / f"quality_{class_name}.json"

        if not quality_file.exists():
            run(
                f"{sys.executable} repro/evaluate_saved_samples.py "
                f"--dataset cifar10 --samples-root {output_dir} "
                f"--targets {i} --output {quality_file} --batch-size 64"
            )

        results = json.loads(quality_file.read_text())
        fids.append(results["average_fid"])

        if "average_accuracy" in results:
            accuracy = results["average_accuracy"]
        else:
            accuracy = na_only_accuracy(output_dir, class_name)

        accuracies.append(accuracy)

    avg_accuracy = float(np.mean(accuracies))
    avg_fid = float(np.mean(fids))

    quality[method] = {
        "average_accuracy": avg_accuracy,
        "average_fid": avg_fid,
        "accuracy_per_class": accuracies,
        "fid_per_class": fids
    }

    print(f"{method}: accuracy {avg_accuracy:.1f}%, FID {avg_fid:.1f}")

# Save all results
quality_file = RESULTS / "quality_all.json"
quality_file.write_text(json.dumps(quality, indent=2))


# In[22]:


# 5c-3：
times = {}

times["unguided"] = seconds_per_image(
    "--gaussian-strength 0 --rfm-strength 0"
)

times["NA-only-lambda3"] = seconds_per_image(na_only_strength=3)
times["NA-only-lambda8"] = seconds_per_image(na_only_strength=8)

times["RFM-only"] = seconds_per_image(
    "--gaussian-strength 0 --rfm-sigma-lo 0 --rfm-sigma-hi 200"
)

times["NA-RFM"] = seconds_per_image()

for method in times:
    print(f"{method}: {times[method]:.3f} seconds per image")


# In[24]:


# 5d-1: 

import pandas as pd

# Results from the paper
paper = {
    "unguided": (None, None, None),
    "NA-only-lambda3": (62.0, 99, 5.8),
    "NA-only-lambda8": (80.0, 120, 5.8),
    "RFM-only": (94.8, 40.3, 6.8),
    "NA-RFM": (96.6, 41.4, 6.2)
}

# Compare our results with the paper
rows = []

for name in methods:
    row = {
        "setting": name,
        "accuracy (ours, %)": round(quality[name]["average_accuracy"], 1),
        "accuracy (paper, %)": paper[name][0],
        "FID (ours, 256/class)": round(quality[name]["average_fid"], 1),
        "FID (paper, 2048/class)": paper[name][1],
        "s/image (ours, 4070 Ti SUPER, batch 64)": round(times[name], 3),
        "s/batch of 16 (paper, A100)": paper[name][2]
    }
    rows.append(row)

table = pd.DataFrame(rows)

# Save and display the table
table.to_csv(REPO / "results" / "table2_replication.csv", index=False)
table


# In[25]:


# 5d-2: 
FIGS = REPO / "results" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

for method in methods:
    fig, axes = plt.subplots(10, 1, figsize=(8, 10))
    for k in range(10):
        x = torch.load(RESULTS / method / CLASSES[k] / "samples.pt", weights_only=False)[:16]
        row = ((x.permute(0, 2, 3, 1).numpy() + 1) / 2).clip(0, 1)      # [-1, 1] -> [0, 1]
        axes[k].imshow(np.concatenate(list(row), axis=1))                 # 16 images side by side
        axes[k].set_ylabel(CLASSES[k], rotation=0, ha="right", va="center", fontsize=9)
        axes[k].set_xticks([]); axes[k].set_yticks([])
    fig.suptitle(f"{method}: first 16 images of every class (seeds 0-15)")
    fig.tight_layout()
    fig.savefig(FIGS / f"grid_{method}.png", dpi=150)
    plt.show()


# In[26]:


# 5d-3: 
two = [3, 8]    # cat and ship: the lowest and one of the highest RFM AUCs
fig, axes = plt.subplots(2, 9, figsize=(13, 3.3))
for row, k in enumerate(two):
    f = torch.load(FILTERS / f"class_{k}.pt", weights_only=False)
    axes[row, 0].imshow(as_image(f["mean_patch"].numpy()))
    axes[row, 0].set_title(f"mean, {CLASSES[k]}", fontsize=9)
    for j in range(8):
        axes[row, j + 1].imshow(as_direction(f["eigenvectors"][:, j].numpy()))
        axes[row, j + 1].set_title(f"direction {j + 1}", fontsize=9)
for ax in axes.flat:
    ax.axis("off")
fig.savefig(FIGS / "pca_cat_ship.png", dpi=150)
plt.show()


# In[ ]:




