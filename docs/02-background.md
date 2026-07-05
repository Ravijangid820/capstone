# 02 · Background — the concepts this project rests on

This chapter is the conceptual primer: enough FL, personalization, and medical-
imaging background to understand *why* the design in the following chapters looks
the way it does. If you already know FL and BraTS, skim to §2.4.

## 2.1 Federated Learning (FL) in one page

**The setting.** `K` data owners ("clients" — here, hospitals) each hold a private
dataset they will not share. We want one model trained on the union of all data,
without the data ever leaving each site.

**FedAvg (the canonical algorithm).** Repeat for many *rounds*:

1. The **server** sends the current global model weights to each client.
2. Each **client** trains locally for a few epochs on its own data.
3. Each client sends its updated weights back (never the data).
4. The server **averages** the clients' weights (weighted by data size) into the
   new global model.

After enough rounds the global model approximates what you'd get by pooling the
data — *if* the clients' data is IID (identically distributed). That "if" is the
whole story.

**Why heterogeneity breaks it.** When clients' data distributions differ (**non-IID**),
their local updates pull in different directions. The average is a compromise that
can fit no one well — and systematically fails the clients furthest from the
"average" distribution. This is called **client drift**, and it is the practical
reason a naive global model disappoints in the real world.

## 2.2 The federated method ladder

| Method | Idea | What problem it targets |
|--------|------|-------------------------|
| **Local-only** | Each client trains only on its own data; no sharing. | The *floor*: shows what you get without collaboration. |
| **Centralized** | Pool all data, train one model. | The *ceiling*: the (privacy-violating) best case, for reference. |
| **FedAvg** | Average client weights each round. | Collaboration without sharing data — but assumes ~IID. |
| **FedProx** | FedAvg + a proximal penalty `(μ/2)‖w − w_global‖²` that keeps local training from drifting too far from the global model. | Stabilizes training under mild non-IID; still **one** global model. |
| **Personalized FL** | Give each client its **own** model that shares most of the global knowledge but keeps some parameters local. | Non-IID head-on: each client keeps what makes it different. |

FedProx is a baseline, not a personalization method — it still produces a single
global model. The jump to *personalization* is the jump to per-client models.

## 2.3 Personalization strategies (and why FedBN)

All three personalization strategies here are implemented the *same clean way*: the
server always runs **plain FedAvg**, and personalization is enforced **client-side**
by choosing **which parameters to keep local** when the global model arrives. A
kept-local parameter is never overwritten by the global average, so it specializes
to that client across rounds.

| Strategy | Keeps local | Rationale |
|----------|-------------|-----------|
| **FedBN** *(primary)* | **Normalization layers** (BatchNorm params + running mean/var). | A scanner/protocol difference is an intensity-distribution shift, which lives almost entirely in BatchNorm statistics. Keeping BN local = keeping each hospital's scanner calibration. Tiny parameter cost, exactly matched to the failure mode. |
| **Personal-head** | Norm layers **+ the final output conv** (the segmentation head). | Shared "body" (encoder/decoder features), private "head" — a FedRep-style split. Useful comparison; more parameters kept local than FedBN. |
| **Fine-tune** | (post-hoc) Take the converged FedAvg global model, then train a few more epochs on each hospital's own data. | The simplest, strongest-in-practice personalization baseline; not a during-training strategy but a *starting-from-global* one. |

**Why FedBN is the well-motivated primary.** BatchNorm normalizes activations using
per-feature running mean and variance estimated from the data it sees. If Hospital A
scans brighter than Hospital B, A's and B's BN statistics differ — that *is* the
scanner fingerprint. FedAvg averages these incompatible statistics into one set that
fits neither extreme; FedBN instead lets each hospital keep its own, while still
sharing the convolutional filters (the "what a tumor looks like" knowledge) that
*should* be common. It is the cheapest possible fix aimed precisely at the cause.

> A subtle but important consequence: FedBN only helps when BN statistics actually
> differ across clients. On IID data they don't, so FedBN collapses to FedAvg. This
> is why the project needs a genuine heterogeneity source (real FeTS or the synthetic
> scanner shift) — see [Overview §1.6](01-overview.md#17-why-we-manufacture-heterogeneity-the-honest-caveat).

## 2.4 BraTS and brain-tumor segmentation

**BraTS** (Brain Tumor Segmentation challenge) provides multi-modal MRI of glioma
patients with expert tumor annotations. Each subject has **four MRI sequences** —
FLAIR, T1, T1-contrast-enhanced (T1ce), T2 — and one **segmentation label** volume,
all 3D `.nii.gz` at 240×240×155.

**Labels → evaluation regions.** Raw labels are `1` (necrotic core), `2` (edema),
`4` (enhancing tumor). BraTS scores three **overlapping** regions built from them:

| Region | Abbrev. | Built from labels |
|--------|---------|-------------------|
| Whole Tumor | **WT** | 1 ∪ 2 ∪ 4 (everything) |
| Tumor Core | **TC** | 1 ∪ 4 |
| Enhancing Tumor | **ET** | 4 |

Because the regions **overlap**, segmentation is a **multi-label** problem (a voxel
can be in WT *and* TC *and* ET) — which is why the model uses a **sigmoid** output
with per-channel Dice, not a softmax. See [Methods §Model](05-methods.md#the-model).

**Dice score** — the standard segmentation metric — measures overlap between the
prediction and ground truth: `Dice = 2|P∩G| / (|P|+|G|)`, ranging 0 (no overlap) to
1 (perfect). We report Dice per region (WT/TC/ET) and their mean, per hospital.

## 2.5 Why 2D slices instead of 3D volumes

Full 3D BraTS models are the state of the art but need far more than 4 GB of VRAM.
This project trains on **2D axial slices** extracted on the fly from the 3D volumes.
This is the standard 4 GB-friendly compromise: it fits the hardware, keeps I/O
manageable (no 15 GB of pre-sliced data on disk), and is entirely sufficient to
demonstrate the *federated* research question — the FL/personalization story is
orthogonal to 2D-vs-3D. See [Design decisions](08-design-decisions.md).

## 2.6 The tech stack

| Layer | Tool | Role |
|-------|------|------|
| Imaging model + transforms | **MONAI** | Medical-imaging-specific U-Net and dictionary transforms (intensity normalization, BraTS label conversion, augmentation). |
| Federation | **NVIDIA FLARE 2.8** | Runs the federated job via its **Simulator** (all clients on one machine, sequentially). Server = plain FedAvg workflow. |
| DL framework | **PyTorch (CUDA 12.4)** | Everything underneath — autograd, AMP, the training loop. |
| Env / packaging | **`uv`** on **WSL2** | Deterministic Linux environment; FLARE needs POSIX. |

Why FLARE's **Simulator** and not a real multi-machine deployment: the research
question is about *algorithms* (does personalization recover outlier hospitals),
which the Simulator answers faithfully while running on one 4 GB GPU. It executes
each client in turn, so only one model is on the GPU at a time.
