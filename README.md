# Code for ICASSP 2026 Paper, "Past as Prior: Reweighted Proxy Guidance for Stable Adversarial Training"

This repository implements Reweighted Proxy Guidance (RPG), a history-guided adversarial training framework that uses the immediately preceding model as a proxy prior. RPG can be combined with both single-step and multi-step adversarial training, including Fast-AT, Fast-AT-GA, Fast-BAT, and PGD-AT.

The implementation was developed from a Fast-BAT-style adversarial training codebase. In the original experimental code, RPG variants were named with `c*` modes (`cfast_at`, `cfast_at_ga`, `cfast_bat`, `cpgd`). Public command aliases are provided as `rpg_*`.

## Repository Structure

```text
.
├── train.py                 # baseline AT training: Fast-AT, Fast-AT-GA, Fast-BAT, PGD
├── train_reptile.py         # RPG training entry point
├── trainer.py               # baseline training logic
├── trainer_reptile.py       # RPG/RDU and self-distillation training logic
├── evaluation.py            # PGD / transfer evaluation
├── eval_aa.py               # AutoAttack evaluation
├── datasets.py              # CIFAR/SVHN/TinyImageNet/ImageNet data loaders
├── model_zoo.py             # ResNet, PreActResNet, WideResNet models
├── attack/                  # PGD and AutoAttack-compatible attack code
├── core/                    # minimal AutoAttack evaluation utilities
├── sh/                      # dataset preparation scripts
└── utils/                   # optimizers, schedulers, logging, math utilities
```

Large experimental artifacts are intentionally excluded from this release directory, including datasets, checkpoints, logs, CSV summaries, figures, archives, IDE metadata, and Python cache files.

## Requirements

The original experiments used:

```bash
python -m pip install -r requirements.txt
```

The pinned baseline is:

```text
torch==1.8.0
torchvision==0.9.0
numpy
scipy
matplotlib
tqdm
pandas
```

Newer PyTorch versions may work, but exact reproduction is safest with the versions above.

## Data

By default, scripts expect datasets under `./data/`. CIFAR-10, CIFAR-100, and SVHN can be downloaded by torchvision. For TinyImageNet, use:

```bash
cd sh
bash download_tiny_imagenet.sh
bash process_image_net.sh
```

For ImageNet preparation, see `ImageNet-Download.md`.

## Training

Baseline adversarial training:

```bash
python train.py --mode fast_at --dataset CIFAR10 --attack_eps 8
python train.py --mode fast_at_ga --dataset CIFAR10 --attack_eps 8 --ga_coef 0.2
python train.py --mode fast_bat --dataset CIFAR10 --attack_eps 8
python train.py --mode pgd --dataset CIFAR10 --attack_eps 8 --attack_step 10 --epochs 200 --lr_scheduler multistep --lr_max 0.1
```

RPG-augmented training:

```bash
python train_reptile.py --mode rpg_fast_at --dataset CIFAR10 --attack_eps 8 --lr_max 0.8 --lr_max_inner 0.2 --alpha 0.95 --temperature 6.0
python train_reptile.py --mode rpg_fast_at_ga --dataset CIFAR10 --attack_eps 8 --ga_coef 0.2 --lr_max 0.8 --lr_max_inner 0.2 --alpha 0.95 --temperature 6.0
python train_reptile.py --mode rpg_fast_bat --dataset CIFAR10 --attack_eps 8 --lr_max 0.8 --lr_max_inner 0.2 --alpha 0.95 --temperature 6.0
python train_reptile.py --mode rpg_pgd --dataset CIFAR10 --attack_eps 8 --attack_step 10 --epochs 200 --lr_scheduler multistep --lr_max 0.8 --lr_max_inner 0.1 --alpha 0.95 --temperature 6.0
```

Useful options:

```text
--dataset        CIFAR10 | CIFAR100 | SVHN | GTSRB | TINY_IMAGENET | IMAGENET
--model_type     PreActResNet | ResNet | WideResNet
--depth          18 | 34 | 50, plus WideResNet-specific depths
--attack_eps     perturbation budget in /255 units, e.g. 8 or 16
--attack_step    training attack steps
--lr_max         outer learning rate / RPG aggregation scale
--lr_max_inner   proxy fast-update learning rate
--alpha          self-distillation KL ratio; use 0.95 for the paper setting
--temperature    distillation temperature; use 6.0 for the paper setting
```

Checkpoints are written to `results_reptile/checkpoints/`, logs to `log_reptile/`, and CSV summaries to `results_reptile/accuracy/`.

## Evaluation

PGD evaluation:

```bash
python evaluation.py \
  --dataset CIFAR10 \
  --model_path results_reptile/checkpoints/MODEL.pth \
  --model_type PreActResNet \
  --depth 18 \
  --attack_method PGD \
  --attack_step 50 \
  --eps 8 16
```

AutoAttack evaluation:

```bash
python eval_aa.py \
  --dataset CIFAR10 \
  --model_path results_reptile/checkpoints/MODEL.pth \
  --model_type PreActResNet \
  --depth 18 \
  --eps 8
```

## Notes on Method Names

The paper name is **Reweighted Proxy Guidance (RPG)**. The code also keeps the old internal mode names for backward compatibility:

```text
rpg_fast_at     -> cfast_at
rpg_fast_at_ga  -> cfast_at_ga
rpg_fast_bat    -> cfast_bat
rpg_pgd         -> cpgd
```

The core RPG update is implemented in `trainer_reptile.py` through the proxy model update and parameter differential between the target model and proxy fast weights.

## Code Reading Guide

Start from `train_reptile.py`. It parses public `rpg_*` modes, maps them to the legacy internal mode names, creates the target model and proxy model, and passes both models to `trainer_reptile.py`.

The RPG logic is in `trainer_reptile.py`:

```text
cfast_at      RPG + Fast-AT
cfast_at_ga   RPG + Fast-AT-GA
cfast_bat     RPG + Fast-BAT
cpgd          RPG + PGD-AT
```

Each RPG branch follows the same structure: generate adversarial perturbations with the target model, update the proxy model once on the defense objective, then apply the RDU direction `theta - proxy_fast_weights` as the target-model gradient before synchronizing the proxy to the current target for the next minibatch.

## Citation

Please cite our ICASSP 2026 paper when using this code:

```bibtex
@inproceedings{liu2026past,
  title     = {Past as Prior: Reweighted Proxy Guidance for Stable Adversarial Training},
  author    = {Liu, Yaohua and Gao, Jiaxin},
  booktitle = {ICASSP},
  year      = {2026}
}
```
