import argparse
import copy
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

BATCH_SIZE = 256
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 5
PATIENCE = 5

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

RUN_SEED_OFFSETS = {"random_mask": 10_000, "stateless": 30_000}
STAGES = (0.05, 0.25, 0.50, 1.0)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(device):
    m = models.mobilenet_v2()
    m.classifier[1] = nn.Linear(1280, 2)
    m = m.to(device)
    for p in m.features.parameters():
        p.requires_grad = False
    return m


def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device).long()
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    return total_loss / max(1, total), correct / max(1, total)


def apply_gradient_mask(model, mask):
    trainable = 0
    total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        if name not in mask:
            param.requires_grad = False
            continue
        param.requires_grad = True
        m = mask[name].to(param.device).bool()
        trainable += int(m.sum().item())

        def hook_factory(mask_tensor):
            def hook(grad):
                return grad * mask_tensor
            return hook

        param.register_hook(hook_factory(m))
    return trainable, total


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    lr=1e-4,
    num_epochs=10,
    patience=5,
    weight_decay=0.0,
    run_tag="",
    exp_tag="",
    mask=None,
    masked_wd=False,
):
    criterion = nn.CrossEntropyLoss()
    opt_wd = 0.0 if (masked_wd and mask is not None) else weight_decay
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=opt_wd)
    best_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    patience_ctr = 0
    t0 = time.time()
    print(f"[{exp_tag} | {run_tag}] train start: epochs={num_epochs} bs={BATCH_SIZE} lr={lr} wd={weight_decay} opt_wd={opt_wd} device={device}")

    for epoch in range(num_epochs):
        ep_t0 = time.time()
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device).long()
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()

            if masked_wd and mask is not None and weight_decay != 0.0:
                with torch.no_grad():
                    for name, p in model.named_parameters():
                        if p.grad is None:
                            continue
                        if name not in mask:
                            continue
                        m = mask[name].to(p.device).bool()
                        if m.any():
                            p.grad.data[m] = p.grad.data[m] + weight_decay * p.data[m]

            optimizer.step()

            running_loss += loss.item() * x.size(0)
            _, preds = out.max(1)
            running_correct += preds.eq(y).sum().item()
            running_total += y.size(0)

            if (step + 1) % 50 == 0:
                tr_loss = running_loss / max(1, running_total)
                tr_acc = running_correct / max(1, running_total)
                print(f"[{exp_tag} | {run_tag}] epoch {epoch+1}/{num_epochs} step {step+1}/{len(train_loader)} train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}")

        tr_loss = running_loss / max(1, running_total)
        tr_acc = running_correct / max(1, running_total)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        ep_dt = time.time() - ep_t0

        print(
            f"[{exp_tag} | {run_tag}] epoch {epoch+1}/{num_epochs} done "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"dt={ep_dt:.1f}s best_val_loss={best_val_loss:.4f} patience={patience_ctr}/{patience}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_wts = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"[{exp_tag} | {run_tag}] early stop at epoch {epoch+1}/{num_epochs}")
                break

    model.load_state_dict(best_wts)
    dt = time.time() - t0
    print(f"[{exp_tag} | {run_tag}] train end: best_val_loss={best_val_loss:.4f} dt={dt:.1f}s")
    return model


def is_final_layer(name: str):
    return "classifier.1" in name


def selection_mask(parameters, fraction, *, seed=None, changes=None):
    head = {name for name in parameters if is_final_layer(name)}
    candidates = [name for name in parameters if name not in head]
    sizes = [parameters[name].numel() for name in candidates]
    head_count = sum(parameters[name].numel() for name in head)
    count = int((head_count + sum(sizes)) * fraction) - head_count
    if not 0 <= count <= sum(sizes):
        raise ValueError("Budget must cover the complete classifier and fit the model.")
    if changes is None:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(sum(sizes), generator=generator)[:count]
    else:
        scores = torch.cat([changes[name].abs().reshape(-1) for name in candidates])
        if not torch.isfinite(scores).all():
            raise ValueError("Nonfinite parameter changes.")
        indices = torch.argsort(scores, descending=True, stable=True)[:count]
    selected = torch.zeros(sum(sizes), dtype=torch.bool)
    selected[indices] = True
    mask = {name: torch.ones_like(parameters[name], dtype=torch.bool) for name in head}
    for name, segment in zip(candidates, selected.split(sizes)):
        mask[name] = segment.reshape(parameters[name].shape)
    return mask


def apply_update(edge, server, mask):
    updated = {name: value.clone() for name, value in edge.items()}
    for name, value in server.items():
        if name in mask:
            updated[name][mask[name]] = value[mask[name]]
        else:
            updated[name] = value.clone()
    return updated


def run(strategy, seed, fraction, folder, baseline, dataset, order, validation, test, stages):
    template = build_model(torch.device("cpu"))
    parameters = {name: baseline[name] for name, _ in template.named_parameters()}
    fixed_mask = selection_mask(parameters, fraction, seed=seed) if strategy == "random_mask" else None
    server = edge = baseline
    criterion = nn.CrossEntropyLoss()
    folder.mkdir(parents=True, exist_ok=True)
    rebuild = False
    for stage, availability in enumerate(STAGES[:stages], 1):
        checkpoint = folder / f"stage_{stage}.pth"
        if not rebuild and checkpoint.exists():
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            server, edge = saved["server"], saved["edge"]
            print(f"[{strategy}] seed={seed} resumed stage={stage}")
            continue

        rebuild = True
        for later_stage in range(stage + 1, len(STAGES) + 1):
            (folder / f"stage_{later_stage}.pth").unlink(missing_ok=True)
        if shutil.disk_usage(folder).free < 512 * 1024**2:
            raise RuntimeError("Less than 512 MiB free. Free disk space or resume on a larger volume.")
        stage_seed = seed + 1000 * stage
        seed_everything(stage_seed)
        model = build_model(device)
        model.load_state_dict(baseline if fixed_mask is not None else server)
        if fixed_mask is None:
            model.requires_grad_(True)
        else:
            apply_gradient_mask(model, fixed_mask)

        count = int(len(dataset) * availability)
        loader = DataLoader(
            Subset(dataset, order[:count]), batch_size=BATCH_SIZE,
            shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(stage_seed),
        )
        model = train_model(
            model, loader, validation, device,
            lr=LR, num_epochs=NUM_EPOCHS, patience=PATIENCE,
            weight_decay=WEIGHT_DECAY, run_tag=f"seed={seed}|step={stage}",
            exp_tag=strategy, mask=fixed_mask, masked_wd=fixed_mask is not None,
        )
        trained = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        mask = fixed_mask
        if mask is None:
            changes = {name: trained[name] - server[name] for name in parameters}
            mask = selection_mask(parameters, fraction, changes=changes)
        elif any(not torch.equal(trained[name][~mask[name]], value[~mask[name]])
                 for name, value in parameters.items()):
            raise RuntimeError("Random Mask training changed a parameter outside the mask.")

        edge = apply_update(edge, trained, mask)
        server = trained
        _, server_accuracy = evaluate(model, test, criterion)
        model.load_state_dict(edge)
        _, edge_accuracy = evaluate(model, test, criterion)
        result = {
            "strategy": strategy, "seed": seed, "budget": fraction, "stage": stage,
            "training_images": count, "selected_parameters": sum(int(m.sum()) for m in mask.values()),
            "server_test_accuracy": server_accuracy, "edge_test_accuracy": edge_accuracy,
        }
        temporary = checkpoint.with_suffix(".tmp")
        torch.save({"server": server, "edge": edge, "mask": mask, "metrics": result}, temporary)
        temporary.replace(checkpoint)
        print(json.dumps(result), flush=True)


def main():
    global device
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.home() / "IoT_lab/kek/insectNet/insectnet")
    parser.add_argument("--output", type=Path, default=Path("random_stateless_training"))
    parser.add_argument("--strategy", choices=(*RUN_SEED_OFFSETS, "both"), default="both")
    parser.add_argument("--budget", type=int, choices=(1, 5, 10), nargs="+", default=[1, 5, 10])
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--stages", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    if args.device:
        if args.device == "cuda" and not torch.cuda.is_available():
            parser.error("CUDA is unavailable in this Python session")
        if args.device == "mps" and not torch.backends.mps.is_available():
            parser.error("Metal is unavailable in this Python session")
        device = torch.device(args.device)

    baseline = torch.load(args.project / "models/initial_baseline_full.pth",
                          map_location="cpu", weights_only=True)
    build_model(torch.device("cpu")).load_state_dict(baseline)
    data = {split: ImageFolder(args.project / "data" / split, transform=transform)
            for split in ("train", "val", "test")}
    if any(subset.class_to_idx != data["train"].class_to_idx for subset in data.values()):
        parser.error("Train, validation, and test class mappings differ")
    order = torch.randperm(len(data["train"]), generator=torch.Generator().manual_seed(args.split_seed)).tolist()
    validation, test = (DataLoader(data[split], batch_size=64, num_workers=0) for split in ("val", "test"))
    strategies = RUN_SEED_OFFSETS if args.strategy == "both" else (args.strategy,)
    for strategy in strategies:
        for index in range(args.seeds):
            seed = RUN_SEED_OFFSETS[strategy] + index
            for budget in args.budget:
                folder = args.output / f"split_{args.split_seed}" / strategy / f"budget_{budget}" / f"seed_{seed}"
                run(strategy, seed, budget / 100, folder, baseline, data["train"], order,
                    validation, test, args.stages)


if __name__ == "__main__":
    main()
