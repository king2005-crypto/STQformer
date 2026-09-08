"""Two-stage training. Supports CPU, one GPU, and Linux torchrun DDP."""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .data import NeighborDataset
from .losses import SpatialLoss, charbonnier, temporal_consistency
from .model import STQFormer


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(_):
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total, count = 0., 0
    for batch in loader:
        clip, target = batch["clip"].to(device), batch["target"].to(device)
        loss = charbonnier(model(clip), target)
        total += loss.item() * len(clip)
        count += len(clip)
    # No clean labels: this is ONLY a fixed neighbor-target validation proxy.
    return total / count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    tc = config["training"]
    if tc["spatial_epochs"] < 1 or tc["full_epochs"] < 1 or tc["batch_size"] < 1:
        raise ValueError("Both stage epoch counts and batch_size must be positive")
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        if not torch.cuda.is_available() or os.name == "nt":
            raise RuntimeError("DDP example requires Linux CUDA/NCCL; use single-process on Windows/CPU")
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
        dist.init_process_group("nccl")
    else:
        device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                              if args.device == "auto" else args.device)
    try:
        seed_everything(tc["seed"])
        if device.type == "cpu":
            torch.set_num_threads(tc.get("cpu_threads", 2))
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        if (out / "metrics.jsonl").exists():
            raise ValueError("Output already contains a run; use a new --output directory")
        kwargs = dict(clip_len=config["model"]["clip_len"], image_size=config["model"]["image_size"],
                      seed=tc["seed"], mask_target=tc.get("mask_target", False))
        train = NeighborDataset(args.train_root, training=True, **kwargs)
        val = NeighborDataset(args.val_root, training=False, **kwargs)
        overlap = set(train.videos) & set(val.videos)
        if overlap:
            raise ValueError(f"Video IDs overlap across splits: {sorted(overlap)}. Split by video/patient.")
        sampler = DistributedSampler(train, shuffle=True, seed=tc["seed"]) if world > 1 else None
        train_loader = DataLoader(train, batch_size=tc["batch_size"], sampler=sampler,
                                  shuffle=sampler is None, num_workers=args.workers,
                                  worker_init_fn=seed_worker,
                                  generator=torch.Generator().manual_seed(tc["seed"] + rank))
        val_loader = DataLoader(val, batch_size=tc["batch_size"], shuffle=False,
                                num_workers=args.workers)
        model = STQFormer(config["model"]).to(device)
        config["model"] = model.config  # Store architecture for network-free checkpoint loading.
        loss_fn = SpatialLoss(config["loss"]).to(device)
        amp = bool(tc.get("amp", False)) and device.type == "cuda"
        if rank == 0:
            (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
            metadata = {"torch": torch.__version__, "device": str(device), "world_size": world,
                        "per_rank_batch": tc["batch_size"], "global_batch": tc["batch_size"] * world,
                        "train_samples": len(train), "val_samples": len(val),
                        "parameters": sum(p.numel() for p in model.parameters()),
                        "train_video_ids": sorted(train.videos), "val_video_ids": sorted(val.videos)}
            (out / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            print(json.dumps(metadata), flush=True)
        for stage in ("spatial", "full"):
            model.set_stage(stage)  # BEFORE DDP construction, identical architecture across stages.
            wrapped = DistributedDataParallel(model, device_ids=[device.index]) if world > 1 else model
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                         lr=tc[f"lr_{stage}"], weight_decay=tc["weight_decay"])
            scaler = torch.amp.GradScaler("cuda", enabled=amp)
            best = float("inf")
            for epoch in range(1, tc[f"{stage}_epochs"] + 1):
                if sampler is not None:
                    sampler.set_epoch(epoch + (tc["spatial_epochs"] if stage == "full" else 0))
                wrapped.train()
                total, count = 0., 0
                for batch in train_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, enabled=amp):
                        if stage == "full" and config["loss"]["w_tc"] > 0:
                            pred, prev = wrapped(torch.cat([batch["clip"], batch["previous"]])).chunk(2)
                            loss = loss_fn(pred, batch["target"])
                            loss = loss + config["loss"]["w_tc"] * temporal_consistency(
                                pred, prev, batch["center"], batch["prev_center"])
                        else:
                            loss = loss_fn(wrapped(batch["clip"]), batch["target"])
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite training loss")
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
                    scaler.step(optimizer)
                    scaler.update()
                    n = len(batch["clip"])
                    total += float(loss.detach()) * n
                    count += n
                totals = torch.tensor([total, count], dtype=torch.float64, device=device)
                if world > 1:
                    dist.all_reduce(totals)
                if rank == 0:
                    val_proxy = validate(model, val_loader, device)
                    if not np.isfinite(val_proxy):
                        raise FloatingPointError("Non-finite validation loss")
                    row = {"stage": stage, "epoch": epoch, "train_loss": (totals[0] / totals[1]).item(),
                           "val_neighbor_charbonnier": val_proxy, "lr": tc[f"lr_{stage}"]}
                    print(json.dumps(row), flush=True)
                    with (out / "metrics.jsonl").open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row) + "\n")
                    if val_proxy < best:
                        best = val_proxy
                        torch.save({"model": model.state_dict(), "config": config, "stage": stage,
                                    "epoch": epoch, "val_neighbor_charbonnier": best},
                                   out / f"best_{stage}.pt")
                if world > 1:
                    dist.barrier()
            del wrapped
            # Stage 2 starts from the BEST spatial checkpoint; strict transfer.
            checkpoint = torch.load(out / f"best_{stage}.pt", map_location=device, weights_only=True)
            model.load_state_dict(checkpoint["model"], strict=True)
        if rank == 0:
            print(f"Finished. Checkpoint: {out / 'best_full.pt'}", flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
