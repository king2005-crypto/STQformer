"""Restore center frames; evaluate only if clean/reference frames are supplied."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .data import EvaluationDataset
from .losses import ssim_per_image, temporal_consistency
from .model import STQFormer


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = STQFormer(checkpoint["config"]["model"], initialize_pretrained=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.set_stage(checkpoint["stage"])
    return model.to(device).eval(), checkpoint["config"]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="root/video_id/frame.png")
    parser.add_argument("--output", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--clean-root", help="Matched clean frames for paired evaluation")
    mode.add_argument("--synthetic", action="store_true", help="Treat input frames as references; add noise")
    parser.add_argument("--sigma-s", type=float, default=.005)
    parser.add_argument("--sigma-c", type=float, default=.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    if device.type == "cpu":
        torch.set_num_threads(2)
    model, config = load_model(args.checkpoint, device)
    data = EvaluationDataset(args.input, config["model"]["image_size"], args.clean_root,
                             args.synthetic, args.sigma_s, args.sigma_c, args.seed)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False)
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output must be empty to avoid mixing evaluations")
    out.mkdir(parents=True, exist_ok=True)
    mse_sum, psnr_sum, ssim_sum, noisy_psnr_sum = 0., 0., 0., 0.
    count, tc_sum, tc_count = 0, 0., 0
    previous = None
    with (out / "per_frame.jsonl").open("w", encoding="utf-8") as log:
        for batch in loader:
            clip = batch["clip"].to(device)
            pred = model(clip).float().clamp(0, 1)
            if not torch.isfinite(pred).all():
                raise FloatingPointError("Non-finite prediction")
            if "target" in batch:
                target = batch["target"].to(device)
                mse = (pred - target).square().mean((1, 2, 3))
                psnr = -10 * mse.clamp_min(1e-12).log10()
                ssim = ssim_per_image(pred, target)
                noisy_mse = (clip[:, 2].clamp(0, 1) - target).square().mean((1, 2, 3))
                noisy_psnr = -10 * noisy_mse.clamp_min(1e-12).log10()
                mse_sum += mse.sum().item()
                psnr_sum += psnr.sum().item()
                ssim_sum += ssim.sum().item()
                noisy_psnr_sum += noisy_psnr.sum().item()
            for i, restored in enumerate(pred):
                video, name, index = batch["video"][i], batch["name"][i], int(batch["index"][i])
                folder = out / video
                folder.mkdir(exist_ok=True)
                # Keep original extension in the name to prevent stem collisions.
                destination = folder / (name + ".png")
                pixels = restored.mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
                Image.fromarray(pixels).save(destination)
                row = {"video": video, "frame": name, "index": index}
                if "target" in batch:
                    row.update(psnr=psnr[i].item(), ssim=ssim[i].item(), mse=mse[i].item())
                center = clip[i, 2]
                if previous is not None and previous[0] == video and previous[1] + 1 == index:
                    tc = temporal_consistency(restored, previous[2], center, previous[3]).item()
                    tc_sum += tc
                    tc_count += 1
                    row["tc_residual_difference"] = tc
                previous = (video, index, restored.clone(), center.clone())
                log.write(json.dumps(row) + "\n")
                count += 1
    metrics = {"count": count, "reference_mode": "synthetic" if args.synthetic else
               ("paired" if args.clean_root else "none"), "tc_pairs": tc_count,
               "tc_residual_difference": tc_sum / tc_count if tc_count else None,
               "image_size": config["model"]["image_size"], "seed": args.seed,
               "sigma_s": args.sigma_s if args.synthetic else None,
               "sigma_c": args.sigma_c if args.synthetic else None,
               "stage": model.stage}
    if args.synthetic or args.clean_root:
        metrics.update(psnr_mean_db=psnr_sum / count, ssim_mean=ssim_sum / count,
                       mse_mean=mse_sum / count,
                       psnr_from_global_mse_db=-10 * math.log10(max(mse_sum / count, 1e-12)),
                       noisy_psnr_mean_db=noisy_psnr_sum / count)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
