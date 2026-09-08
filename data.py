"""One directory per temporally ordered video; never join different videos."""
import hashlib
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def natural_key(path):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", path.name)]


def discover(root):
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Missing frame root: {root}")
    videos = {}
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        frames = sorted((p for p in folder.iterdir() if p.suffix.lower() in EXTENSIONS), key=natural_key)
        if frames:
            videos[folder.name] = frames
    if not videos:
        raise ValueError(f"No video subfolders found in {root}; expected root/video_id/000001.png")
    return videos


def read_frame(path, size):
    with Image.open(path) as im:
        im = im.convert("RGB")
        if size:
            im = im.resize((size, size), Image.Resampling.BICUBIC)
        return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255


class NeighborDataset(Dataset):
    """Load T+1 REAL frames to form current and previous length-T windows."""
    def __init__(self, root, clip_len=5, image_size=224, training=True, seed=42,
                 mask_target=False):
        if clip_len != 5:
            raise ValueError("This paper example implements T=5 and neighbor offsets {-2,-1,1,2}")
        self.videos = discover(root)
        self.size, self.training, self.seed = image_size, training, seed
        self.mask_target = mask_target
        self.samples = [(video, start) for video, frames in self.videos.items()
                        for start in range(len(frames) - clip_len)]
        if not self.samples:
            raise ValueError("Each usable training video needs at least 6 consecutive frames")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        video, start = self.samples[index]
        six = torch.stack([read_frame(p, self.size) for p in self.videos[video][start:start + 6]])
        if self.training and torch.rand(()) < .5:
            six = six.flip(-1)  # All inputs/targets share the transform.
        gen = None if self.training else torch.Generator().manual_seed(self.seed + index)
        offset = (-2, -1, 1, 2)[torch.randint(4, (), generator=gen).item()]
        target_index = 3 + offset
        target = six[target_index].clone()
        center, prev_center = six[3].clone(), six[2].clone()
        clip, previous = six[1:].clone(), six[:5].clone()
        if self.mask_target:
            # Optional experiment, not specified in the paper. Remove the target
            # from BOTH forwards so it cannot leak through the TC branch.
            clip[target_index - 1] = center
            if target_index < 5:
                previous[target_index] = center
        return {"clip": clip, "previous": previous, "target": target,
                "center": center, "prev_center": prev_center, "offset": offset}


def frame_seed(seed, video, name):
    digest = hashlib.sha256(f"{seed}/{video}/{name}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2 ** 63 - 1)


def poisson_gaussian(clean, sigma_s, sigma_c, generator):
    """y=s*Poisson(x/s)+c*N(0,1); conditional variance s*x+c^2.

    Do NOT clip here; clipping breaks the stated zero-mean model.
    """
    if sigma_s <= 0 or sigma_c < 0:
        raise ValueError("sigma_s must be positive and sigma_c nonnegative")
    shot = torch.poisson(clean / sigma_s, generator=generator) * sigma_s
    return shot + sigma_c * torch.randn(clean.shape, generator=generator)


class EvaluationDataset(Dataset):
    def __init__(self, root, image_size, clean_root=None, synthetic=False,
                 sigma_s=.005, sigma_c=.005, seed=42):
        if synthetic and clean_root:
            raise ValueError("Use synthetic OR paired evaluation, not both")
        self.videos = discover(root)
        self.size, self.clean_root, self.synthetic = image_size, clean_root, synthetic
        self.sigma_s, self.sigma_c, self.seed = sigma_s, sigma_c, seed
        self.samples = [(v, i) for v, frames in self.videos.items() for i in range(2, len(frames) - 2)]
        if not self.samples:
            raise ValueError("Evaluation needs at least 5 consecutive frames per usable video")
        if clean_root:
            for v, i in self.samples:
                if not (Path(clean_root) / v / self.videos[v][i].name).is_file():
                    raise ValueError(f"Missing clean reference for {v}/{self.videos[v][i].name}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        video, center = self.samples[index]
        frames = self.videos[video]
        clip = []
        for p in frames[center - 2:center + 3]:
            frame = read_frame(p, self.size)
            if self.synthetic:
                gen = torch.Generator().manual_seed(frame_seed(self.seed, video, p.name))
                frame = poisson_gaussian(frame, self.sigma_s, self.sigma_c, gen)
            clip.append(frame)
        result = {"clip": torch.stack(clip), "video": video, "index": center,
                  "name": frames[center].name}
        if self.synthetic:
            result["target"] = read_frame(frames[center], self.size)
        elif self.clean_root:
            result["target"] = read_frame(Path(self.clean_root) / video / frames[center].name, self.size)
        return result
