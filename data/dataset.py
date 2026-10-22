"""PyTorch Dataset for DeltaShot screenshot sequences."""
import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

class ScreenshotDataset(Dataset):
    def __init__(
        self,
        root_dir: Path,
        split: str = "train",
        patch_size: int = 256,
        no_reference: bool = False,
        max_pairs_per_bucket: int | None = None,
    ):
        """
        Args:
            root_dir: Path to the synthetic_pairs root.
            split: 'train', 'val', or 'test'.
            patch_size: size of random crops for training.
            no_reference: If True, only returns the target image. 
                          If False, returns (reference, target) tuple.
        """
        self.split_dir = Path(root_dir) / split
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")
        
        pair_dirs = sorted(d for d in self.split_dir.iterdir() if d.is_dir())
        if max_pairs_per_bucket is not None:
            selected = []
            bucket_counts = {}
            for pair_dir in pair_dirs:
                with (pair_dir / "meta.json").open(encoding="utf-8") as meta_file:
                    bucket = json.load(meta_file).get("degree_of_change", "unknown")
                if bucket_counts.get(bucket, 0) < max_pairs_per_bucket:
                    selected.append(pair_dir)
                    bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
            pair_dirs = selected
        self.pair_dirs = pair_dirs
        self.no_reference = no_reference
        self.split = split
        self.patch_size = patch_size

    def __len__(self):
        return len(self.pair_dirs)

    def __getitem__(self, idx):
        pair_dir = self.pair_dirs[idx]
        
        target_path = pair_dir / "target.png"
        target_img = Image.open(target_path).convert("RGB")
        
        if self.no_reference:
            if self.split == "train":
                i, j, h, w = transforms.RandomCrop.get_params(target_img, output_size=(self.patch_size, self.patch_size))
                target_img = TF.crop(target_img, i, j, h, w)
            else:
                target_img = TF.center_crop(target_img, output_size=(self.patch_size, self.patch_size))
            return TF.to_tensor(target_img)
            
        reference_path = pair_dir / "reference.png"
        ref_img = Image.open(reference_path).convert("RGB")
        
        if self.split == "train":
            i, j, h, w = transforms.RandomCrop.get_params(target_img, output_size=(self.patch_size, self.patch_size))
            target_img = TF.crop(target_img, i, j, h, w)
            ref_img = TF.crop(ref_img, i, j, h, w)
        else:
            target_img = TF.center_crop(target_img, output_size=(self.patch_size, self.patch_size))
            ref_img = TF.center_crop(ref_img, output_size=(self.patch_size, self.patch_size))
            
        target_tensor = TF.to_tensor(target_img)
        ref_tensor = TF.to_tensor(ref_img)
        
        return ref_tensor, target_tensor
