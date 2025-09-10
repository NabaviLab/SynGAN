import os, glob, cv2, torch
from torch.utils.data import Dataset

VIEW_TOKENS = ["LCC", "LMLO", "RCC", "RMLO"]
IMG_EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]

class PairedMammoDataset(Dataset):
    """
    Patient-centric structure with optional per-view masks:

    data/
      train/ (or val/test)
        PATIENT001/
          prior/   LCC.png LMLO.png RCC.png RMLO.png
          current/ LCC.png LMLO.png RCC.png RMLO.png
          masks/   LCC.png LMLO.png            (RCC/RMLO missing is OK)
        PATIENT002/ ...

    Only views present in BOTH prior/ and current/ are used.
    If masks/<VIEW>.png exists for that view, we load it; otherwise has_mask=0.
    """
    def __init__(self, root_dir: str, split: str = "train", img_size=(1024, 1024)):
        assert split in ["train", "val", "test"]
        self.root_dir = root_dir
        self.split = split
        self.img_size = tuple(img_size)
        self.split_dir = os.path.join(root_dir, split)
        if not os.path.isdir(self.split_dir):
            raise FileNotFoundError(f"Split folder not found: {self.split_dir}")
        self.items = self._index_patients(self.split_dir)
        if len(self.items) == 0:
            print(f"[WARN] No paired prior/current views found under {self.split_dir}")

    # ---------- indexing ----------
    def _index_patients(self, split_dir):
        items = []
        for pdir in sorted(d for d in glob.glob(os.path.join(split_dir, "*")) if os.path.isdir(d)):
            pid = os.path.basename(pdir)
            prior_dir   = os.path.join(pdir, "prior")
            current_dir = os.path.join(pdir, "current")
            mask_dir    = os.path.join(pdir, "masks")  # optional

            if not (os.path.isdir(prior_dir) and os.path.isdir(current_dir)):
                continue

            prior_map   = self._view_map(prior_dir)
            current_map = self._view_map(current_dir)
            mask_map    = self._view_map(mask_dir) if os.path.isdir(mask_dir) else {}

            for view in sorted(set(prior_map) & set(current_map)):
                items.append({
                    "patient_id": pid,
                    "view": view,
                    "prior": prior_map[view],
                    "current": current_map[view],
                    "mask": mask_map.get(view, None),  # may be None
                })
        return items

    def _view_map(self, folder):
        if not folder or not os.path.isdir(folder):
            return {}
        files = []
        for ext in IMG_EXTS:
            files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        files = sorted(files)

        mapping = {}
        # exact basename first (e.g., LCC.png)
        for f in files:
            base = os.path.splitext(os.path.basename(f))[0].lower()
            for v in VIEW_TOKENS:
                if base == v.lower():
                    mapping[v] = f
        # then “contains token” fallback (e.g., patient_LCC_left.png)
        for f in files:
            name = os.path.basename(f).lower()
            for v in VIEW_TOKENS:
                if v.lower() in name and v not in mapping:
                    mapping[v] = f
        return mapping

    # ---------- preprocessing ----------
    def _read_gray(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return img

    def _resize(self, img):  # (H, W)
        return cv2.resize(img, self.img_size, interpolation=cv2.INTER_AREA)

    def _clahe(self, img):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img)

    def _normalize01(self, img):
        img = img.astype("float32") / 255.0
        return img

    def _preprocess_image(self, path):
        img = self._read_gray(path)
        img = self._resize(img)
        # hooks for artifact removal / breast cropping can be added here if you want
        img = self._clahe(img)
        return self._normalize01(img)

    def _preprocess_mask(self, path):
        mask = self._read_gray(path)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
        return (mask > 127).astype("float32")

    # ---------- torch Dataset API ----------
    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        prior   = self._preprocess_image(it["prior"])
        current = self._preprocess_image(it["current"])

        if it["mask"] is not None and os.path.exists(it["mask"]):
            mask = self._preprocess_mask(it["mask"])
            has_mask = torch.tensor([1.0], dtype=torch.float32)
        else:
            mask = torch.zeros_like(current, dtype="float32")
            has_mask = torch.tensor([0.0], dtype=torch.float32)

        return {
            "X_prior":   torch.from_numpy(prior).unsqueeze(0),   # (1,H,W)
            "X_current": torch.from_numpy(current).unsqueeze(0), # (1,H,W)
            "M_gt":      torch.from_numpy(mask).unsqueeze(0),    # (1,H,W)
            "has_mask":  has_mask,
            "patient_id": it["patient_id"],
            "view": it["view"],
        }