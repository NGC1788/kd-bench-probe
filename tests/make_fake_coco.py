"""Tiny fake COCO-single tree in the benchmark's manifest format (smoke tests only)."""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

CLASSES = ["giraffe", "airplane", "clock", "zebra", "train", "bird", "elephant", "toilet", "stop sign", "bear"]
root = Path(sys.argv[1])
rng = np.random.default_rng(0)
images = []
for split, per_class in (("train", 3), ("val", 2), ("test", 2)):
    for label in range(len(CLASSES)):
        for k in range(per_class):
            name = f"{split}_{label}_{k}"
            img = root / "images" / split / f"{name}.jpg"
            mask = root / "masks" / split / f"{name}.png"
            img.parent.mkdir(parents=True, exist_ok=True)
            mask.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((rng.random((64, 80, 3)) * 255).astype("uint8")).save(img)
            m = np.zeros((64, 80), dtype="uint8")
            m[16:48, 20:60] = 255
            Image.fromarray(m).save(mask)
            images.append({"id": len(images), "label": label, "split": split, "probe": split == "val",
                           "image": str(img.relative_to(root)), "mask": str(mask.relative_to(root))})
(root / "manifest.json").write_text(json.dumps({"selection": "one_annotated_instance_per_image",
                                                "classes": CLASSES, "images": images}))
print(f"fake coco_single: {len(images)} images at {root}")
