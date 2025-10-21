from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    checkpoint_path = script_dir / "sam2_hiera_small.pt"
    image_path = script_dir / "German_shepherd.jpeg"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")

    config_name = "sam2_hiera_s"
    predictor = SAM2ImagePredictor(
        build_sam2(
            config_name,
            checkpoint_path.as_posix(),
        )
    )

    image = np.array(Image.open(image_path).convert("RGB"))
    predictor.set_image(image)

    input_points = np.array([[100, 90], [200, 200]])
    input_labels = np.array([1, 0])
    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=False,
    )

    mask = masks[0]
    overlay_segmentation(
        image=image,
        mask=mask,
        points=input_points,
        output_dir=script_dir / "outputs",
        output_name="German_shepherd_mask.png",
    )

    print("Segmentation mask saved to the outputs directory.")


def overlay_segmentation(
    image: np.ndarray,
    mask: np.ndarray,
    points: np.ndarray,
    output_dir: Path,
    output_name: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / output_name

    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.scatter(points[:, 0], points[:, 1], c="red", s=40)
    plt.imshow(mask, alpha=0.5)
    plt.axis("off")
    plt.savefig(result_path.as_posix(), bbox_inches="tight")
    plt.close()
    print(f"Saved visualization to {result_path}")


if __name__ == "__main__":
    main()
