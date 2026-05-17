"""Local sanity-check demo for a trained WordDetector checkpoint.

No web UI, no HuggingFace Space, no telemetry (ADR 007): just run a checkpoint
on one or more images and write the predicted word boxes to disk so you can
eyeball whether the trained model works.

    uv run python -m xournalpp_htr.training.word_detector.demo --help

With no ``--image`` the bundled example images are downloaded and used.
"""

import argparse
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np

from xournalpp_htr.training.shared.postprocessing import draw_bboxes_on_image
from xournalpp_htr.training.word_detector.infer import run_image_through_network
from xournalpp_htr.training.word_detector.utils import get_example_list


def annotate(image_path: Path, model_path: Path, device: str, output_path: Path) -> int:
    """Run the checkpoint on one image, save the annotated copy, return #words."""
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    result = run_image_through_network(image_gray, model_path=model_path, device=device)

    # Scale boxes from the fixed network-input size back to the input image.
    scaling = np.array(image_gray.shape) / np.array(result["model_input_image"].shape)
    boxes = [aabb.scale(*scaling[::-1]) for aabb in result["aabbs"]]
    vis = draw_bboxes_on_image(image_bgr, boxes, denormalise=False)

    cv2.imwrite(str(output_path), vis)
    print(f"{image_path} -> {output_path}  ({len(boxes)} words)")
    return len(boxes)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Image to run on. If omitted, the bundled examples are used.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("best_model.pth"),
        help="Path to the trained .pth checkpoint.",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("demo_output"),
        help="Directory the annotated images are written into.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.image is not None:
        images = [args.image]
    else:
        images = []
        for url in get_example_list():
            dst = args.output_dir / Path(url).name
            if not dst.exists():
                urlretrieve(url, dst)
            images.append(dst)
        if not images:
            parser.error("No --image given and no example images reachable online.")

    for image_path in images:
        out = args.output_dir / f"{Path(image_path).stem}_detected.jpg"
        annotate(image_path, args.model_path, args.device, out)


if __name__ == "__main__":
    main()
