"""Tests for the functions in the CUDA extension.

Usage:
```bash
pytest <THIS_PY_FILE> -s
```
"""

from typing import Optional
from pathlib import Path
from PIL import Image
import pytest
import torch

from examples.image_fitting import SimpleTrainer

from gsplat.rendering import _rasterization, rasterization

device = torch.device("cuda:0")

def image_path_to_tensor(image_path: Path):
    import torchvision.transforms as transforms

    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).permute(1, 2, 0)[..., :3]
    return img_tensor

def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def pytest_addoption(parser):
    parser.addoption(
        "--use_pytorch",
        action="store_true",
        default=False,
        help="Use PyTorch rasterization instead of default"
    )

def train_simpletrainer(num_iterations: int, rasterize_fnc):
    num_points=100000
    height, width = 250, 250
    img_path = None
    if img_path:
        gt_image = image_path_to_tensor(img_path)
    else:
        gt_image = torch.ones((height, width, 3)) * 1.0
        # make top left and bottom right red, blue
        gt_image[: height // 2, : width // 2, :] = torch.tensor([1.0, 0.0, 0.0])
        gt_image[height // 2 :, width // 2 :, :] = torch.tensor([0.0, 0.0, 1.0])

    trainer = SimpleTrainer(gt_image=gt_image, num_points=num_points)
    final_img = trainer.train(
        rasterize_fnc=rasterize_fnc,
        iterations=num_iterations,
        lr=1e-2,
        save_imgs=False,
        model_type="3dgs",
    )

    return final_img

@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_gsplat_training(request):
    # Get rasterization choice from CLI parameter
    use_torch_rasterization = request.config.getoption("--use_pytorch")
    print(f"use_torch_rasterization: {use_torch_rasterization}")
    if use_torch_rasterization:
        rasterize_fnc = _rasterization
    else:
        rasterize_fnc = rasterization

    num_runs = 3
    num_iterations = 1
    
    while True:
        renders = []
        means = []
        print(f"Running for num_iterations: {num_iterations}")
        for i in range(num_runs):
            seed_everything(42)
            final_img = train_simpletrainer(num_iterations=num_iterations, rasterize_fnc=rasterize_fnc)
            print(f"Final img mean: {final_img.mean()}")
            means.append(final_img.mean())
            renders.append(final_img)

        renders = torch.stack(renders)

        for i in range(len(renders)):
            for j in range(i + 1, len(renders)):
                diff = (renders[i] - renders[j]).abs().max()
                print(f"Max difference between renders {i} and {j}: {diff}")
                torch.testing.assert_close(renders[i], renders[j], rtol=1e-4, atol=1e-4)

        num_iterations += 1
        
        print(f"{'*' * 50}")
