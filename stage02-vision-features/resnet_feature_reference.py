from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


def get_device():
    """Choose GPU first when CUDA is available."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_image_path():
    """Use test.png first, then fall back to test.jpg."""
    image_dir = Path(__file__).parent / "images"
    png_path = image_dir / "test.png"
    jpg_path = image_dir / "test.jpg"

    if png_path.exists():
        return png_path
    if jpg_path.exists():
        return jpg_path

    raise FileNotFoundError("Please put test.png or test.jpg in the images folder.")


def build_transform():
    """Convert a local image into the input format expected by ResNet."""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),               #？这里对图片尺寸的改变不会丢失要判断物体形状吗
            transforms.ToTensor(),                    #？以下尺寸的选择和参数的选择
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],           
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_image(image_path, transform):
    """Load one image and add a batch dimension."""
    image = Image.open(image_path).convert("RGB")    #？为什么要转变为RGB
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)         #？增加一维的作用和意义
    return image_tensor


def build_feature_extractor(device):
    """Load pretrained ResNet18 and replace the final classifier with Identity."""
    weights = models.ResNet18_Weights.DEFAULT        #？这里的权重意义是什么
    model = models.resnet18(weights=weights)
    model.fc = nn.Identity()                         #？这行代码的作用
    model = model.to(device)
    model.eval()
    return model


def main():
    device = get_device()
    image_path = get_image_path()

    transform = build_transform()
    image_tensor = load_image(image_path, transform).to(device)
    model = build_feature_extractor(device)

    with torch.no_grad():
        feature = model(image_tensor)

    print(f"device: {device}")
    print(f"image path: {image_path}")
    print(f"image tensor shape: {image_tensor.shape}")
    print(f"feature shape: {feature.shape}")
    print(f"first 10 feature values: {feature[0, :10].cpu()}")


if __name__ == "__main__":
    main()
