from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor               #？CLIPProcessor是什么


MODEL_NAME = "openai/clip-vit-base-patch32"

TEXTS = [
    "a photo of a mouse",
    "a photo of a keyboard",
    "a photo of a cup",
    "a photo of a book",
    "a photo of a phone",
]


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_image_path():
    image_dir = Path(__file__).parent / "images"
    png_path = image_dir / "test.png"
    jpg_path = image_dir / "test.jpg"

    if png_path.exists():
        return png_path
    if jpg_path.exists():
        return jpg_path

    raise FileNotFoundError("Please put test.png or test.jpg in the images folder.")


def load_image(image_path):
    return Image.open(image_path).convert("RGB")


def load_clip(device):
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)               #？作用
    model = CLIPModel.from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.eval()
    return processor, model                                             #？为什么需要使用这两个量


def match_image_text(image, texts, processor, model, device):
    inputs = processor(
        text=texts,
        images=image,
        return_tensors="pt",                                            #？参数作用
        padding=True,                                                   #？参数作用
    )
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits_per_image
    probabilities = logits.softmax(dim=1)[0]                            #？这里的softmax为什么可以直接使用后面的参数意义
    best_index = probabilities.argmax().item()

    return best_index, probabilities


def main():
    device = get_device()
    image_path = get_image_path()
    image = load_image(image_path)
    processor, model = load_clip(device)

    best_index, probabilities = match_image_text(
        image=image,
        texts=TEXTS,
        processor=processor,
        model=model,
        device=device,
    )

    print(f"device: {device}")
    print(f"image path: {image_path}")
    print(f"prediction: {TEXTS[best_index]}")

    for text, probability in zip(TEXTS, probabilities):
        print(f"{text}: {probability.item():.4f}")


if __name__ == "__main__":
    main()
