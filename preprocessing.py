"""Preprocessing variants used by the revised CRNN experiments."""

from dataclasses import dataclass

from PIL import Image
from torchvision import transforms


IMAGE_SIZE = (32, 128)  # height, width
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def convert_rgb(image):
    return image.convert("RGB")


@dataclass(frozen=True)
class AspectRatioPad:
    """Resize an RGB image without distortion, then center it on a white canvas."""

    height: int = IMAGE_SIZE[0]
    width: int = IMAGE_SIZE[1]
    fill: tuple[int, int, int] = (255, 255, 255)

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        source_width, source_height = image.size
        if source_width <= 0 or source_height <= 0:
            raise ValueError("Image dimensions must be positive")

        scale = min(self.width / source_width, self.height / source_height)
        resized_width = max(1, min(self.width, round(source_width * scale)))
        resized_height = max(1, min(self.height, round(source_height * scale)))
        resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)

        canvas = Image.new("RGB", (self.width, self.height), self.fill)
        left = (self.width - resized_width) // 2
        top = (self.height - resized_height) // 2
        canvas.paste(resized, (left, top))
        return canvas


def build_preprocessing(mode: str):
    """Build deterministic preprocessing for one of the registered experiment modes."""

    if mode == "direct_resize":
        geometric_transform = transforms.Resize(IMAGE_SIZE)
    elif mode == "aspect_ratio_padding":
        geometric_transform = AspectRatioPad()
    else:
        raise ValueError(
            f"Unknown preprocessing mode: {mode!r}. "
            "Use 'direct_resize' or 'aspect_ratio_padding'."
        )

    return transforms.Compose(
        [
            transforms.Lambda(convert_rgb),
            geometric_transform,
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
