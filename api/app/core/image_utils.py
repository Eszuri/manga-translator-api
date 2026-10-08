from PIL import Image


def to_rgb_image(image: Image.Image) -> Image.Image:
    """Flatten transparency onto white, matching the browser's image capture."""
    if "A" in image.getbands() or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGB", image.size, "white")
        background.paste(rgba, (0, 0), rgba.getchannel("A"))
        return background
    return image if image.mode == "RGB" else image.convert("RGB")
