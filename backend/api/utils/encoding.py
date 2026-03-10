import base64
from io import BytesIO

from PIL import Image


def pil_image_to_data_url(image: Image.Image, format_name: str = "PNG") -> str:
    buffer = BytesIO()
    image.save(buffer, format=format_name)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    mime = "image/png" if format_name.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"
