import cv2
import numpy as np
from IPython.display import Image, display


def display_image(img: np.ndarray) -> None:
    """Display an OpenCV image inline in a Jupyter notebook."""
    ok, encoded = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("Could not encode image as PNG")
    display(Image(data=encoded.tobytes()))
