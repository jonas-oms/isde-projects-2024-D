import os
from PIL import Image
import matplotlib.pyplot as plt
import io
import base64
import numpy as np

from app.config import Configuration

conf = Configuration()


def list_images():
    """Returns the list of available images."""
    img_names = filter(
        lambda x: x.endswith(".JPEG"), os.listdir(conf.image_folder_path)
    )
    return list(img_names)

def get_image_path(image_id: str) -> str:
    """Returns the path to the image with the given id."""
    return os.path.join(conf.image_folder_path, image_id)


def generate_histogram(image_path):
    """Generates a histogram for a given Image and returns the base64 encoded plot."""
    img = Image.open(image_path)
    img_array = np.array(img)

    plt.figure(figsize=(6, 4))

    if len(img_array.shape) == 2:  # For Grayscale Images
        plt.hist(img_array.ravel(), bins=256, color='gray', alpha=0.7, label="Grayscale")
    else:  # For RGB Images
        colors = ['red', 'green', 'blue']
        labels = ['Red', 'Green', 'Blue']
        for i, color in enumerate(colors):
            plt.hist(img_array[:, :, i].ravel(), bins=256, color=color, alpha=0.5, label=labels[i])

    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()

    # Save the plot to a buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)

    # returns the Buffer
    return base64.b64encode(buffer.getvalue()).decode()