"""
Transformation service
"""
from PIL import ImageEnhance
from app.ml.classification_utils import fetch_image


def transform_image(img_id, color, brightness, contrast, sharpness):
    """
    Take the given parameters to transform the given image

    :param img_id: the id of the image
    :param color: the color of the image
    :param brightness: the brightness of the image
    :param contrast: the contrast of the image
    :param sharpness: the sharpness of the image
    :return: the transformed image
    """
    img = fetch_image(img_id)

    img = ImageEnhance.Color(img).enhance(color)
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Sharpness(img).enhance(sharpness)

    return img
