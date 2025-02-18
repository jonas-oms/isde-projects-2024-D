"""
Transformation service
"""
from PIL import ImageEnhance
from app.ml.classification_utils import fetch_image


def save_transformed(img, name):
    """
    Save the given image with the given name

    :param img: the transformed image to save
    :param name: the name of the image
    """
    img.save("app/static/"+str(name)+".jpg")


def transform_image(img_id, color, brightness, contrast, sharpness, name):
    """
    Take the given parameters to transform the given image

    :param img_id: the id of the image
    :param color: the color of the image
    :param brightness: the brightness of the image
    :param contrast: the contrast of the image
    :param sharpness: the sharpness of the image
    :param name: the name of the image
    """
    img = fetch_image(img_id)

    img = ImageEnhance.Color(img).enhance(float(color))
    img = ImageEnhance.Brightness(img).enhance(float(brightness))
    img = ImageEnhance.Contrast(img).enhance(float(contrast))
    img = ImageEnhance.Sharpness(img).enhance(float(sharpness))

    save_transformed(img, name)
