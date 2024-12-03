import os

project_root = os.path.dirname(os.path.abspath(__file__))


class Configuration:
    """Contains the configuration information for the app."""

    # classification
    image_folder_path = os.path.join(project_root, "static/imagenet_subset")
    models = (
        "resnet18",
        "alexnet",
        "vgg16",
        "inception_v3",
    )
