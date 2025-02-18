from fastapi import Request


"""
Allow the given image to be transformed with the given parameters
"""
class TransformationForm:
    """
    TransformationForm controller
    """
    def __init__(self, request: Request) -> None:
        self.request: Request = request
        self.errors: list = []
        self.image_id: str = ""

        self.color: float = 0.0
        self.brightness: float = 0.0
        self.contrast: float = 0.0
        self.sharpness: float = 0.0


    """
    Save the parameters wanted for the transformation
    """
    async def load_data(self):
        form = await self.request.form()
        self.image_id = form.get("image_id")

        self.color = form.get("color")
        self.brightness = form.get("brightness")
        self.contrast = form.get("contrast")
        self.sharpness = form.get("sharpness")


    """
    Check the validity of each value of the TransformationForm object
    """
    def is_valid(self):
        if not self.image_id or not isinstance(self.image_id, str):
            self.errors.append("A valid image id is required")
        if not self.color or not isinstance(self.color, str):
            self.errors.append("A valid color value is required")
        if not self.brightness or not isinstance(self.brightness, str):
            self.errors.append("A valid brightness value is required")
        if not self.contrast or not isinstance(self.contrast, str):
            self.errors.append("A valid contrast value is required")
        if not self.sharpness or not isinstance(self.sharpness, str):
            self.errors.append("A valid sharpness value is required")
        if not self.errors:
            return True
        return False
