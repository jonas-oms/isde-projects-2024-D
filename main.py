import json
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse ,JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.config import Configuration
from app.forms.classification_form import ClassificationForm
from app.forms.transformation_form import TransformationForm
from app.ml.classification_utils import classify_image
from app.ml.transformation_utils import transform_image
from app.forms.histogram_form import HistogramForm
from app.utils import list_images,generate_histogram,get_image_path
from pathlib import Path

app = FastAPI()
config = Configuration()

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


@app.get("/info")
def info() -> dict[str, list[str]]:
    """Returns a dictionary with the list of models and
    the list of available image files."""
    list_of_images = list_images()
    list_of_models = Configuration.models
    data = {"models": list_of_models, "images": list_of_images}
    return data


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """The home page of the service."""
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/classifications")
def create_classify(request: Request):
    return templates.TemplateResponse(
        "classification_select.html",
        {"request": request, "images": list_images(), "models": Configuration.models},
    )


@app.post("/classifications")
async def request_classification(request: Request):
    form = ClassificationForm(request)
    await form.load_data()
    image_id = form.image_id
    model_id = form.model_id
    classification_scores = classify_image(model_id=model_id, img_id=image_id)
    return templates.TemplateResponse(
        "classification_output.html",
        {
            "request": request,
            "image_id": image_id,
            "classification_scores": json.dumps(classification_scores),
        },
    )


@app.get("/upload", response_class=HTMLResponse)
def upload_page(request: Request):
    """Shows the upload page with model selection."""
    return templates.TemplateResponse("upload.html", {"request": request, "models": Configuration.models})

@app.post("/upload")
async def upload_image(request: Request, file: UploadFile = File(...), model: str = Form(...)):
    """Saves the uploaded image and passes it to the selected classification model."""
    UPLOAD_FOLDER = "app/static/imagenet_subset"
    Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)

    # Speichern des Bildes
    file_location = Path(UPLOAD_FOLDER) / file.filename
    with open(file_location, "wb") as buffer:
        buffer.write(await file.read())

    # Prüfen, ob das ausgewählte Modell gültig ist
    if model not in Configuration.models:
        return JSONResponse(status_code=400, content={"error": "Invalid model selected"})

    # Klassifikation mit dem gewählten Modell starten
    classification_scores = classify_image(model_id=model, img_id=file.filename)

    return templates.TemplateResponse(
        "classification_output.html",
        {
            "request": request,
            "image_id": file.filename,
            "classification_scores": json.dumps(classification_scores),
        },
    )
  
@app.get("/transformation")
def create_transformation(request: Request):
    """
    Give the wanted page to the user

    :param request: Request the request asking for the page
    :return: TemplateResponse containing the page with the list of images
    """
    return templates.TemplateResponse(
        "transformation_select.html",
        {"request": request, "images": list_images()},
    )


@app.post("/transformation")
async def request_transformation(request: Request):
    """
    Create the transformed image with the given parameters of the request before sending it

    :param request: Request the request asking for the transformed page
    :return: TemplateResponse containing the result page with the image and its transformed version
    """
    form = TransformationForm(request)
    await form.load_data()

    image_id = form.image_id
    color = form.color
    brightness = form.brightness
    contrast = form.contrast
    sharpness = form.sharpness

    if not form.is_valid():
        return templates.TemplateResponse(
            "transformation_select.html",
            {"request": request, "images": list_images(), "errors": form.errors},
        )

    transform_image(image_id, color, brightness, contrast, sharpness, "transformed_image")

    return templates.TemplateResponse(
        "transformation_output.html",
        {
            "request": request,
            "image_id": image_id,
            "transformed_image": "transformed_image",
        },
    )

@app.get("/histogram", response_class=HTMLResponse)
def create_histogram(request: Request):
    """Displays the form for selecting an image."""
    return templates.TemplateResponse(
        "histogram.html",
        {"request": request, "images": list_images()}
    )

@app.post("/histogram")
async def request_histogram(request: Request):
    """Processes the form submission and returns the histogram image."""
    form = HistogramForm(request)
    await form.load_data()

    if not form.is_valid():
        return templates.TemplateResponse("histogram.html", {"request": request, "errors": form.errors})

    image_path = get_image_path(form.image_id)
    histogram = generate_histogram(image_path)

    return templates.TemplateResponse(
        "histogram_output.html",
        {"request": request, "image_id": form.image_id, "histogram_data": histogram}
    )

