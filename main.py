import json
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse ,JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.config import Configuration
from app.forms.classification_form import ClassificationForm
from app.ml.classification_utils import classify_image
from app.utils import list_images

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
