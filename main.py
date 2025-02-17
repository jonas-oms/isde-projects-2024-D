import json
from fastapi import FastAPI, Request
import matplotlib.pyplot as plt
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.config import Configuration
from app.forms.classification_form import ClassificationForm
from app.ml.classification_utils import classify_image
from app.utils import list_images
import os


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

@app.get("/download_results")
async def download_results(image_id: str, scores: str):
    """Generates and returns a JSON file containing classification results."""
    try:
        classification_data = json.loads(scores)

        output_filename = f"classification_results_{image_id}.json"
        output_filepath = os.path.join("app/static", output_filename)

        with open(output_filepath, "w") as f:
            json.dump({"image_id": image_id, "classification_scores": classification_data}, f, indent=4)

        return FileResponse(output_filepath, media_type="application/json", filename=output_filename)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/download_plot")
async def download_plot(image_id: str, scores: str):
    """Generates and returns a bar chart (PNG) of classification results."""
    try:
        classification_data = json.loads(scores)

        labels = [item[0] for item in classification_data]
        values = [item[1] for item in classification_data]

        output_filename = f"classification_plot_{image_id}.png"
        output_filepath = os.path.join("app/static", output_filename)

        plt.figure(figsize=(8, 5))
        plt.barh(labels[::-1], values[::-1], color=["green", "red", "orange", "blue", "purple"])
        plt.xlabel("Confidence Score (%)")
        plt.title(f"Classification Results for {image_id}")
        plt.xlim(0, 100)  # Confidence scores are in percentage
        plt.tight_layout()

        plt.savefig(output_filepath)
        plt.close()

        return FileResponse(output_filepath, media_type="image/png", filename=output_filename)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
