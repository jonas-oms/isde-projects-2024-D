# Web server for image classification

## Download the repository

Use git to clone the repository:

```bash
git clone https://github.com/unica-isde/isde-projects-2024-D
```

Optional but recommended - create conda environment: https://docs.conda.io/projects/miniconda/en/latest/
```bash
conda create --name isde python=3.10
conda activate isde
```

And install the requirements with 

```bash
pip install -r requirements.txt
```

## Configuration

Configure the service by editing the file `config.py`.

## Prepare the resources

It is recommended to pre-download images and models before running 
the server. This is to avoid unnecessary waits for users.

Run `prepare_images.py` and `prepare_models.py`. Models will 
be stored in your PyTorch cache directory, while the path for 
the image directory can be found in the `config.py` file. 

```bash
python app/prepare_images.py
python app/prepare_models.py
```

## Usage

### Run locally


To run the code without containers, it is sufficient to run 
separately the server,

```bash
uvicorn main:app --reload
```
