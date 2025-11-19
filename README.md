# Breast Cancer ML Project

This project trains a machine learning model to classify breast cancer data.

## Requirements

- Python 3.14
- pip

---

## Using a Python virtual environment (venv)

### Create and activate venv

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Install dependencies

```powershell
pip install -r requirements.txt
```

### Run the training script

```powershell
python src/train.py
```

### Deactivate the virtual environment

```powershell
deactivate
```

---

## Using Docker

### Build the Docker image

```powershell
docker build -t breast_cancer_ml .
```

### Run the container

```powershell
docker run --rm -v ${PWD}/models:/breast_cancer_ml_app/models breast_cancer_ml
```

> Notes:  
> - `${PWD}/models` → local folder for saving trained models  
> - `/breast_cancer_ml_app/models` → folder inside the container
