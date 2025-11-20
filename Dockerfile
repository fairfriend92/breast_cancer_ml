FROM python:3.14-slim

# Set working directory
WORKDIR /breast_cancer_ml_app

# Copy python requirements in the container
COPY requirements.txt .

# Install libraries
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code in the container
COPY src/ ./src/

# Create folder for models 
RUN mkdir /breast_cancer_ml_app/models

# Create folder for outputs
RUN mkdir /breast_cancer_ml_app/outputs

# Default comand when container starts
CMD ["python", "src/train.py"]
