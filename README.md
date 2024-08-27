# Breast Cancer Prediction Using Neural Network

## Overview
This project implements a breast cancer prediction system using a neural network built with PyTorch. The system is encapsulated within a Flask web application, allowing users to input relevant medical features and receive predictions on whether the breast cancer is likely malignant or benign.

## Features
- **Neural Network Model:** The project utilizes a simple feedforward neural network with three fully connected layers designed for binary classification.
- **Data Preprocessing:** The input data is standardized using `StandardScaler` from scikit-learn to improve the model's performance.
- **Web Interface:** A user-friendly web interface built with Flask, allowing users to input their data and receive predictions directly.
- **Model Persistence:** The trained model is saved and loaded using PyTorch's `state_dict`, enabling easy reuse and deployment.

## Project Structure
```
├── app.py                 # Main Flask application
├── Breast_Cancer_Prediction_model.pth  # Saved neural network model
├── templates/
│   └── index.html         # HTML template for the web interface
├── static/
│   └── style.css          # CSS file for custom styling (optional)
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## Installation

### Prerequisites
- Python 3.x
- Pip (Python package installer)

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Pahinithi/Breast-Cancer-Prediction-Using-Neural-Network-Deep-Learning
   cd Breast-Cancer-Prediction
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv env
   source env/bin/activate   # On Windows use `env\Scripts\activate`
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask application:**
   ```bash
   python app.py
   ```

5. **Access the web application:**
   Open your web browser and go to `http://127.0.0.1:5000/` to interact with the model.

## Dataset
The project uses the Breast Cancer Wisconsin (Diagnostic) Dataset, which is included in the scikit-learn library. This dataset contains 30 feature columns and a binary target variable:
- **Features:** Mean, standard error, and worst values for various cell nuclei characteristics like radius, texture, perimeter, area, etc.
- **Target:** A binary value where `0` indicates malignant and `1` indicates benign.

## Model Architecture
The neural network model consists of:
- **Input Layer:** 30 neurons corresponding to the 30 features in the dataset.
- **Hidden Layer 1:** 64 neurons with ReLU activation function.
- **Hidden Layer 2:** 32 neurons with ReLU activation function.
- **Output Layer:** 1 neuron with Sigmoid activation function to produce a probability value indicating the likelihood of malignancy.

## Data Preprocessing
- **Standardization:** The feature values are standardized using `StandardScaler` to ensure that they have a mean of 0 and a standard deviation of 1, which helps in speeding up the convergence of the neural network.

## Usage
1. **Input the required features in the web form:**
   The web interface provides input fields for the 30 features used by the model. Enter these values based on medical examinations.
   
2. **Predict:**
   Click the "Predict" button to submit the form. The app will process the input and display whether the prediction is "Malignant" or "Benign".


## How It Works
- **Model Loading:** The pre-trained neural network model is loaded from the file `Breast_Cancer_Prediction_model.pth`.
- **Prediction:** When the form is submitted, the application converts the input values into a tensor, feeds it into the model, and generates a prediction.
- **Result Display:** The result is rendered on the same page, indicating whether the cancer is predicted to be malignant or benign.

## Docker Deployment
If you prefer deploying using Docker, a `Dockerfile` can be included:
```Dockerfile
# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV NAME BreastCancerPrediction

# Run app.py when the container launches
CMD ["python", "app.py"]
```

### Build and Run Docker Container
```bash
docker build -t breast-cancer-prediction .
docker run -p 5000:5000 breast-cancer-prediction
```

## License
This project is licensed under the MIT License.

## Kaggle Notebook : https://www.kaggle.com/code/pahirathannithilan/breast-cancer-prediction-using-neural-networks-dl?scriptVersionId=194223987

<img width="1728" alt="DL10" src="https://github.com/user-attachments/assets/5e8ee147-9023-4579-8b5b-569f36a4fe92">


## Author
Developed by [Nithilan](https://github.com/Pahinithi).

## Acknowledgments
- [scikit-learn](https://scikit-learn.org/)
- [PyTorch](https://pytorch.org/)
- [Flask](https://flask.palletsprojects.com/)
