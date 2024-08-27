from flask import Flask, render_template, request
import torch
import torch.nn as nn

app = Flask(__name__)

class BreastCancerNN(nn.Module):
    def __init__(self):
        super(BreastCancerNN, self).__init__()
        self.fc1 = nn.Linear(30, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

model = BreastCancerNN()
model.load_state_dict(torch.load('Breast_Cancer_Prediction_model.pth', map_location=torch.device('cpu')))
model.eval()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Define the feature names based on your model input
    feature_names = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 
                      'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 
                      'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 
                      'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se', 
                      'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 
                      'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 
                      'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst']
    
    input_data = []
    missing_fields = []
    for feature in feature_names:
        value = request.form.get(feature)
        if value:
            try:
                input_data.append(float(value))
            except ValueError:
                return f"Invalid input: Ensure all fields are numeric for {feature}."
        else:
            missing_fields.append(feature)

    if missing_fields:
        return f"Missing values for fields: {', '.join(missing_fields)}"

    # Convert input data to a tensor and make prediction
    input_tensor = torch.tensor([input_data], dtype=torch.float32)
    with torch.no_grad():
        prediction = model(input_tensor).item()

    return render_template('index.html', prediction=f'Prediction: {"Malignant" if prediction > 0.5 else "Benign"}')

if __name__ == '__main__':
    app.run(debug=True)
