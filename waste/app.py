from flask import Flask, request, render_template, redirect, url_for
from flask import send_from_directory
import torch
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Class names (update this based on your dataset)
class_names = sorted(['trash', 'plastic', 'paper', 'metal', 'glass', 'cardboard'])

# Load your trained model
model_path = '../my_model.pth'  # Update this to your model's path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet50(weights=None)  # Load without pretrained weights
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))  # Adjust this if necessary
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

def predict(img_name):
    img = Image.open(img_name).convert('RGB')  # Load image
    img = transform(img)  # Apply transformations
    img = img.unsqueeze(0).to(device)  # Add batch dimension and move to device

    with torch.no_grad():
        prediction = model(img)  # Make prediction

    output = torch.argmax(prediction, dim=1).item()
    return class_names[output]

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    # Save the uploaded image
    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)

    # Process the image and make a prediction
    predicted_class = predict(img_name=filepath)

    # Update image_url to be accessible via the new route
    image_url = f'/uploads/{file.filename}'

    return render_template('result.html', class_name=predicted_class.upper(), image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True)
