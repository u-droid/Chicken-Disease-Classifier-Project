import numpy as np
import torch
import os
from PIL import Image
from torchvision import transforms

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # Load the model
        model = torch.load(os.path.join("artifacts", "training", "trained_model.pt"))
        model.eval()  # Set the model to evaluation mode

        # Load and preprocess the input image
        image = Image.open(self.filename)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        image = transform(image).unsqueeze(0)  # Add a batch dimension

        # Make a prediction
        with torch.no_grad():
            output = model(image)

        # Get the predicted class (assuming binary classification)
        _, predicted_class = torch.max(output, 1)

        # Map the predicted class to a label
        if predicted_class.item() == 1:
            prediction = 'Healthy'
        else:
            prediction = 'Coccidiosis'

        return [{"image": prediction}]

# Example usage:
if __name__ == "__main__":
    # Provide the path to your input image
    input_image_path = r"artifacts\data_ingestion\Chicken-fecal-images\Healthy\healthy.1.jpg"

    # Create a PredictionPipeline instance
    pipeline = PredictionPipeline(input_image_path)

    # Make a prediction
    result = pipeline.predict()

    print(result)
