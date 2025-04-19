from flask import Flask, request, render_template, redirect, url_for
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import base64
import io
from PIL import Image
import torch.nn.functional as F
import numpy as np
from hireachial import *
app = Flask(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available()  else 'cpu')

# Load your model
model = ResNet50()
model = model.to(device)
model.load_state_dict(torch.load('baselinedhc.pth', map_location=torch.device('cpu')))

model.eval()

# Albumentations transform
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0), std=(1)),
    ToTensorV2(),
])

# Class labels
class_labels = {
    0: 'Breast cancer',
    1: 'Lung cancer',
    2: 'Skin cancer',
    3: 'Brain cancer',
}

class_labels_sub = {
    0: 'no',
    1: 'no',
    2: 'yes',
}


def read_image(file):
    """Read and prepare the image for prediction."""
    image = Image.open(io.BytesIO(file.read()))
    image = np.array(image)
    # Convert the PIL Image to a numpy array and ensure RGB
    if image.shape[-1] == 4:  # Check if image is RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            image = read_image(file)
            
            transformed = transform(image=image)["image"].unsqueeze(0)
            transformed = transformed.to(device)
            # Make prediction
            with torch.no_grad():
                superclass_pred, subclass_pred = model(transformed)
                probabilities = F.softmax(superclass_pred, dim=1)
                indices = torch.max(probabilities, dim=1)
                prob = indices[0].cpu().detach().numpy()
                prob = prob[0]*100
                prob = round(prob, 2)  # Round to two decimal points
                superclass_pred = superclass_pred.argmax().item()  # Get predicted index for superclass
                subclass_pred = subclass_pred.argmax().item()  # Get predicted index for subclass

            # Translate indices to class labels
            superclass_label = class_labels[superclass_pred]
            subclass_label = class_labels_sub[subclass_pred]  # Assuming the same label dictionary applies

            # Encode the original image for HTML display
            img_io = io.BytesIO()
            pil_img = Image.fromarray(image)
            pil_img.save(img_io, 'JPEG', quality=70)
            img_io.seek(0)
            base64_img = base64.b64encode(img_io.getvalue()).decode('ascii')

            return render_template('index.html', image=base64_img, superclass=superclass_label, subclass=subclass_label,prob=prob)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

