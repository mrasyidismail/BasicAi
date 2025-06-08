import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Model definition
class MultiOutputResNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=None)
        self.features = nn.Sequential(*list(base.children())[:-1])
        in_features = base.fc.in_features

        self.age_head = nn.Linear(in_features, 1)
        self.gender_head = nn.Linear(in_features, 2)
        self.race_head = nn.Linear(in_features, 5)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        age = self.age_head(x)
        gender = self.gender_head(x)
        race = self.race_head(x)
        return age, gender, race

# Constants
gender_labels = ["Male", "Female"]
race_labels = ["White", "Black", "Asian", "Indian", "Other"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def preprocess_image(image: Image.Image):
    """Preprocess PIL image for model input."""
    return transform(image).unsqueeze(0)

def get_predictions(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        age, gender, race = model(image_tensor)
    age_pred = age.item()
    gender_probs = torch.softmax(gender, dim=1)
    gender_pred = torch.argmax(gender_probs, dim=1).item()
    gender_conf = gender_probs[0, gender_pred].item()
    race_probs = torch.softmax(race, dim=1)
    race_pred = torch.argmax(race_probs, dim=1).item()
    race_conf = race_probs[0, race_pred].item()
    return age_pred, gender_pred, gender_conf, race_pred, race_conf

# Load model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiOutputResNet().to(DEVICE)
model.load_state_dict(torch.load("./model/multioutput_utkface.pth", map_location=DEVICE))
model.eval()

st.title("Age, Gender, and Race Prediction")

tab1, tab2 = st.tabs(["Camera", "Upload Image"])

with tab1:
    enable = st.checkbox("Enable camera")
    picture = st.camera_input("Take a picture", disabled=not enable)
    if picture:
        st.image(picture, caption="Captured Image", use_column_width=True)
        image = Image.open(picture).convert("RGB")
        input_tensor = preprocess_image(image)
        age_pred, gender_pred, gender_conf, race_pred, race_conf = get_predictions(model, input_tensor, DEVICE)
        st.write(f"**Predicted Age:** {age_pred:.1f}")
        st.write(f"**Predicted Gender:** {gender_labels[gender_pred]} ({gender_conf:.0%})")
        st.write(f"**Predicted Race:** {race_labels[race_pred]} ({race_conf:.0%})")

with tab2:
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        input_tensor = preprocess_image(image)
        age_pred, gender_pred, gender_conf, race_pred, race_conf = get_predictions(model, input_tensor, DEVICE)
        st.write(f"**Predicted Age:** {age_pred:.1f}")
        st.write(f"**Predicted Gender:** {gender_labels[gender_pred]} ({gender_conf:.0%})")
        st.write(f"**Predicted Race:** {race_labels[race_pred]} ({race_conf:.0%})")