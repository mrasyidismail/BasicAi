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
st.write("Anggota Kelompok:\n- Muhammad Rasyid Ismail (2702377480)\n- Ranindya Faradhani (2702397406)\n - Lydia Cong Andinata (2702376490)\n - Muhammad Zick Al-Farizi (2702384473)")
st.write("Link Presentasi: [Canva](https://www.canva.com/design/DAGo--ONQnc/3a9g-Ds6pxUeObgwVHpkeg/view?utm_content=DAGo--ONQnc&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=hc8ea6725f1)")
st.write("Link Dataset: [UTKFace Dataset](https://susanqq.github.io/UTKFace/)")
st.write("Link GitHub: [GitHub Repository](https://github.com/mrasyidismail/BasicAi)")

# Feature selection
st.sidebar.header("Select Prediction Targets")
predict_age = st.sidebar.checkbox("Predict Age", value=True)
predict_gender = st.sidebar.checkbox("Predict Gender", value=True)
predict_race = st.sidebar.checkbox("Predict Race", value=True)

def get_selected_predictions(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        age, gender, race = model(image_tensor)
    results = {}
    if predict_age:
        results["age"] = age.item()
    if predict_gender:
        gender_probs = torch.softmax(gender, dim=1)
        gender_pred = torch.argmax(gender_probs, dim=1).item()
        gender_conf = gender_probs[0, gender_pred].item()
        results["gender"] = (gender_pred, gender_conf)
    if predict_race:
        race_probs = torch.softmax(race, dim=1)
        race_pred = torch.argmax(race_probs, dim=1).item()
        race_conf = race_probs[0, race_pred].item()
        results["race"] = (race_pred, race_conf)
    return results

tab1, tab2 = st.tabs(["Camera", "Upload Image"])

with tab1:
    enable = st.checkbox("Enable camera")
    picture = st.camera_input("Take a picture", disabled=not enable)
    if picture:
        st.image(picture, caption="Captured Image", use_column_width=True)
        image = Image.open(picture).convert("RGB")
        input_tensor = preprocess_image(image)
        results = get_selected_predictions(model, input_tensor, DEVICE)
        if not (predict_age or predict_gender or predict_race):
            st.warning("Please select at least one prediction target from the sidebar.")
        else:
            if predict_age:
                st.write(f"**Predicted Age:** {results['age']:.1f}")
            if predict_gender:
                gender_pred, gender_conf = results["gender"]
                st.write(f"**Predicted Gender:** {gender_labels[gender_pred]} ({gender_conf:.0%})")
            if predict_race:
                race_pred, race_conf = results["race"]
                st.write(f"**Predicted Race:** {race_labels[race_pred]} ({race_conf:.0%})")

with tab2:
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        input_tensor = preprocess_image(image)
        results = get_selected_predictions(model, input_tensor, DEVICE)
        if not (predict_age or predict_gender or predict_race):
            st.warning("Please select at least one prediction target from the sidebar.")
        else:
            if predict_age:
                st.write(f"**Predicted Age:** {results['age']:.1f}")
            if predict_gender:
                gender_pred, gender_conf = results["gender"]
                st.write(f"**Predicted Gender:** {gender_labels[gender_pred]} ({gender_conf:.0%})")
            if predict_race:
                race_pred, race_conf = results["race"]
                st.write(f"**Predicted Race:** {race_labels[race_pred]} ({race_conf:.0%})")