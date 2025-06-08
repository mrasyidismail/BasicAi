# Age, Gender, and Race Prediction App

A Streamlit web app for predicting age, gender, and race from images using a multi-output ResNet model.

## Features

- Upload an image or use your camera (desktop only) for prediction.
- Predicts age, gender, and race from a single face image.

## Setup

1. **Clone this repository:**
    ```bash
    git clone https://github.com/yourusername/age-gender-race-app.git
    cd basicAi
    ```

2. **Add your model file:**
    - Place your trained model at `model/new/multioutput_utkface.pth`.

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run locally:**
    ```bash
    streamlit run multi_app.py
    ```

## Deploy on Streamlit Community Cloud

1. Push your code and model to a public GitHub repository.
2. Go to [Streamlit Cloud](https://streamlit.io/cloud) and connect your repo.
3. Set the main file path to `multi_app.py`.
4. (Optional) If your model file is large, use [Git LFS](https://git-lfs.com/) or host it elsewhere and download at runtime.

## Notes

- **Camera input** may not work on all mobile browsers. Use the upload feature if camera is unavailable.
- For HTTPS or custom ports, see Streamlit documentation.

---

**Enjoy!**
