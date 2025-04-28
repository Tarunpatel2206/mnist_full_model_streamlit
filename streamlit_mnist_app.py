import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps

# --- Device setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Load the saved model ---
@st.cache_resource
def load_model():
    model = torch.load('mnist_full_model.pth', map_location=device)
    model.eval()
    return model

model = load_model()

# --- Streamlit UI ---
st.title("🖌️ MNIST Digit Recognizer (Full Model Version)")
st.write("Upload an image of a handwritten digit (0-9) and I'll predict it!")

uploaded_file = st.file_uploader("Choose a digit image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    img = ImageOps.invert(image)  # MNIST digits are white on black
    img = transform(img)
    img = img.unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output.data, 1)

    st.success(f"🎯 Predicted Digit: **{predicted.item()}**")
