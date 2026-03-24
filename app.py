import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="AI Image Classifier", layout="centered")

# -------------------------
# CUSTOM CSS (UI DESIGN)
# -------------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}
h1 {
    text-align: center;
}
.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# CNN MODEL (same as training)
# -------------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(4*4*128,256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# -------------------------
# LOAD MODEL (FIXED)
# -------------------------
model = CNN()
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()

# -------------------------
# CLASS LABELS
# -------------------------
classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# -------------------------
# TRANSFORM
# -------------------------
transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

# -------------------------
# UI
# -------------------------
st.title("🚀 AI Image Classifier")
st.write("Upload an image and let AI predict!")

# Sidebar
st.sidebar.title("📊 Model Info")
st.sidebar.write("""
- Dataset: CIFAR-10  
- Model: Custom CNN  
- Epochs: 15  
- Optimizer: Adam  
- Loss: CrossEntropy  
""")

# Upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])

# -------------------------
# PREDICTION
# -------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = transform(image).unsqueeze(0)

    with st.spinner("🤖 AI is thinking..."):
        with torch.no_grad():
            output = model(img)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            _, predicted = torch.max(output, 1)

    st.success(f"Prediction: {classes[predicted.item()]}")

    # -------------------------
    # CONFIDENCE BAR CHART
    # -------------------------
    st.subheader("📊 Confidence Scores")

    prob_dict = {classes[i]: float(probs[i])*100 for i in range(len(classes))}
    st.bar_chart(prob_dict)

    # -------------------------
    # TOP 3
    # -------------------------
    st.subheader("🔥 Top 3 Predictions")
    top3 = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:3]

    for label, score in top3:
        st.write(f"{label}: {score:.2f}%")