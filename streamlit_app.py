import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Fashion MNIST Classifier",
    page_icon="👔",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .confidence-text {
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        font-size: 1.1rem;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model (cached)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

try:
    model = load_model()
except Exception as e:
    st.error(f"⚠️ Error loading model: {str(e)}")
    st.stop()

# Class names with emojis
class_info = {
    "T-shirt/top": "👕",
    "Trouser": "👖",
    "Pullover": "🧥",
    "Dress": "👗",
    "Coat": "🧥",
    "Sandal": "👡",
    "Shirt": "👔",
    "Sneaker": "👟",
    "Bag": "👜",
    "Ankle boot": "👢"
}
class_names = list(class_info.keys())

# Header
st.markdown('<h1 class="main-header">👔 Fashion MNIST Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload a fashion item image to classify it using deep learning</p>', unsafe_allow_html=True)

# Layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a 28x28 grayscale image (PNG/JPG)",
        type=["png", "jpg", "jpeg"],
        help="Upload a clear image of a fashion item"
    )
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("L").resize((28, 28))
        
        # Display with better styling
        st.markdown("#### 🖼️ Input Image")
        st.image(img, caption="Uploaded Image (28x28)", width=200)
        
        # Show image info
        st.info(f"📏 Image size: {img.size[0]}x{img.size[1]} pixels")

with col2:
    if uploaded_file is not None:
        st.markdown("### 🎯 Prediction Results")
        
        # Preprocess image
        img_array = np.array(img).reshape(1, -1).astype(float)
        
        # Standardize
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        img_array = scaler.fit_transform(img_array)
        
        # Make prediction
        with st.spinner("🔍 Analyzing image..."):
            predictions = model.predict(img_array, verbose=0)
            predicted_idx = np.argmax(predictions)
            predicted_class = class_names[predicted_idx]
            confidence = float(predictions[0][predicted_idx] * 100)
        
        # Display prediction
        emoji = class_info[predicted_class]
        st.markdown(
            f'<div class="prediction-box">{emoji} {predicted_class}<br>'
           ,
            unsafe_allow_html=True
        )
        
       
        
        # Show all predictions
        st.markdown("#### 📈 All Class Probabilities")
        
        # Create bar chart with Plotly
        fig = go.Figure(data=[
            go.Bar(
                x=[p * 100 for p in predictions[0]],
                y=[f"{class_info[name]} {name}" for name in class_names],
                orientation='h',
                marker=dict(
                    color=predictions[0] * 100,
                    colorscale='Viridis',
                    showscale=False
                ),
                text=[f"{p*100:.1f}%" for p in predictions[0]],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            xaxis_title="Confidence (%)",
            yaxis_title="Class",
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👆 Upload an image to see prediction results")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>💡 <b>Tip:</b> For best results, use clear images of fashion items on a plain background</p>
        <p>Built with Streamlit & TensorFlow</p>
    </div>
    """, unsafe_allow_html=True)