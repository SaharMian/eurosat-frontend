import streamlit as st
import requests
from PIL import Image
import io
import os

API_URL = os.getenv("API_URL", "https://Sahii007-eurosat-classifier.hf.space")

st.set_page_config(page_title="EuroSAT Classifier", page_icon="🛰️", layout="centered")
st.title("🛰️ EuroSAT Land Use Classifier")
st.caption("Upload a satellite image and the model will classify the land use type.")

CLASS_EMOJI = {
    "AnnualCrop": "🌾", "Forest": "🌲", "HerbaceousVegetation": "🌿",
    "Highway": "🛣️", "Industrial": "🏭", "Pasture": "🐄",
    "PermanentCrop": "🍇", "Residential": "🏘️", "River": "🌊", "SeaLake": "🌊"
}

uploaded = st.file_uploader("Choose a satellite image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded image", use_container_width=True)

    with st.spinner("Classifying..."):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()

            pred  = result["prediction"]
            conf  = result["confidence"]
            probs = result["all_probs"]

            emoji = CLASS_EMOJI.get(pred, "🛰️")
            st.success(f"**{emoji} {pred}** — confidence: {conf:.1%}")
            st.progress(conf)

            st.subheader("All class probabilities")
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            labels = [f"{CLASS_EMOJI.get(c,'')}{c}" for c, _ in sorted_probs]
            values = [v for _, v in sorted_probs]
            st.bar_chart(dict(zip(labels, values)))

        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to the API. Make sure the API container is running.")
        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("---")
st.markdown("**Classes:** AnnualCrop · Forest · HerbaceousVegetation · Highway · Industrial · Pasture · PermanentCrop · Residential · River · SeaLake")
