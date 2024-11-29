import streamlit as st
from PIL import Image
from clf import predict
import pandas as pd


# Sidebar for additional information
st.sidebar.title("About Artify")
st.sidebar.info(
    """
    **Artify** predicts the artist of a painting from a selected set of famous painters (Claude Monet, 
    Georges Braque, Pablo Picasso, Paul Cezanne, Pierre-August Renoir, Salvador Dalí, Vincent Van Gogh). 
    Upload an image to explore the magic of art and AI.
    """
)
st.sidebar.markdown("🖌️ **Let's Discover Art Together!**")

# App Title
st.title("🎨 Artify: Identify the Artist")
st.markdown("Upload a painting and let the AI analyze its artist.")

# File Upload Section
file_up = st.file_uploader("Upload an image", type=["jpg", "png"])

if file_up is not None:
    # Display uploaded image and prediction side by side using columns
    col1, col2 = st.columns([2, 3])

    # Column 1: Uploaded Image
    with col1:
        st.image(Image.open(file_up), caption="Uploaded Image", use_container_width=True)

    # Column 2: Prediction Results
    with col2:
        st.markdown("## 🎯 **Prediction Results**")
        st.write("Analyzing the painting, please wait...")

        # Get the predicted classes and their probabilities
        class_probs = predict(file_up)

        # Check if class_probs is a dictionary and contains valid data
        if isinstance(class_probs, dict) and len(class_probs) > 0:
            # Display the top prediction with the highest probability
            top_class = max(class_probs, key=class_probs.get)
            st.success(f"**Top Prediction**: {top_class} with {class_probs[top_class]:.2f}% confidence.")

            # Convert probabilities to a DataFrame and keep them as floats for the bar chart
            prob_df = pd.DataFrame(list(class_probs.items()), columns=["Artist", "Probability"])
            prob_df["Probability"] = prob_df["Probability"].apply(lambda x: float(x))  # Convert to float for charting

            # Display probabilities as a bar chart
            st.markdown("### 📊 **Artist Probabilities**")
            st.bar_chart(prob_df.set_index("Artist")["Probability"])

            # Display probabilities as percentages for readability
            prob_df["Probability"] = prob_df["Probability"].apply(lambda x: f"{x:.2f}%")  # Format as percentages
            st.write(prob_df)
        else:
            st.error("Error: The prediction did not return a valid dictionary of class probabilities.")