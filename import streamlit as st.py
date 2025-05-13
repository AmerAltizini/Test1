import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("random_forest_model.pkl")

st.set_page_config(page_title="Product Predictor", layout="centered")
st.title("🔮 Predict Product Weight and Worktimes")
st.write("Fill in the values and click predict.")

with st.form("prediction_form"):
    length = st.number_input("Item Length (mm)", value=6500)
    width = st.number_input("Item Width (mm)", value=46)
    pieces = st.number_input("Pieces", value=2000)
    cavities = st.number_input("Tool Cavities", value=16)
    machine_time = st.number_input("Machine Worktime per Piece (min)", value=1.5, format="%.4f")
    personnel_time = st.number_input("Personnel Worktime per Piece (min)", value=0.05, format="%.4f")
    thickness = st.number_input("Item Thickness (mm)", value=6.1)
    area = st.number_input("Area (mm²)", value=108.91)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([{
        "Item_Length": length,
        "Item_Width": width,
        "Pieces": pieces,
        "Tool_Cavities": cavities,
        "Machine_Worktime_Piece_Real": machine_time,
        "Personnel_Worktime_Piece_Real": personnel_time,
        "Item_Thickness": thickness,
        "Area": area
    }])
    
preds = model.predict(input_df)

# If predicting only one target (e.g. Weight)
if preds.ndim == 1:
    st.success(f"📦 Predicted Weight: {preds[0]:.2f} g")
# If using a multi-output model
else:
    st.success(f"📦 Predicted Weight: {preds[0][0]:.2f} g")
    st.info(f"⏱ Machine Worktime: {preds[0][1]:.3f} min")
    st.info(f"👷 Personnel Worktime: {preds[0][2]:.3f} min")