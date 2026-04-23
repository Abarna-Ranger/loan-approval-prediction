import streamlit as st
import numpy as np
import pickle

# =========================
# LOAD MODEL & SCALER
# =========================
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# =========================
# PAGE SETUP
# =========================
st.set_page_config(page_title="Loan Prediction", layout="wide")

st.title("💰 Loan Approval Prediction System")
st.write("Predict whether a loan will be approved or rejected")

# =========================
# SIDEBAR INPUTS
# =========================
st.sidebar.header("📊 Enter Business Details")

loan_amount = st.sidebar.number_input("Loan Amount", min_value=0.0)
loan_tenure = st.sidebar.number_input("Loan Tenure (months)", min_value=1)

business_size = st.sidebar.selectbox("Business Size", ["Small", "Medium", "Large"])
business_type = st.sidebar.selectbox("Business Type", ["Retail", "Service", "Manufacturing"])

credit_line = st.sidebar.number_input("Credit Line", min_value=0.0)

documentation = st.sidebar.selectbox("Documentation Level", ["Low", "Medium", "High"])
location = st.sidebar.selectbox("Location", ["Urban", "Rural"])

loss_amount = st.sidebar.number_input("Previous Loss Amount", min_value=0.0)

# =========================
# ENCODING (MUST MATCH TRAINING)
# =========================
size_map = {"Small": 0, "Medium": 1, "Large": 2}
type_map = {"Retail": 0, "Service": 1, "Manufacturing": 2}
doc_map = {"Low": 0, "Medium": 1, "High": 2}
loc_map = {"Urban": 1, "Rural": 0}

business_size = size_map[business_size]
business_type = type_map[business_type]
documentation = doc_map[documentation]
location = loc_map[location]

# =========================
# PREDICTION
# =========================
st.subheader("📌 Prediction")

if st.button("Predict Loan Status"):

    # ✅ CREATE INPUT (8 FEATURES)
    input_data = np.array([[
        loan_amount,
        loan_tenure,
        business_size,
        business_type,
        credit_line,
        documentation,
        location,
        loss_amount
    ]])

    # ✅ SCALE INPUT
    input_scaled = scaler.transform(input_data)

    # ✅ PREDICT
    result = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)[0][1]

    st.divider()

    # =========================
    # OUTPUT
    # =========================
    if result[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.write(f"### 📊 Approval Probability: {prob:.2%}")

# =========================
# FOOTER
# =========================
st.divider()
st.markdown(
    "<center>Built using Streamlit | MBA Analytics Project</center>",
    unsafe_allow_html=True
)