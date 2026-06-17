import streamlit as st
import pandas as pd
import pickle

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""

<style>                       
/* Main Background */
.stApp{
    background: linear-gradient(135deg,#5a667d,#f7b7b7);
}

/* Hide Streamlit Menu */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}

/* Card */
.main-card{
    background:white;
    padding:40px;
    border-radius:25px;
    box-shadow:0px 10px 30px rgba(0,0,0,0.15);
    margin-top:40px;
}

/* Title */
.title{
    text-align:center;
    color:#0f0e0e;
    font-size:42px;
    font-weight:700;
    margin-bottom:30px;
}

/* Inputs */
.stNumberInput input,
.stSelectbox div[data-baseweb="select"]{
    border-radius:12px !important;
}

/* Button */
.stButton button{
    width:100%;
    background:#17823B;
    color:white;
    border:none;
    border-radius:12px;
    height:50px;
    font-size:18px;
    font-weight:bold;
}

.stButton button:hover{
    background:#12692f;
}

</style>
""", unsafe_allow_html=True)

# ---------------- MODEL ----------------
with open("churn_model.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------- CARD START ----------------
# st.markdown('<div class="main-card">', unsafe_allow_html=True)

st.markdown(
    '<div class="title">📊 Customer Churn Prediction Model</div>',
    unsafe_allow_html=True
)

# Inputs

tenure = st.number_input(
    "Customer Tenure (Months)",
    min_value=0,
    max_value=100,
    value=12
)

monthly = st.number_input(
    "Customer Monthly Charges",
    min_value=0.0,
    value=50.0
)

total = st.number_input(
    "Total Charges:",
    min_value=0.0,
    value=600.0
)

partner = st.selectbox(
    "Partner",
    ["No", "Yes"]
)

dependents = st.selectbox(
    "Dependents",
    ["No", "Yes"]
)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

payment = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Credit card (automatic)"
    ]
)

# Predict Button
if st.button("Predict Churn"):

    data = pd.DataFrame({
        "tenure":[tenure],
        "MonthlyCharges":[monthly],
        "TotalCharges":[total],
        "Partner_Yes":[1 if partner=="Yes" else 0],
        "Dependents_Yes":[1 if dependents=="Yes" else 0],
        "Contract_One year":[1 if contract=="One year" else 0],
        "Contract_Two year":[1 if contract=="Two year" else 0],
        "PaymentMethod_Credit card (automatic)":[
            1 if payment=="Credit card (automatic)" else 0
        ],
        "PaymentMethod_Electronic check":[
            1 if payment=="Electronic check" else 0
        ],
        "PaymentMethod_Mailed check":[
            1 if payment=="Mailed check" else 0
        ]
    })

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:

        st.markdown("""
        <div style="
            background-color:#f8c8c8;
            padding:15px;
            border-radius:8px;
            color:black;
            font-weight:bold;
            font-size:18px;">
            ⚠ Customer Likely To Churn
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="
            background-color:#b8d8b8;
            padding:15px;
            border-radius:8px;
            color:black;
            font-weight:bold;
            font-size:18px;">
            ✅ Customer Likely To Stay
        </div>
        """, unsafe_allow_html=True)


    # if prediction == 1:
    #     st.error("⚠ Customer Likely To Churn")
    # else:
    #     st.success("✅ Customer Likely To Stay")

    # st.write(f"Churn Probability: {probability:.2%}")

    st.markdown(
    f"""
    <p style="
        font-size:20px;
        font-weight:bold;
        color:black;">
        Churn Probability: {probability:.2%}
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("</div>", unsafe_allow_html=True)