
import streamlit as st
import pandas as pd
import joblib

@st.cache_data
def load_data():
    return pd.read_csv("cleaned_Agri_data.csv")

@st.cache_resource
def load_model():
    return joblib.load("loan_model.pkl")

def main():
    st.title("AgriTrust Loan Eligibility Predictor")
    df = load_data()
    model = load_model()

    st.write("### Sample of the dataset")
    st.write(df.head())

    st.write("### Predict loan eligibility")
    age = st.slider("Age", 18, 25, 21)
    gender = st.selectbox("Gender", ["Male", "Female"])
    education = st.selectbox("Education Level", sorted(df["Edu_level"].dropna().unique()))
    bvn = st.selectbox("Has BVN?", df["BVN"].dropna().unique())

    if st.button("Predict"):
        # Simulated prediction input; adjust based on actual model requirements
        sample_input = pd.DataFrame([[age, 1 if gender == "Male" else 2]], columns=["Age", "Gender"])
        prediction = model.predict(sample_input)
        st.success(f"Loan Eligibility Prediction: {'Eligible' if prediction[0] == 1 else 'Not Eligible'}")

if __name__ == "__main__":
    main()
