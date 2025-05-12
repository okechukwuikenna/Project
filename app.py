
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_resource
def load_model():
    return joblib.load("loan_model.pkl")

def preprocess_input(data):
    # Placeholder: Modify based on actual preprocessing
    return data

def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    sorted_idx = importance.argsort()[::-1][:10]
    plt.figure(figsize=(10, 5))
    sns.barplot(x=importance[sorted_idx], y=[feature_names[i] for i in sorted_idx])
    plt.title("Top 10 Important Features")
    st.pyplot(plt.gcf())

def main():
    st.title("AgriTrust Loan Eligibility Predictor")

    age = st.slider("Age", 18, 60)
    gender = st.radio("Gender", ["Male", "Female"])
    income = st.selectbox("Avg Income Category", ["Below N15,000 per month", "N15,001 - N35,000 per month", 
                                                  "N35001 - N55,000 per month", "N55,001 - N75,000 per month"])
    education = st.selectbox("Education Level", ["No education", "Primary complete", "Secondary complete", 
                                                  "University/Polytechnic OND", "University/Polytechnic HND"])

    input_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Avg_income": income,
        "Edu_level": education
    }])

    model = load_model()

    if st.button("Check Eligibility"):
        processed = preprocess_input(input_data)
        prediction = model.predict(processed)[0]
        st.success("Loan eligibility result: {}".format("Eligible" if prediction == 1 else "Not Eligible"))

        if st.checkbox("Show Feature Importance"):
            plot_feature_importance(model, processed.columns)

if __name__ == "__main__":
    main()
