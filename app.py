import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_Agri_data.csv")
    df = df.dropna(subset=["lc1a", "e13a_1"])
    df = df[(df["e7"] >= 18) & (df["e7"] <= 25)]
    df["loan_eligible"] = df["lc1a"].apply(lambda x: 1 if x in [2, 3, 4] else 0)
    features = ["e7", "e8", "ie1b", "lc1_1", "lc1_2", "gen3_1", "e13a_1", "e14_17", "e14_15", "e14_3"]
    df = df[features + ["loan_eligible"]]
    return df

@st.cache_resource
def train_model(df):
    X = df.drop("loan_eligible", axis=1)
    y = df["loan_eligible"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X.columns

def main():
    st.title("AgriTrust Loan Eligibility Predictor")

    df = load_data()
    model, feature_names = train_model(df)

    st.sidebar.header("Input Features")
    user_input = {feature: st.sidebar.number_input(feature, float(df[feature].min()), float(df[feature].max()), float(df[feature].mean())) for feature in feature_names}

    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][prediction]

    st.subheader("Prediction Result")
    st.write("Eligible for loan ✅" if prediction == 1 else "Not eligible for loan ❌")
    st.write(f"Confidence: {probability:.2%}")

if __name__ == "__main__":
    main()