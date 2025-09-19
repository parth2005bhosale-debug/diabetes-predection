import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Diabetes Dashboard", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("diabetes.csv")

df = load_data()

X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.title("📊 Diabetes Prediction App")

left, right = st.columns(2)

with left:
    st.subheader("📈 Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Patients", len(df))
    c2.metric("Diabetes Cases", int(df['Outcome'].sum()))
    c3.metric("No Diabetes Cases", int(len(df)-df['Outcome'].sum()))
    c4.metric("Model Accuracy", f"{acc*100:.2f}%")

    st.bar_chart(df['Outcome'].value_counts())
    st.line_chart(df[['Glucose','BloodPressure']][:50])

with right:
    st.subheader("🤖 Predict Diabetes")
    p1, p2 = st.columns(2)
    with p1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        glucose = st.number_input("Glucose", 0, 200, 120)
        blood_pressure = st.number_input("Blood Pressure", 0, 140, 70)
        skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
    with p2:
        insulin = st.number_input("Insulin", 0, 900, 80)
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
        age = st.number_input("Age", 1, 120, 30)

    if st.button("Predict"):
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        if prediction[0] == 1:
            st.error("🩺 The person is likely to have diabetes.")
        else:
            st.success("✅ The person is not likely to have diabetes.")
