import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.title("🩺 Diabetes Prediction App")

page = st.sidebar.radio("Go to", ["Prediction", "Dashboard"])

df = pd.read_csv('diabetes.csv')

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

if page == "Prediction":
    st.header("🔮 Diabetes Prediction")

    pregnancies = st.number_input('Pregnancies', 0, 20, 0)
    glucose = st.number_input('Glucose Level', 0, 300, 120)
    blood_pressure = st.number_input('Blood Pressure', 0, 200, 70)
    skin_thickness = st.number_input('Skin Thickness', 0, 100, 20)
    insulin = st.number_input('Insulin', 0, 900, 79)
    bmi = st.number_input('BMI', 0.0, 70.0, 25.0)
    dpf = st.number_input('Diabetes Pedigree Function', 0.0, 2.5, 0.5)
    age = st.number_input('Age', 1, 120, 25)

    if st.button('Predict'):
        data = np.array([[pregnancies, glucose, blood_pressure,
                          skin_thickness, insulin, bmi, dpf, age]])
        data_scaled = scaler.transform(data)
        prediction = model.predict(data_scaled)
        if prediction[0] == 1:
            st.error("⚠️ The person is likely to have Diabetes.")
        else:
            st.success("✅ The person is not likely to have Diabetes.")

elif page == "Dashboard":
    st.header("📊 Diabetes Dataset Dashboard")

    st.subheader("Quick Overview")
    st.dataframe(df.head())

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Patients", len(df))
    col2.metric("Diabetic Patients", df['Outcome'].sum())
    col3.metric("Non-Diabetic Patients", len(df) - df['Outcome'].sum())

    st.subheader("Charts")
    colA, colB = st.columns(2)

    with colA:
        fig1, ax1 = plt.subplots(figsize=(3, 3))
        df['Outcome'].value_counts().plot(kind='bar', ax=ax1, color=['green', 'red'])
        ax1.set_title('Diabetes Count')
        st.pyplot(fig1)

    with colB:
        fig2, ax2 = plt.subplots(figsize=(3, 3))
        sns.histplot(df['Age'], bins=10, kde=True, ax=ax2, color='blue')
        ax2.set_title('Age Distribution')
        st.pyplot(fig2)

    colC, colD = st.columns(2)

    with colC:
        fig3, ax3 = plt.subplots(figsize=(3, 3))
        sns.scatterplot(x='Glucose', y='BMI', hue='Outcome', data=df, ax=ax3)
        ax3.set_title('Glucose vs BMI')
        st.pyplot(fig3)

    with colD:
        fig4, ax4 = plt.subplots(figsize=(3, 3))
        sns.boxplot(x='Outcome', y='Glucose', data=df, ax=ax4)
        ax4.set_title('Glucose by Outcome')
        st.pyplot(fig4)
