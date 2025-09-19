import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Prediction & Dashboard", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Prediction"])

if page == "Dashboard":
    st.title("📊 Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Users", "1,250", "+5%")
    with col2:
        st.metric("Active Users", "980", "+3%")
    with col3:
        st.metric("Revenue", "$45K", "+8%")
    with col4:
        st.metric("Growth", "12%", "+2%")

    st.markdown("---")

    col5, col6 = st.columns(2)
    with col5:
        df1 = pd.DataFrame({"Month": ["Jan", "Feb", "Mar", "Apr", "May"],"Sales": [100, 150, 200, 250, 300]})
        fig1 = px.line(df1, x="Month", y="Sales", title="Monthly Sales")
        fig1.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig1, use_container_width=True)

    with col6:
        df2 = pd.DataFrame({"Category": ["A", "B", "C", "D"],"Count": [40, 60, 30, 80]})
        fig2 = px.bar(df2, x="Category", y="Count", title="Category Distribution")
        fig2.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig2, use_container_width=True)

    col7, col8 = st.columns(2)
    with col7:
        df3 = pd.DataFrame({"Region": ["East", "West", "North", "South"],"Profit": [10, 15, 7, 20]})
        fig3 = px.pie(df3, names="Region", values="Profit", title="Profit by Region")
        fig3.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig3, use_container_width=True)

    with col8:
        df4 = pd.DataFrame({"Day": ["Mon", "Tue", "Wed", "Thu", "Fri"],"Visitors": [120, 150, 170, 130, 190]})
        fig4 = px.area(df4, x="Day", y="Visitors", title="Daily Visitors")
        fig4.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig4, use_container_width=True)

elif page == "Prediction":
    st.title("🤖 Prediction")

    st.subheader("Enter Input Data")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 18, 100, 25)
        salary = st.number_input("Salary", 0, 200000, 50000)
    with col2:
        experience = st.number_input("Experience (Years)", 0, 40, 2)
        score = st.number_input("Score", 0, 100, 50)

    if st.button("Predict"):
        result = np.random.choice(["Approved", "Rejected"])
        st.success(f"Prediction Result: {result}")
