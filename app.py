
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Hot Wheels Popularity Predictor",
    page_icon="🏎️",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("hotwheels_decision_tree.pkl")

model = load_model()

st.title("🏎️ Hot Wheels Popularity Predictor")
st.write("Decision Tree Machine Learning Project")
st.info(
    "This model uses simulated training data. "
    "Predictions are educational estimates, "
    "not actual Hot Wheels sales figures."
)

cars = {
    "Nissan Skyline GT-R": ("Nissan", "JDM"),
    "Toyota Supra": ("Toyota", "JDM"),
    "Porsche 911": ("Porsche", "Sports"),
    "Lamborghini Aventador": ("Lamborghini", "Supercar"),
    "Bugatti Chiron": ("Bugatti", "Hypercar"),
    "Ford Mustang": ("Ford", "Muscle"),
    "Chevrolet Camaro": ("Chevrolet", "Muscle"),
    "Dodge Charger": ("Dodge", "Muscle"),
    "McLaren P1": ("McLaren", "Hypercar"),
    "Honda Civic Type R": ("Honda", "JDM"),
    "Mercedes-Benz G-Class": ("Mercedes", "SUV"),
    "BMW M3": ("BMW", "Sports"),
    "Mazda RX-7": ("Mazda", "JDM"),
    "Audi R8": ("Audi", "Supercar"),
    "Twin Mill": ("Hot Wheels Original", "Fantasy")
}

st.subheader("Select Your Hot Wheels Car")

car = st.selectbox("Choose Car", list(cars.keys()))
year = st.slider("Select Year", 2010, 2026, 2025)

brand, category = cars[car]

st.write("**Brand:**", brand)
st.write("**Category:**", category)

if st.button("Predict Popularity", type="primary"):

    input_data = pd.DataFrame({
        "Year": [year],
        "Car": [car],
        "Brand": [brand],
        "Category": [category]
    })

    prediction = model.predict(input_data)[0]
    prediction = max(0, min(100, float(prediction)))

    st.success("Prediction Completed!")

    st.metric(
        label="Predicted Popularity Score",
        value=f"{prediction:.2f} / 100"
    )

    st.progress(int(round(prediction)))

    st.write(
        f"Estimated popularity of {car} "
        f"in {year}: {prediction:.2f}/100"
    )

st.divider()
st.caption("Hot Wheels ML Project | Decision Tree Regressor")
