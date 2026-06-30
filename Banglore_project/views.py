from django.shortcuts import render
import joblib
import pandas as pd

model = joblib.load("ridge_model.pkl")
data = pd.read_csv("Cleaned_data.csv")

def home(request):

    locations = sorted(data["location"].unique())
    display_price = None

    if request.method == "POST":

        location = request.POST["location"]
        bhk = int(request.POST["bhk"])
        bath = int(request.POST["bath"])
        sqft = float(request.POST["sqft"])

        input_df = pd.DataFrame({
            "location": [location],
            "total_sqft": [sqft],
            "bath": [bath],
            "bhk": [bhk]
        })

        prediction = round(model.predict(input_df)[0], 2)

        if prediction <= 100:
            display_price = f"₹ {prediction/100:.2f} Crore"
        else:
            display_price = f"₹ {prediction:.2f} Lakh"

    return render(request, "index.html", {
        "display_price": display_price,
        "locations": locations,
    })