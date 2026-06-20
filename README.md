# House-Price-prediction-ml
# Machine Learning project using Python, Pandas, Matplotlib and Linear Regression to predict house prices.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.DataFrame({
    "Area": [500, 700, 900, 1100, 1300],
    "Price": [10, 15, 20, 25, 30]
})

X = df[["Area"]]
y = df["Price"]

model = LinearRegression()
model.fit(X, y)

predicted_price = model.predict([[1000]])

print("Predicted Price =", predicted_price[0], "Lakh")

plt.scatter(df["Area"], df["Price"])
plt.xlabel("Area (sq ft)")
plt.ylabel("Price (Lakh)")
plt.title("House Price Prediction")
plt.show()
