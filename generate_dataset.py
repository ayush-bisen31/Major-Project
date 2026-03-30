import pandas as pd
import numpy as np

np.random.seed(42)

dates = pd.date_range(start="2021-01-01", end="2023-12-31", freq="W")
n = len(dates)

trend = np.linspace(500, 900, n)
seasonality = 150 * np.sin(2 * np.pi * np.arange(n) / 52)
noise = np.random.normal(0, 40, n)
sales = trend + seasonality + noise
sales = np.clip(sales, 100, None)

categories = np.random.choice(["Electronics", "Clothing", "Groceries", "Home & Garden", "Sports"], n)
regions = np.random.choice(["North", "South", "East", "West"], n)
promotion = np.random.choice([0, 1], n, p=[0.75, 0.25])

df = pd.DataFrame({
    "Date": dates,
    "Sales": np.round(sales, 2),
    "Units_Sold": np.round(sales / np.random.uniform(5, 15, n)).astype(int),
    "Category": categories,
    "Region": regions,
    "Promotion": promotion,
    "Revenue": np.round(sales * np.random.uniform(1.5, 2.5, n), 2),
})

df.to_csv("retail_sales_data.csv", index=False)
print(f"Dataset created: {len(df)} rows")
print(df.head())
