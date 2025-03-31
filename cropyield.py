import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import norm

# Load dataset
df = pd.read_csv("ICRISAT-District Level Data.csv")

# User input for crop selection
crop = input("Enter the crop name (e.g., RICE, WHEAT, MAIZE): ").upper()
yield_column = f"{crop} YIELD (Kg per ha)"

if yield_column not in df.columns:
    print("⚠ Crop data not found in dataset. Please check the crop name and try again.")
    exit()

# Select relevant columns
selected_columns = ["Year", "State Name", "Dist Name", yield_column]
df_selected = df[selected_columns].copy()

# Convert categorical variables to numerical
df_selected = pd.get_dummies(df_selected, columns=["State Name", "Dist Name"], drop_first=True)

# Handle missing values
df_selected.interpolate(method='linear', inplace=True)

# Aggregate data by year to reduce clutter
df_grouped = df_selected.groupby("Year")[yield_column].mean().reset_index()

# Split features and target
X = df_selected.drop(columns=[yield_column])
y = df_selected[yield_column]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Model Evaluation
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")

# Plot Crop Yield Over Time
plt.figure(figsize=(10, 5))
plt.plot(df_grouped['Year'], df_grouped[yield_column], marker='o', linestyle='-', label=f"{crop} Yield")
plt.xlabel("Year")
plt.ylabel(f"{crop} Yield (Kg per ha)")
plt.title(f"{crop} Yield Trends Over Time")
plt.legend()
plt.grid()
plt.show()

# Correlation Heatmap
print("Correlation matrix:")
print(df_selected.corr())  # Debugging step

plt.figure(figsize=(10, 5))
corr_matrix = df_selected.corr()
sns.heatmap(corr_matrix, cmap="coolwarm", annot=True, fmt=".2f")
plt.title(f"Correlation Between Factors & {crop} Yield")
plt.show()

# Probability of Low Yield
mean_yield = np.mean(y)
std_yield = np.std(y)

# Debugging: Ensure standard deviation is valid
print(f"Mean Yield: {mean_yield}, Std Dev: {std_yield}")
if std_yield == 0 or np.isnan(std_yield):
    print("⚠ Unable to compute probability due to zero or NaN standard deviation.")
else:
    threshold = mean_yield * 0.7
    prob_low_yield = norm.cdf(threshold, loc=mean_yield, scale=std_yield)
    print(f"Probability of Low {crop} Yield: {prob_low_yield:.2%}")

    # Conclusion
    if prob_low_yield > 0.5:
        print(f"⚠ High risk of low {crop} yield this season. Consider irrigation & fertilizers.")
    else:
        print(f"✅ {crop} yield is expected to be stable this season.")