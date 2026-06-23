import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("ai4i2020.csv")

# Drop unnecessary columns
df.drop(columns=[
    'UDI',
    'Product ID',
    'TWF',
    'HDF',
    'PWF',
    'OSF',
    'RNF'
], inplace=True)

# Encode Type
Type_dict = {"L": 0, "M": 1, "H": 2}
df["Type"] = df["Type"].map(Type_dict)

# Features
df["Temp_Difference"] = df["Process temperature [K]"] - df["Air temperature [K]"]
df["Power_Index"] = df["Rotational speed [rpm]"] * df["Torque [Nm]"]
df["Torque_RPM_Ratio"] = df["Torque [Nm]"] / df["Rotational speed [rpm]"]

y = df["Machine failure"]

x = df[
    [
        'Type',
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]',
        'Temp_Difference',
        'Power_Index',
        'Torque_RPM_Ratio'
    ]
]

# Train/Test Split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)

# Save model
joblib.dump(model, "predictive_maintenance_model.pkl")

print("Model saved successfully!")