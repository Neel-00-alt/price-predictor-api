import joblib
import numpy as np

# Load the scaler
scaler = joblib.load("model/scaler.pkl")

# Test inverse transformation with a sample value
test_value = np.array([[0.5]])  # A random scaled value
original_value = scaler.inverse_transform(test_value)[0][0]

print("✅ Scaler loaded successfully!")
print("🔹 Sample scaled value:", test_value[0][0])
print("🔹 Converted back to original scale:", original_value)
