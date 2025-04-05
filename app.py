import tensorflow as tf
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# ✅ Load model and scaler
custom_objects = {"mse": tf.keras.losses.MeanSquaredError()}
model = tf.keras.models.load_model("model/model.h5", custom_objects=custom_objects)
scaler = joblib.load("model/scaler.pkl")

# ✅ Expected input keys
EXPECTED_KEYS = [
    "Produce Name", "Location", "Year", "Season", "Market Type",
    "Weather Condition", "Demand Level", "Supply Level"
]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # ✅ Check for missing fields
        for key in EXPECTED_KEYS:
            if key not in data:
                return jsonify({"error": f"Missing key: {key}"}), 400

        # ✅ Prepare model input — pass strings and floats directly
        model_input = {
            "Produce Name": tf.convert_to_tensor([data["Produce Name"]], dtype=tf.string),
            "Location": tf.convert_to_tensor([data["Location"]], dtype=tf.string),
            "Year": tf.convert_to_tensor([float(data["Year"])], dtype=tf.float32),
            "Season": tf.convert_to_tensor([data["Season"]], dtype=tf.string),
            "Market Type": tf.convert_to_tensor([data["Market Type"]], dtype=tf.string),
            "Weather Condition": tf.convert_to_tensor([data["Weather Condition"]], dtype=tf.string),
            "Demand Level": tf.convert_to_tensor([data["Demand Level"]], dtype=tf.string),
            "Supply Level": tf.convert_to_tensor([data["Supply Level"]], dtype=tf.string),
        }

        # ✅ Run prediction
        prediction_scaled = model(model_input, training=False)

        # ✅ Unscale result
        predicted_price = scaler.inverse_transform(prediction_scaled.numpy())[0][0]

        return jsonify({"predicted_price": float(predicted_price)})

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
