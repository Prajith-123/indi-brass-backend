import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import joblib
import pandas as pd

# New imports for the casting inspection model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from PIL import Image
import io


app = Flask(__name__)
CORS(app)

# Load the pre-trained models
with open('pickles/timeline_model.pkl', 'rb') as f:
    timeline_model = pickle.load(f)

with open('pickles/price_model.pkl', 'rb') as f:
    price_model = pickle.load(f)
    
with open('pickles/labor_model.pkl', 'rb') as f: 
    labor_model = pickle.load(f)

# # Save to new joblib files
joblib.dump(timeline_model, 'pickles/timeline_model.joblib')
joblib.dump(price_model, 'pickles/price_model.joblib')
joblib.dump(labor_model, 'pickles/labor_model.joblib')


# New imports for the casting inspection model
# Load your trained casting inspection model
image_model = load_model("casting_inspection_model_saved.keras")

# Function to calculate carton utilization
def calculate_carton_utilization(length, width, height, total_cbm):
    # Convert CBM from cubic meters to cubic centimeters
    total_cbm_in_cm3 = total_cbm * 1_000_000  # 1 m³ = 1,000,000 cm³
    carton_volume = length * width * height
    utilization = (total_cbm_in_cm3 / carton_volume) * 100  # Calculate the percentage utilization
    return utilization

# Function to predict timeline based on packing quantities
def predict_timeline(ip_qty, mp_qty, item_length, item_width, item_height, quantity_pcs):
    # Prepare the input data for prediction (same as training data)
    input_data = np.array([[ip_qty, mp_qty, item_length, item_width, item_height, quantity_pcs]])
    
    # Use the trained model to predict completion days
    completion_days = timeline_model.predict(input_data)

    # Log the prediction value to debug
    print(f"Predicted Completion Days: {completion_days}")

    return completion_days[0]

# Function to predict price based on features
# Function to predict price based on features
def predict_price(material, item_type, quantity_pcs, price):
    # Prepare the input data for prediction (same as training data)
    data = {
        'Material': [material],
        'Item Type': [item_type],
        'Quantity (Pcs)': [quantity_pcs],
        'Price ($)': [price]
    }
    
    # Convert the data into a pandas DataFrame
    input_data = pd.DataFrame(data)
    
    # One-hot encode the 'Material' and 'Item Type' columns (same as training data)
    input_data_encoded = pd.get_dummies(input_data, columns=['Material', 'Item Type'], drop_first=True)

    # Make sure the input data matches the feature columns used in training
    # Align the columns with the model (adding missing columns or dropping extra ones)
    input_data_encoded = input_data_encoded.reindex(columns=price_model.feature_names_in_, fill_value=0)

    # Use the trained price model to predict the price
    predicted_price = price_model.predict(input_data_encoded)

    return predicted_price[0]

# Function to predict labor efficiency
def predict_labor_efficiency(ip_qty, mp_qty, quantity_pcs):
    # Prepare the feature array
    features = np.array([[ip_qty, mp_qty, quantity_pcs]])

    # Calculate the ratios and append them as new columns to the feature array
    ip_to_pcs_ratio = ip_qty / quantity_pcs
    mp_to_pcs_ratio = mp_qty / quantity_pcs

    # Stack the new features
    features = np.hstack((features, np.array([[ip_to_pcs_ratio, mp_to_pcs_ratio]])))

    # Predict labor efficiency
    labor_efficiency = labor_model.predict(features)

    # Convert the numpy result to a Python float and return it
    return float(labor_efficiency[0])  # Ensure this is a float

# Route to handle carton utilization prediction
@app.route("/predict-carton-utilization", methods=["POST"])
def predict_carton_utilization():
    try:
        data = request.get_json()
        print(f"Received data for carton utilization: {data}")  # Log the incoming data to debug

        # Convert received data to appropriate types (floats)
        length = data["length"]
        width = data["width"]
        height = data["height"]
        total_cbm = data["totalCBM"]

        # Input validation
        if not all(isinstance(i, (int, float)) for i in [length, width, height, total_cbm]):
            return jsonify({"error": "All inputs must be numbers."}), 400
        if length <= 0 or width <= 0 or height <= 0 or total_cbm <= 0:
            return jsonify({"error": "All inputs must be positive numbers."}), 400

        # Calculate utilization
        utilization = calculate_carton_utilization(length, width, height, total_cbm)

        return jsonify({"utilization": utilization})  # Return decimal value, multiplication done on frontend

    except Exception as e:
        print(f"Error: {str(e)}")  # Log the error message for debugging
        return jsonify({"error": f"Error: {str(e)}"}), 400

# Route to handle timeline prediction
@app.route("/predict-timeline", methods=["POST"])
def predict_timeline_route():
    try:
        data = request.get_json()
        print(f"Received data for timeline prediction: {data}")  # Log the incoming data to debug

        # Extract all features from the data
        ip_qty = data["ipQty"]  # IP quantity
        mp_qty = data["mpQty"]  # MP quantity
        item_length = data["itemLength"]  # Item length
        item_width = data["itemWidth"]  # Item width
        item_height = data["itemHeight"]  # Item height
        quantity_pcs = data["quantityPcs"]  # Quantity in PCS

        # Input validation
        if not all(isinstance(i, (int, float)) for i in [ip_qty, mp_qty, item_length, item_width, item_height, quantity_pcs]):
            return jsonify({"error": "All inputs must be numbers."}), 400
        if ip_qty <= 0 or mp_qty <= 0 or item_length <= 0 or item_width <= 0 or item_height <= 0 or quantity_pcs <= 0:
            return jsonify({"error": "All inputs must be positive numbers."}), 400

        # Predict timeline (completion days)
        completion_days = predict_timeline(ip_qty, mp_qty, item_length, item_width, item_height, quantity_pcs)

        # Log the predicted completion days
        print(f"Predicted Completion Days: {completion_days}")

        return jsonify({"completion_days": completion_days})  # Return the predicted completion days

    except Exception as e:
        print(f"Error: {str(e)}")  # Log the error message for debugging
        return jsonify({"error": f"Error: {str(e)}"}), 400

# Route to handle price prediction
@app.route("/predict-price", methods=["POST"])
def predict_price_route():
    try:
        data = request.get_json()
        print(f"Received data for price prediction: {data}")  # Log the incoming data for debugging

        # Extract all features for price prediction
        material = data["material"]
        item_type = data["itemType"]
        quantity_pcs = data["quantityPcs"]
        price = data["price"]

        # Input validation for categorical fields: material and itemType should be non-empty strings
        if not isinstance(material, str) or not material.strip():
            return jsonify({"error": "Material must be a non-empty string."}), 400
        if not isinstance(item_type, str) or not item_type.strip():
            return jsonify({"error": "Item Type must be a non-empty string."}), 400

        # Input validation for numeric fields: quantityPcs and price should be positive numbers
        if not isinstance(quantity_pcs, (int, float)) or quantity_pcs <= 0:
            return jsonify({"error": "Quantity (Pcs) must be a positive number."}), 400
        if not isinstance(price, (int, float)) or price <= 0:
            return jsonify({"error": "Price must be a positive number."}), 400

        # Predict price using the model
        predicted_price = predict_price(material, item_type, quantity_pcs, price)
        predicted_price_rounded = round(predicted_price, 2)

        return jsonify({"predicted_price": predicted_price_rounded})  # Return the predicted price

    except Exception as e:
        print(f"Error: {str(e)}")  # Log the error message for debugging
        return jsonify({"error": f"Error: {str(e)}"}), 400

# Route to handle labor efficiency prediction
@app.route("/predict-labor-efficiency", methods=["POST"])
def predict_labor_efficiency_route():
    try:
        data = request.get_json()

        # Extract input values
        ip_qty = data["ipQty"]
        mp_qty = data["mpQty"]
        quantity_pcs = data["quantityPcs"]

        # Input validation
        if not all(isinstance(i, (int, float)) for i in [ip_qty, mp_qty, quantity_pcs]):
            return jsonify({"error": "All inputs must be numbers."}), 400
        if ip_qty <= 0 or mp_qty <= 0 or quantity_pcs <= 0:
            return jsonify({"error": "All inputs must be positive numbers."}), 400

        # Calculate the ratios before passing the features to the model
        ip_to_pcs_ratio = ip_qty / quantity_pcs
        mp_to_pcs_ratio = mp_qty / quantity_pcs

        # Prepare the feature array with the calculated ratios
        features = np.array([[ip_qty, mp_qty, quantity_pcs, ip_to_pcs_ratio, mp_to_pcs_ratio]])

        # Predict labor efficiency using the trained model
        labor_efficiency = labor_model.predict(features)

        # Ensure it's a float before returning
        labor_efficiency_float = float(labor_efficiency[0]) * 100

        # Return the labor efficiency as a float in JSON format
        return jsonify({"labor_efficiency": labor_efficiency_float})

    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 400
    
# Route to handle casting inspection
@app.route("/predict-casting", methods=["POST"])
def predict_casting():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded."}), 400

        file = request.files['image']
        print("Image filename:", file.filename)

        from PIL import Image
        img = Image.open(file.stream).convert("L")  # Grayscale
        img = img.resize((300, 300))  # ✅ Match training image size

        img_array = np.array(img).astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=(0, -1))  # Shape: (1, 300, 300, 1)

        prediction = image_model.predict(img_array)
        prediction_score = float(prediction[0][0])  # Since class_mode="binary", model outputs one value

        if prediction_score < 0.4:
            result = "OK"
        elif prediction_score < 0.85:
            result = "Needs Improvement"
        else:
            result = "Defective"

        print(f"Prediction Score: {prediction_score:.4f}, Result: {result}")

        return jsonify({
            "result": result,
            "confidence": round(prediction_score * 100, 2)
        })

    except Exception as e:
        print("Error in prediction:", str(e))
        return jsonify({"error": str(e)}), 500

    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
    print(f"Server running on port {port}")