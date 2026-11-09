
from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

MODEL_FILE = "superkart_model.joblib"
model = joblib.load(MODEL_FILE)

EXPECTED_COLUMNS = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category"
]


@app.get("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "model_file": MODEL_FILE
    }), 200


@app.post("/v1/predict")
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "error": "A JSON request body is required."
            }), 400

        missing_columns = [
            column
            for column in EXPECTED_COLUMNS
            if column not in data
        ]

        if missing_columns:
            return jsonify({
                "error": "Required input fields are missing.",
                "missing_columns": missing_columns
            }), 400

        input_data = pd.DataFrame([data])
        input_data = input_data[EXPECTED_COLUMNS]

        prediction = model.predict(input_data)[0]

        return jsonify({
            "predicted_sales": float(prediction)
        }), 200

    except Exception as error:
        return jsonify({
            "error": str(error)
        }), 500


@app.post("/v1/predictbatch")
def predict_batch():
    try:
        if "file" not in request.files:
            return jsonify({
                "error": "A CSV file is required."
            }), 400

        uploaded_file = request.files["file"]
        batch_data = pd.read_csv(uploaded_file)

        missing_columns = [
            column
            for column in EXPECTED_COLUMNS
            if column not in batch_data.columns
        ]

        if missing_columns:
            return jsonify({
                "error": "Required CSV columns are missing.",
                "missing_columns": missing_columns
            }), 400

        batch_features = batch_data[EXPECTED_COLUMNS].copy()
        predictions = model.predict(batch_features)

        results = batch_data.copy()
        results["Predicted_Sales"] = predictions

        return jsonify({
            "row_count": len(results),
            "predictions": results.to_dict(orient="records")
        }), 200

    except Exception as error:
        return jsonify({
            "error": str(error)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=7860,
        debug=False
    )
