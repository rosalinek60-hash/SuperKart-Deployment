
import streamlit as st
import pandas as pd
import requests

st.set_page_config(
    page_title="SuperKart Sales Forecasting",
    layout="wide"
)

st.title("SuperKart Sales Forecasting")
st.write(
    "Predict product-store sales using the deployed "
    "SuperKart machine learning model."
)

# Backend URL when containers share a Docker network
API_URL = "http://backend:7860"

tab1, tab2 = st.tabs([
    "Single Prediction",
    "Batch Prediction"
])

# ============================================================
# SINGLE PREDICTION
# ============================================================

with tab1:
    st.subheader("Single Product-Store Prediction")

    product_weight = st.number_input(
        "Product Weight",
        min_value=0.0,
        value=12.66
    )

    sugar_content = st.selectbox(
        "Product Sugar Content",
        ["Low Sugar", "Regular", "No Sugar"]
    )

    allocated_area = st.number_input(
        "Product Allocated Area",
        min_value=0.0,
        value=0.027,
        format="%.3f"
    )

    product_mrp = st.number_input(
        "Product MRP",
        min_value=0.0,
        value=117.08
    )

    store_size = st.selectbox(
        "Store Size",
        ["Small", "Medium", "High"]
    )

    city_type = st.selectbox(
        "Store Location City Type",
        ["Tier 1", "Tier 2", "Tier 3"]
    )

    store_type = st.selectbox(
        "Store Type",
        [
            "Departmental Store",
            "Supermarket Type1",
            "Supermarket Type2",
            "Food Mart"
        ]
    )

    product_id_char = st.selectbox(
        "Product ID Prefix",
        ["FD", "DR", "NC"]
    )

    store_age = st.number_input(
        "Store Age in Years",
        min_value=0,
        value=16
    )

    product_category = st.selectbox(
        "Product Type Category",
        ["Perishables", "Non Perishables"]
    )

    if st.button("Predict Sales"):
        payload = {
            "Product_Weight": product_weight,
            "Product_Sugar_Content": sugar_content,
            "Product_Allocated_Area": allocated_area,
            "Product_MRP": product_mrp,
            "Store_Size": store_size,
            "Store_Location_City_Type": city_type,
            "Store_Type": store_type,
            "Product_Id_char": product_id_char,
            "Store_Age_Years": store_age,
            "Product_Type_Category": product_category
        }

        try:
            response = requests.post(
                f"{API_URL}/v1/predict",
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                predicted_sales = (
                    response.json()["predicted_sales"]
                )

                st.success(
                    f"Predicted Sales: "
                    f"{predicted_sales:,.2f}"
                )
            else:
                st.error(response.text)

        except Exception as error:
            st.error(f"Backend connection error: {error}")


# ============================================================
# BATCH PREDICTION
# ============================================================

with tab2:
    st.subheader("Batch CSV Prediction")

    uploaded_file = st.file_uploader(
        "Upload a CSV file containing product-store records",
        type=["csv"]
    )

    if uploaded_file is not None:

        preview_df = pd.read_csv(uploaded_file)

        st.write("Uploaded Data Preview")
        st.dataframe(preview_df)

        if st.button("Run Batch Prediction"):

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "text/csv"
                )
            }

            try:
                response = requests.post(
                    f"{API_URL}/v1/predictbatch",
                    files=files,
                    timeout=60
                )

                if response.status_code == 200:

                    api_result = response.json()

                    prediction_df = pd.DataFrame(
                        api_result["predictions"]
                    )

                    st.success(
                        f"Successfully predicted "
                        f"{api_result['row_count']} rows."
                    )

                    st.write("Prediction Results")
                    st.dataframe(prediction_df)

                    csv_output = prediction_df.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="Download Predictions",
                        data=csv_output,
                        file_name="superkart_predictions.csv",
                        mime="text/csv"
                    )

                else:
                    st.error(response.text)

            except Exception as error:
                st.error(
                    f"Backend connection error: {error}"
                )
