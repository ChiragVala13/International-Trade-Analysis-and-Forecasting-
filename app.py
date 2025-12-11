from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import pickle

app = Flask(__name__)

# ---------- MODEL & ENCODERS ----------

model_import = joblib.load("tuned_import_xgb.pkl")
model_export = joblib.load("tuned_export_xgb.pkl")
le_product = joblib.load("le_product.pkl")
le_country = joblib.load("le_country.pkl")

with open("feature_cols.pkl", "rb") as f:
    feature_cols = pickle.load(f)

# Safe mapping from fitted LabelEncoders
product_to_code = {cls: int(code) for code, cls in enumerate(le_product.classes_)}
country_to_code = {cls: int(code) for code, cls in enumerate(le_country.classes_)}

DEFAULT_PRODUCT_CODE = 0
DEFAULT_COUNTRY_CODE = 0


def make_feature_row(form_dict):
    """
    Build one feature row with the SAME column names used in training.
    Any unseen product/country is mapped to encoder code 0.
    """
    product = str(form_dict["product"]).strip()
    country = str(form_dict["country"]).strip()

    row = pd.DataFrame([{
        "PRODUCT": product,
        "COUNTRY": country,
        "VALUE_USD": float(form_dict["valueusd"]),
        "QUANTITY": float(form_dict["quantity"]),
        "QUANTITY_LAG1": float(form_dict["quantity_lag1"]),
        "VALUELAG1": float(form_dict["value_lag1"]),  # note: no underscore here
        "QUANTITY_rolling_mean": float(form_dict["quantity_rolling"]),
        "COUNTRY_QUANTITY_median": float(form_dict["country_qty_median"]),
        "PRODUCT_VALUE_median": float(form_dict["product_val_median"])
    }])

    # extra feature used in training
    row["LOG_VALUE_USD"] = np.log1p(row["VALUE_USD"])

    # safe encoding for categories
    row["PRODUCTENC"] = product_to_code.get(product, DEFAULT_PRODUCT_CODE)
    row["COUNTRYENC"] = country_to_code.get(country, DEFAULT_COUNTRY_CODE)

    # reorder exactly as model expects
    return row[feature_cols]


# ---------- DATA LOADING ----------

rename_map = {
    "PRINCIPLE COMMODITY": "PRODUCT",
    "COUNTRY": "COUNTRY",
    "UNIT": "UNIT",
    "QUANTITY": "QUANTITY",
    "Value(US$ million)": "VALUE_USD"
}

df_import_raw = pd.read_csv(
    r"C:\Users\CHIRAG\Desktop\minor project\Principal_Commodity_wise_import_for_the_year_202223.csv"
)
df_export_raw = pd.read_csv(
    r"C:\Users\CHIRAG\Desktop\minor project\Principal_Commodity_wise_export_for_the_year_202223 (2) (1).csv"
)

df_import_raw.rename(columns=rename_map, inplace=True)
df_export_raw.rename(columns=rename_map, inplace=True)

df_import_raw["PRODUCT"] = df_import_raw["PRODUCT"].astype(str).str.strip()
df_export_raw["PRODUCT"] = df_export_raw["PRODUCT"].astype(str).str.strip()


# ---------- CATEGORY & GST LOGIC ----------

def get_category(product_name: str) -> str:
    n = product_name.lower()
    if any(k in n for k in ["rice", "wheat", "grain", "cereal", "pulses", "dal", "lentil"]):
        return "grains"
    if "vegetable" in n or "vegetables" in n:
        return "vegetables"
    if any(k in n for k in ["ore", "crude", "raw", "cotton", "iron", "steel"]):
        return "raw_materials"
    return "others"


def get_quantity_factor_2026(category: str) -> float:
    if category in ["grains", "vegetables"]:
        return 1.20
    if category == "raw_materials":
        return 0.92
    return 0.90


def get_gst_rate(product_name: str) -> float:
    n = product_name.lower()
    low_gst_keywords = [
        "grain", "rice", "wheat", "pulses", "dal", "lentil",
        "cereal", "vegetable", "vegetables", "fruit", "fruits"
    ]
    if any(k in n for k in low_gst_keywords):
        return 0.05
    return 0.18


# ---------- ROUTES ----------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/search_product", methods=["GET"])
def search_product():
    name = request.args.get("name", "").strip()
    ptype = request.args.get("type", "import").lower()

    if not name:
        return jsonify({"error": "Missing 'name' query parameter"}), 400

    df = df_export_raw if ptype == "export" else df_import_raw

    mask = df["PRODUCT"].str.contains(name, case=False, na=False)
    df_prod = df[mask].copy()

    if df_prod.empty:
        return jsonify({"product": name, "message": "No records found"}), 200

    grouped = df_prod.groupby("COUNTRY", as_index=False).agg(
        quantity=("QUANTITY", "sum"),
        value_usd=("VALUE_USD", "sum")
    )

    total_quantity = float(grouped["quantity"].sum())
    total_value_usd = float(grouped["value_usd"].sum())

    gst_rate = get_gst_rate(name)

    by_country = []
    projected_2026 = []

    for _, row in grouped.iterrows():
        country = row["COUNTRY"]
        qty = float(row["quantity"])
        val = float(row["value_usd"])
        new_val = val * (1.0 + gst_rate)

        by_country.append({
            "country": country,
            "quantity": qty,
            "value_usd": val
        })
        projected_2026.append({
            "country": country,
            "quantity_2026": qty,
            "value_usd_2026": new_val
        })

    top = grouped.sort_values("quantity", ascending=False).head(5)
    series = [
        {
            "country": row["COUNTRY"],
            "quantity": float(row["quantity"]),
            "value_usd": float(row["value_usd"])
        }
        for _, row in top.iterrows()
    ]

    return jsonify({
        "product": name.upper(),
        "type": ptype,
        "gst_rate": gst_rate,
        "summary": {
            "total_quantity": total_quantity,
            "total_value_usd": total_value_usd,
            "by_country": by_country
        },
        "series": series,
        "gst_projection_2026": projected_2026
    })


@app.route("/api/category_summary", methods=["GET"])
def category_summary():
    ptype = request.args.get("type", "import").lower()
    df = df_export_raw if ptype == "export" else df_import_raw

    df_cat = df.copy()
    df_cat["CATEGORY"] = df_cat["PRODUCT"].apply(get_category)

    grouped = df_cat.groupby("CATEGORY", as_index=False).agg(
        total_quantity=("QUANTITY", "sum"),
        total_value_usd=("VALUE_USD", "sum")
    )

    results = []
    for _, row in grouped.iterrows():
        cat = row["CATEGORY"]
        qty = float(row["total_quantity"])
        val = float(row["total_value_usd"])
        factor = get_quantity_factor_2026(cat)
        qty_2026 = qty * factor
        results.append({
            "category": cat,
            "quantity_now": qty,
            "quantity_2026": qty_2026,
            "value_now": val
        })

    return jsonify({
        "type": ptype,
        "categories": results
    })


# ---------- XGBoost PREDICTION ROUTE ----------

@app.route("/predict_quantity", methods=["POST"])
def predict_quantity():
    """
    Expects form fields:
    product, country, valueusd, quantity,
    quantity_lag1, value_lag1, quantity_rolling,
    country_qty_median, product_val_median,
    type = 'import' or 'export'
    """
    try:
        form = request.form
        X = make_feature_row(form)
        model = model_import if form.get("type", "import") == "import" else model_export
        y_log = model.predict(X)[0]
        y = float(np.expm1(y_log))
        print("PREDICT_DEBUG:", dict(form), "=>", y)
        return jsonify({"predicted_quantity": y})
    except Exception as e:
        print("PREDICT_ERROR:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)



