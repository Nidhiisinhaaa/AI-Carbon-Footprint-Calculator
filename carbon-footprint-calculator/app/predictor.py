"""Feature engineering, encoding/scaling, prediction and segmentation logic."""
import pandas as pd

from utils import (
    CATEGORICAL_COLUMNS,
    COOKING_OPTIONS,
    RECYCLING_OPTIONS,
    VEHICLE_IMPACT_MAP,
    WASTE_BAG_SIZE_MAP,
    load_artifacts,
)


def engineer_features(user_inputs: dict) -> dict:
    """Build a raw (pre-encoding) feature dict from sidebar user inputs.

    Replicates the same transformations applied to the training data in
    notebooks/model_training.ipynb (Phase 1 cleaning + Phase 2 feature
    engineering), so a raw row here matches the model's expected schema.
    """
    is_private = user_inputs["Transport"] == "private"
    vehicle_type = user_inputs.get("Vehicle Type") if is_private else "None"
    vehicle_type = vehicle_type or "None"
    vehicle_distance = user_inputs["Vehicle Monthly Distance Km"] if is_private else 0

    row = {
        "Body Type": user_inputs["Body Type"],
        "Sex": user_inputs["Sex"],
        "Diet": user_inputs["Diet"],
        "How Often Shower": user_inputs["How Often Shower"],
        "Heating Energy Source": user_inputs["Heating Energy Source"],
        "Transport": user_inputs["Transport"],
        "Vehicle Type": vehicle_type,
        "Social Activity": user_inputs["Social Activity"],
        "Monthly Grocery Bill": user_inputs["Monthly Grocery Bill"],
        "Frequency of Traveling by Air": user_inputs["Frequency of Traveling by Air"],
        "Vehicle Monthly Distance Km": vehicle_distance,
        "Waste Bag Size": user_inputs["Waste Bag Size"],
        "Waste Bag Weekly Count": user_inputs["Waste Bag Weekly Count"],
        "How Long TV PC Daily Hour": user_inputs["How Long TV PC Daily Hour"],
        "How Many New Clothes Monthly": user_inputs["How Many New Clothes Monthly"],
        "How Long Internet Daily Hour": user_inputs["How Long Internet Daily Hour"],
        "Energy efficiency": user_inputs["Energy efficiency"],
    }

    selected_recycling = user_inputs.get("Recycling", []) or []
    for r_type in RECYCLING_OPTIONS:
        row[f"Recycling_{r_type}"] = 1 if r_type in selected_recycling else 0

    selected_cooking = user_inputs.get("Cooking With", []) or []
    for c_type in COOKING_OPTIONS:
        row[f"Cooking_{c_type}"] = 1 if c_type in selected_cooking else 0

    row["Total_Screen_Time"] = row["How Long TV PC Daily Hour"] + row["How Long Internet Daily Hour"]
    row["Waste_Score"] = row["Waste Bag Weekly Count"] * WASTE_BAG_SIZE_MAP.get(row["Waste Bag Size"], 0)
    row["Green_Score"] = sum(row[f"Recycling_{r}"] for r in RECYCLING_OPTIONS)
    row["Travel_Impact"] = row["Vehicle Monthly Distance Km"] * VEHICLE_IMPACT_MAP.get(row["Vehicle Type"], 0)
    row["Shopping_Impact"] = row["How Many New Clothes Monthly"] * 10

    return row


def _prepare_model_input(user_inputs: dict):
    """Return (artifacts, raw_features_dict, scaled_dataframe) for a single user."""
    artifacts = load_artifacts()
    feature_names = artifacts["feature_names"]

    raw = engineer_features(user_inputs)
    df = pd.DataFrame([raw])[feature_names]

    encoded = df.copy()
    for col in CATEGORICAL_COLUMNS:
        le = artifacts["encoders"][col]
        value = str(encoded.at[0, col])
        if value not in le.classes_:
            value = le.classes_[0]
        encoded[col] = le.transform([value])

    scaled = artifacts["scaler"].transform(encoded[feature_names])
    scaled_df = pd.DataFrame(scaled, columns=feature_names)

    return artifacts, raw, scaled_df


def predict_carbon(user_inputs: dict) -> float:
    """Return predicted CO2 in kg/year for the given user inputs."""
    artifacts, _raw, scaled_df = _prepare_model_input(user_inputs)
    prediction = artifacts["model"].predict(scaled_df)[0]
    return float(max(prediction, 0))


def get_user_segment(user_inputs: dict) -> str:
    """Return the KMeans-derived segment name for the given user inputs."""
    artifacts, _raw, scaled_df = _prepare_model_input(user_inputs)
    cluster_id = artifacts["kmeans"].predict(scaled_df)[0]
    return artifacts["cluster_labels"].get(cluster_id, "Average Consumer")


def get_recommendations(user_inputs: dict, predicted_co2: float) -> dict:
    """Rules-based personalized recommendations, sorted by potential savings.

    Returns {"recommendations": [...], "total_potential_savings": float}
    where each recommendation is {"text", "savings", "tip"}.
    """
    recommendations = []

    if user_inputs.get("Diet") == "omnivore":
        recommendations.append({
            "text": "Switch to a vegetarian diet",
            "savings": 600,
            "tip": "Meat production is far more carbon-intensive than plant-based foods due to land use, methane emissions, and feed requirements.",
        })

    if user_inputs.get("Transport") == "private" and user_inputs.get("Vehicle Type") == "petrol":
        recommendations.append({
            "text": "Consider switching to an electric vehicle",
            "savings": 1200,
            "tip": "EVs produce zero tailpipe emissions and are significantly cleaner over their lifecycle, especially on renewable grids.",
        })
    elif user_inputs.get("Transport") == "private":
        recommendations.append({
            "text": "Use public transport 3 days/week",
            "savings": 800,
            "tip": "Sharing a ride with many passengers cuts your per-person emissions dramatically compared to driving alone.",
        })

    if user_inputs.get("Heating Energy Source") in ("coal", "natural gas"):
        recommendations.append({
            "text": "Switch to solar or another renewable heating source",
            "savings": 500,
            "tip": "Fossil-fuel heating is one of the largest contributors to home energy emissions.",
        })

    if user_inputs.get("How Many New Clothes Monthly", 0) > 10:
        recommendations.append({
            "text": "Reduce fast fashion purchases",
            "savings": 200,
            "tip": "Clothing manufacturing is resource- and energy-intensive; buying fewer, longer-lasting items reduces demand.",
        })

    total_screen_time = user_inputs.get("How Long TV PC Daily Hour", 0) + user_inputs.get("How Long Internet Daily Hour", 0)
    if total_screen_time > 8:
        recommendations.append({
            "text": "Reduce screen time by 2 hours",
            "savings": 100,
            "tip": "Devices and the data centers behind streaming/browsing consume electricity that adds up over a year.",
        })

    if not user_inputs.get("Recycling"):
        recommendations.append({
            "text": "Start recycling paper and plastic",
            "savings": 150,
            "tip": "Recycling reduces the energy needed to manufacture new materials from raw resources.",
        })

    if user_inputs.get("Waste Bag Weekly Count", 0) > 3:
        recommendations.append({
            "text": "Reduce waste - composting can help",
            "savings": 200,
            "tip": "Composting organic waste avoids methane emissions from landfill decomposition.",
        })

    if user_inputs.get("Frequency of Traveling by Air") in ("frequently", "very frequently"):
        recommendations.append({
            "text": "Reduce air travel or buy carbon offsets",
            "savings": 1000,
            "tip": "Air travel is one of the most carbon-intensive activities per hour of any common lifestyle choice.",
        })

    if user_inputs.get("Energy efficiency") == "No":
        recommendations.append({
            "text": "Use energy-efficient appliances",
            "savings": 300,
            "tip": "Energy-efficient appliances use less electricity for the same output, lowering both emissions and bills.",
        })

    recommendations.sort(key=lambda r: r["savings"], reverse=True)
    total_potential_savings = sum(r["savings"] for r in recommendations)

    return {
        "recommendations": recommendations,
        "total_potential_savings": total_potential_savings,
    }


def analyze(user_inputs: dict) -> dict:
    """Single-pass helper: prediction + segment + raw engineered features.

    Avoids recomputing the feature pipeline three times when app.py needs
    the prediction, the segment, and the raw features (for the breakdown
    dashboard) all at once.
    """
    artifacts, raw, scaled_df = _prepare_model_input(user_inputs)
    prediction = float(max(artifacts["model"].predict(scaled_df)[0], 0))
    cluster_id = artifacts["kmeans"].predict(scaled_df)[0]
    segment = artifacts["cluster_labels"].get(cluster_id, "Average Consumer")
    return {
        "prediction": prediction,
        "segment": segment,
        "raw_features": raw,
    }
