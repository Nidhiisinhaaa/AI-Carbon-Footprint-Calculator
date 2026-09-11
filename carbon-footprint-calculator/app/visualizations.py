"""Plotly chart builders for the CarbonIQ dashboard.

All charts share a clean light theme and a consistent red/pink/peach color
scheme (#F43F5E, #EC4899, #FFB4A2) with transparent backgrounds so they
blend into the app's white background. Severity-coded elements (gauge
zones) keep their green-to-red meaning so "good vs bad" stays legible.
"""
import plotly.graph_objects as go

RED = "#F43F5E"
RED_DARK = "#BE123C"
PINK = "#EC4899"
PEACH = "#FFB4A2"
PALETTE_SEQUENCE = ["#F43F5E", "#EC4899", "#FB7185", "#FFB4A2", "#FDBA94", "#FECDD3"]

HEATING_CO2_MAP = {"coal": 2000, "natural gas": 1200, "electricity": 900, "wood": 1500}
DIET_CO2_MAP = {"omnivore": 1800, "pescatarian": 1200, "vegetarian": 900, "vegan": 600}

COUNTRY_AVERAGES = {
    "India": 1900,
    "Brazil": 2300,
    "World Average": 4700,
    "UK": 5500,
    "China": 7400,
    "Germany": 9400,
    "Japan": 9700,
    "USA": 16000,
}

TEXT_COLOR = "#1F2937"
GRID_COLOR = "rgba(31, 41, 55, 0.1)"

_TRANSPARENT_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT_COLOR),
    title_font=dict(color=TEXT_COLOR),
    legend=dict(font=dict(color=TEXT_COLOR)),
    xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, color=TEXT_COLOR),
    yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, color=TEXT_COLOR),
)


def compute_breakdown(user_inputs, raw_features):
    """Approximate CO2 contribution by category for a single user."""
    return {
        "Transport": raw_features.get("Travel_Impact", 0),
        "Home Energy": HEATING_CO2_MAP.get(user_inputs.get("Heating Energy Source"), 1200),
        "Diet": DIET_CO2_MAP.get(user_inputs.get("Diet"), 1200),
        "Shopping": raw_features.get("Shopping_Impact", 0),
        "Digital": raw_features.get("Total_Screen_Time", 0) * 50,
        "Waste": raw_features.get("Waste_Score", 0) * 100,
    }


def compute_dataset_average_breakdown(df):
    """Same breakdown formulas averaged across the cleaned dataset."""
    if df is None or df.empty:
        return None

    waste_bag_map = {"small": 1, "medium": 2, "large": 3, "extra large": 4}
    vehicle_map = {"petrol": 2.3, "diesel": 2.7, "hybrid": 1.5, "lpg": 1.8, "electric": 0.5, "None": 0}

    travel_impact = df["Vehicle Monthly Distance Km"] * df["Vehicle Type"].map(vehicle_map).fillna(0)
    shopping_impact = df["How Many New Clothes Monthly"] * 10
    total_screen_time = df["How Long TV PC Daily Hour"] + df["How Long Internet Daily Hour"]
    waste_score = df["Waste Bag Weekly Count"] * df["Waste Bag Size"].map(waste_bag_map)
    home_energy = df["Heating Energy Source"].map(HEATING_CO2_MAP).fillna(1200)
    diet = df["Diet"].map(DIET_CO2_MAP).fillna(1200)

    return {
        "Transport": float(travel_impact.mean()),
        "Home Energy": float(home_energy.mean()),
        "Diet": float(diet.mean()),
        "Shopping": float(shopping_impact.mean()),
        "Digital": float((total_screen_time * 50).mean()),
        "Waste": float((waste_score * 100).mean()),
    }


def breakdown_donut_chart(breakdown):
    labels = list(breakdown.keys())
    values = list(breakdown.values())
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=PALETTE_SEQUENCE),
        textinfo="label+percent",
    )])
    fig.update_layout(title="Your CO2 Breakdown by Category", **_TRANSPARENT_LAYOUT)
    return fig


def breakdown_comparison_bar(user_breakdown, avg_breakdown):
    categories = list(user_breakdown.keys())
    user_vals = [user_breakdown[c] for c in categories]
    avg_vals = [avg_breakdown[c] if avg_breakdown else 0 for c in categories]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="You", x=categories, y=user_vals, marker_color=RED))
    fig.add_trace(go.Bar(name="Dataset Average", x=categories, y=avg_vals, marker_color=PEACH))
    fig.update_layout(
        title="Your Value vs Dataset Average by Category",
        barmode="group",
        yaxis_title="kg CO2/year",
        **_TRANSPARENT_LAYOUT,
    )
    return fig


def global_comparison_chart(user_co2):
    countries = dict(COUNTRY_AVERAGES)
    countries["You"] = user_co2
    sorted_items = sorted(countries.items(), key=lambda kv: kv[1])
    names = [k for k, _ in sorted_items]
    values = [v for _, v in sorted_items]
    colors = [PINK if n == "You" else PEACH for n in names]

    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation="h",
        marker_color=colors,
        text=[f"{v:,.0f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Your Footprint vs Country Averages (kg CO2/year)",
        xaxis_title="kg CO2/year",
        **_TRANSPARENT_LAYOUT,
    )
    return fig


def whatif_comparison_chart(before, after):
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Before", x=["Carbon Footprint"], y=[before], marker_color=RED_DARK))
    fig.add_trace(go.Bar(name="After", x=["Carbon Footprint"], y=[after], marker_color=PEACH))
    fig.update_layout(
        title="Before vs After — What-If Simulation",
        barmode="group",
        yaxis_title="kg CO2/year",
        **_TRANSPARENT_LAYOUT,
    )
    return fig


def carbon_gauge_chart(value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": " kg", "font": {"color": TEXT_COLOR}},
        gauge={
            "axis": {"range": [0, 20000], "tickcolor": TEXT_COLOR},
            "bar": {"color": PINK},
            "bgcolor": "white",
            "bordercolor": GRID_COLOR,
            "steps": [
                {"range": [0, 2000], "color": "#A7F3D0"},
                {"range": [2000, 5000], "color": "#FDE68A"},
                {"range": [5000, 8000], "color": "#FED7AA"},
                {"range": [8000, 20000], "color": "#FECACA"},
            ],
            "threshold": {
                "line": {"color": TEXT_COLOR, "width": 3},
                "thickness": 0.9,
                "value": value,
            },
        },
        title={"text": "Carbon Footprint Gauge (kg CO2/year)", "font": {"color": TEXT_COLOR}},
    ))
    fig.update_layout(**_TRANSPARENT_LAYOUT)
    return fig
