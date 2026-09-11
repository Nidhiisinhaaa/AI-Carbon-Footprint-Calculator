"""CarbonIQ — main Streamlit app."""
import os

import streamlit as st

import visualizations as viz
from predictor import analyze, get_recommendations, predict_carbon
from report_generator import (
    CAR_FREE_DAY_KG,
    SOLAR_PANEL_OFFSET_KG_PER_YEAR,
    TREE_OFFSET_KG_PER_YEAR,
    generate_report,
)
from utils import (
    CLUSTER_COLORS,
    ModelLoadError,
    get_dataset_average_emission,
    get_dropdown_options,
    get_vehicle_type_options,
    load_artifacts,
    load_cleaned_dataset,
)

st.set_page_config(
    page_title="CarbonIQ",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    :root {
        --accent: #F43F5E;
        --accent-dark: #BE123C;
        --accent-pink: #EC4899;
        --accent-peach: #FFB4A2;
    }
    h1, h2, h3 { color: var(--accent) !important; }
    .stTabs [data-baseweb="tab"] { font-weight: 600; }
    .stTabs [aria-selected="true"] { color: var(--accent) !important; }
    div[data-testid="stMetric"] {
        background: rgba(244, 63, 94, 0.08);
        border: 1px solid rgba(244, 63, 94, 0.35);
        border-radius: 12px;
        padding: 14px 16px;
        transition: transform 0.15s ease;
    }
    div[data-testid="stMetric"]:hover { transform: translateY(-2px); }
    div[data-testid="stExpander"] {
        border: 1px solid rgba(236, 72, 153, 0.25);
        border-radius: 10px;
    }
    .stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent-pink) 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover, .stDownloadButton > button:hover {
        background: linear-gradient(135deg, var(--accent-dark) 0%, var(--accent) 100%);
        color: white;
    }
    footer.app-footer {
        text-align: center;
        color: #6b7280;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(244, 63, 94, 0.2);
    }

    /* Hero section */
    .hero {
        background: linear-gradient(135deg, #FFB4A2 0%, #EC4899 55%, #F43F5E 100%);
        border-radius: 18px;
        padding: 2.5rem 2rem;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 30px rgba(244, 63, 94, 0.25);
    }
    .hero h1 {
        color: white !important;
        font-size: 2.4rem;
        margin: 0 0 0.5rem 0;
    }
    .hero p {
        color: rgba(255,255,255,0.92);
        font-size: 1.05rem;
        margin: 0;
    }

    /* Nav bar buttons */
    div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button {
        background: rgba(244, 63, 94, 0.08);
        color: #F43F5E;
        border: 1px solid rgba(244, 63, 94, 0.3);
        font-weight: 600;
    }
    div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button:hover {
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent-pink) 100%);
        color: white;
    }

    /* About / Contact cards */
    .info-card {
        background: rgba(244, 63, 94, 0.06);
        border: 1px solid rgba(244, 63, 94, 0.25);
        border-radius: 14px;
        padding: 1.5rem 1.75rem;
        margin-bottom: 1rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Hero section + Home / About / Contact navigation
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🌍 CarbonIQ</h1>
        <p>Estimate your yearly carbon footprint from lifestyle habits, powered by machine learning.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "page" not in st.session_state:
    st.session_state["page"] = "Home"

nav_col1, nav_col2, nav_col3 = st.columns(3)
if nav_col1.button("🏠 Home", width="stretch"):
    st.session_state["page"] = "Home"
if nav_col2.button("ℹ️ About", width="stretch"):
    st.session_state["page"] = "About"
if nav_col3.button("✉️ Contact", width="stretch"):
    st.session_state["page"] = "Contact"

page = st.session_state["page"]


def render_about():
    st.header("About This Project")
    st.markdown(
        """
        <div class="info-card">
        <b>CarbonIQ</b> estimates an individual's yearly carbon
        footprint (kg CO2/year) from everyday lifestyle habits — diet, transport, home
        energy, digital usage, and waste — using a machine learning model trained on
        real-world survey data.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("What it does")
    st.markdown(
        "- Predicts your yearly CO2 emissions from 30+ lifestyle features\n"
        "- Grades your footprint A–F and segments you into a KMeans-based user group\n"
        "- Breaks your footprint down by category and compares it to country averages\n"
        "- Lets you simulate lifestyle changes and see the potential savings\n"
        "- Suggests personalized, ranked recommendations\n"
        "- Exports a full PDF report you can keep or share"
    )

    st.subheader("Tech stack")
    st.markdown(
        "- **Modeling**: scikit-learn (Random Forest, Linear Regression, KMeans), XGBoost\n"
        "- **App**: Streamlit\n"
        "- **Visualization**: Plotly, Matplotlib, Seaborn\n"
        "- **Reports**: fpdf2"
    )

    st.subheader("Dataset")
    st.markdown("Carbon Emission dataset, sourced from Kaggle.")


def render_contact():
    st.header("Contact")
    st.markdown(
        """
        <div class="info-card">
        Questions, feedback, or ideas for improving the calculator? Reach out below.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("contact_form"):
        name = st.text_input("Your name")
        email = st.text_input("Your email")
        message = st.text_area("Message")
        sent = st.form_submit_button("Send Message", width="stretch")

    if sent:
        if not name or not email or not message:
            st.error("Please fill in all fields before sending.")
        else:
            st.success(f"Thanks, {name}! Your message has been noted. We'll get back to you at {email}.")

    st.subheader("Find us")
    st.markdown(
        "- 📧 Email: contact@carbonfootprintcalculator.example\n"
        "- 💻 GitHub: your-repo-url-here\n"
        "- 🐦 Twitter/X: @your-handle"
    )


def render_home():
    try:
        artifacts = load_artifacts()
    except ModelLoadError as exc:
        st.error(f"⚠️ {exc}")
        st.stop()

    encoders = artifacts["encoders"]

    # -------------------------------------------------------------------
    # Sidebar — user input form
    # -------------------------------------------------------------------
    st.sidebar.header("🌱 CarbonIQ")
    st.sidebar.caption(
        "Fill in your lifestyle details across all 5 sections below, then click "
        "**Calculate** to see your estimated yearly carbon footprint."
    )

    with st.sidebar.form("user_inputs_form"):
        st.subheader("1. Personal")
        body_type = st.selectbox("Body Type", get_dropdown_options(encoders, "Body Type"), help="Your general body type category.")
        sex = st.selectbox("Sex", get_dropdown_options(encoders, "Sex"))
        diet = st.selectbox("Diet", get_dropdown_options(encoders, "Diet"), help="Diet has a major impact on your footprint — meat-heavy diets emit more.")

        st.subheader("2. Home")
        shower_freq = st.selectbox("How Often Shower", get_dropdown_options(encoders, "How Often Shower"))
        heating_source = st.selectbox("Heating Energy Source", get_dropdown_options(encoders, "Heating Energy Source"), help="Fossil-fuel heating (coal, natural gas) emits more than electricity or renewables.")
        energy_efficiency = st.selectbox("Energy Efficiency", get_dropdown_options(encoders, "Energy efficiency"), help="Do you use energy-efficient appliances at home?")
        cooking_with = st.multiselect("Cooking With", ["Stove", "Oven", "Microwave", "Grill", "Airfryer"], help="Select all appliances you regularly cook with.")

        st.subheader("3. Transport")
        transport = st.selectbox("Transport", get_dropdown_options(encoders, "Transport"), help="Your primary mode of transport.")
        vehicle_type = None
        vehicle_distance = 0
        if transport == "private":
            vehicle_type = st.selectbox("Vehicle Type", get_vehicle_type_options(encoders))
            vehicle_distance = st.slider("Vehicle Monthly Distance (Km)", 0, 5000, 500)

        st.subheader("4. Lifestyle")
        social_activity = st.selectbox("Social Activity", get_dropdown_options(encoders, "Social Activity"))
        grocery_bill = st.slider("Monthly Grocery Bill", 0, 500, 150, help="Your average monthly spend on groceries (in your local currency).")
        new_clothes = st.slider("New Clothes Monthly", 0, 50, 5, help="Fast fashion has a surprisingly high carbon cost per item.")
        air_travel_freq = st.selectbox("Frequency of Traveling by Air", get_dropdown_options(encoders, "Frequency of Traveling by Air"), help="Air travel is one of the most carbon-intensive activities per hour.")

        st.subheader("5. Digital & Waste")
        tv_pc_hours = st.slider("TV/PC Daily Hours", 0, 24, 4)
        internet_hours = st.slider("Internet Daily Hours", 0, 24, 4)
        waste_bag_size = st.selectbox("Waste Bag Size", get_dropdown_options(encoders, "Waste Bag Size"))
        waste_bag_count = st.slider("Waste Bag Weekly Count", 0, 10, 2)
        recycling = st.multiselect("Recycling", ["Paper", "Plastic", "Glass", "Metal"], help="Select all materials you regularly recycle.")

        submitted = st.form_submit_button("Calculate My Carbon Footprint", width="stretch")

    # -------------------------------------------------------------------
    # Main area — prediction result
    # -------------------------------------------------------------------
    if submitted:
        user_inputs = {
            "Body Type": body_type,
            "Sex": sex,
            "Diet": diet,
            "How Often Shower": shower_freq,
            "Heating Energy Source": heating_source,
            "Energy efficiency": energy_efficiency,
            "Cooking With": cooking_with,
            "Transport": transport,
            "Vehicle Type": vehicle_type,
            "Vehicle Monthly Distance Km": vehicle_distance,
            "Social Activity": social_activity,
            "Monthly Grocery Bill": grocery_bill,
            "How Many New Clothes Monthly": new_clothes,
            "Frequency of Traveling by Air": air_travel_freq,
            "How Long TV PC Daily Hour": tv_pc_hours,
            "How Long Internet Daily Hour": internet_hours,
            "Waste Bag Size": waste_bag_size,
            "Waste Bag Weekly Count": waste_bag_count,
            "Recycling": recycling,
        }
        if user_inputs["Transport"] == "private" and not user_inputs.get("Vehicle Type"):
            st.error("Please select a Vehicle Type since Transport is set to 'private'.")
            st.stop()

        st.session_state["user_inputs"] = user_inputs

        with st.spinner("Calculating your carbon footprint..."):
            result = analyze(user_inputs)
        st.session_state["result"] = result

    if "result" in st.session_state:
        user_inputs = st.session_state["user_inputs"]
        result = st.session_state["result"]
        predicted_co2 = result["prediction"]
        segment = result["segment"]

        if predicted_co2 < 2000:
            grade, color = "A", "#10B981"
        elif predicted_co2 < 4000:
            grade, color = "B", "#84CC16"
        elif predicted_co2 < 6000:
            grade, color = "C", "#FACC15"
        elif predicted_co2 < 8000:
            grade, color = "D", "#F97316"
        else:
            grade, color = "F", "#EF4444"

        col1, col2, col3 = st.columns(3)
        with col1:
            with st.container(border=True):
                st.metric("Your Carbon Footprint", f"{predicted_co2:,.0f} kg CO2/year")
        with col2:
            with st.container(border=True):
                st.markdown(
                    f"<h2 style='color:{color}; text-align:center; margin:0;'>Grade: {grade}</h2>",
                    unsafe_allow_html=True,
                )
        with col3:
            with st.container(border=True):
                badge_color = CLUSTER_COLORS.get(segment, "#F43F5E")
                st.markdown(
                    f"<h4 style='text-align:center; margin:0;'>Segment</h4>"
                    f"<p style='text-align:center; color:{badge_color}; font-weight:700; font-size:1.1rem;'>{segment}</p>",
                    unsafe_allow_html=True,
                )

        avg_emission = get_dataset_average_emission()
        if avg_emission:
            diff_pct = (predicted_co2 - avg_emission) / avg_emission * 100
            direction = "more" if diff_pct > 0 else "less"
            st.info(f"You emit **{abs(diff_pct):.1f}% {direction}** than the dataset average ({avg_emission:,.0f} kg CO2/year).")

        st.divider()
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Your Breakdown", "🌍 Global Comparison", "🔄 What-If Simulator", "🎯 Carbon Gauge",
        ])

        raw_features = result["raw_features"]
        cleaned_df = load_cleaned_dataset()

        with tab1:
            user_breakdown = viz.compute_breakdown(user_inputs, raw_features)
            avg_breakdown = viz.compute_dataset_average_breakdown(cleaned_df)

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(viz.breakdown_donut_chart(user_breakdown), width="stretch")
            with c2:
                st.plotly_chart(viz.breakdown_comparison_bar(user_breakdown, avg_breakdown), width="stretch")

        with tab2:
            st.plotly_chart(viz.global_comparison_chart(predicted_co2), width="stretch")
            usa_pct = predicted_co2 / viz.COUNTRY_AVERAGES["USA"] * 100
            st.markdown(f"Your footprint is equivalent to **{usa_pct:.1f}%** of an average American's.")

        with tab3:
            st.markdown("Toggle lifestyle changes to see potential savings:")
            wcol1, wcol2 = st.columns(2)
            with wcol1:
                switch_vegan = st.toggle("🥦 Switch diet to vegan")
                use_public_transport = st.toggle("🚌 Use public transport")
                reduce_clothes = st.toggle("👕 Reduce new clothes by 50%")
            with wcol2:
                reduce_screen_time = st.toggle("📱 Reduce screen time by 2 hours")
                recycle_all = st.toggle("♻️ Recycle all materials")

            sim_inputs = dict(user_inputs)
            if switch_vegan:
                sim_inputs["Diet"] = "vegan"
            if use_public_transport:
                sim_inputs["Transport"] = "public"
                sim_inputs["Vehicle Type"] = None
                sim_inputs["Vehicle Monthly Distance Km"] = 0
            if reduce_clothes:
                sim_inputs["How Many New Clothes Monthly"] = sim_inputs["How Many New Clothes Monthly"] * 0.5
            if reduce_screen_time:
                sim_inputs["How Long TV PC Daily Hour"] = max(0, sim_inputs["How Long TV PC Daily Hour"] - 2)
            if recycle_all:
                sim_inputs["Recycling"] = ["Paper", "Plastic", "Glass", "Metal"]

            any_toggle = any([switch_vegan, use_public_transport, reduce_clothes, reduce_screen_time, recycle_all])
            if any_toggle:
                new_prediction = predict_carbon(sim_inputs)
                savings = predicted_co2 - new_prediction
                savings_pct = (savings / predicted_co2 * 100) if predicted_co2 else 0
                st.success(f"Potential savings: **{savings:,.0f} kg CO2/year** ({savings_pct:.1f}% reduction)")
                st.plotly_chart(viz.whatif_comparison_chart(predicted_co2, new_prediction), width="stretch")
            else:
                st.caption("Toggle at least one change above to see the simulated impact.")

        with tab4:
            st.plotly_chart(viz.carbon_gauge_chart(predicted_co2), width="stretch")

        # ---------------------------------------------------------------
        # Recommendations
        # ---------------------------------------------------------------
        st.divider()
        st.header("💡 Personalized Recommendations")

        rec_result = get_recommendations(user_inputs, predicted_co2)
        recommendations = rec_result["recommendations"]
        total_potential_savings = rec_result["total_potential_savings"]

        if recommendations:
            for rec in recommendations:
                with st.expander(f"{rec['text']} — save ~{rec['savings']:,.0f} kg CO2/year"):
                    st.write(rec["tip"])

            savings_pct = (total_potential_savings / predicted_co2 * 100) if predicted_co2 else 0
            st.success(
                f"By following all recommendations, you could save "
                f"**{total_potential_savings:,.0f} kg CO2/year** ({savings_pct:.1f}% reduction)."
            )
        else:
            st.success("Great job! No high-impact recommendations were triggered for your profile.")

        # ---------------------------------------------------------------
        # Carbon Offset Calculator
        # ---------------------------------------------------------------
        st.divider()
        st.header("🌳 Carbon Offset Calculator")

        trees = predicted_co2 / TREE_OFFSET_KG_PER_YEAR
        panels = predicted_co2 / SOLAR_PANEL_OFFSET_KG_PER_YEAR
        car_free_days = predicted_co2 / CAR_FREE_DAY_KG

        o1, o2, o3 = st.columns(3)
        o1.metric("🌲 Trees to offset", f"{trees:,.0f}")
        o2.metric("☀️ Solar panels (1 yr)", f"{panels:,.1f}")
        o3.metric("🚗 Car-free days", f"{car_free_days:,.0f}")

        # ---------------------------------------------------------------
        # PDF report download
        # ---------------------------------------------------------------
        st.divider()
        st.header("📄 Download Your Report")

        if st.button("Generate PDF Report"):
            with st.spinner("Generating your PDF report..."):
                breakdown = viz.compute_breakdown(user_inputs, raw_features)
                pdf_path = generate_report(
                    user_inputs=user_inputs,
                    predicted_co2=predicted_co2,
                    grade=grade,
                    segment=segment,
                    recommendations=recommendations,
                    breakdown=breakdown,
                )
                with open(pdf_path, "rb") as f:
                    st.session_state["pdf_bytes"] = f.read()
                os.remove(pdf_path)

        if "pdf_bytes" in st.session_state:
            st.download_button(
                "⬇️ Download Carbon Footprint Report (PDF)",
                data=st.session_state["pdf_bytes"],
                file_name="carbon_footprint_report.pdf",
                mime="application/pdf",
            )
    else:
        st.info("Fill in the form in the sidebar and click **Calculate My Carbon Footprint** to get started.")


if page == "Home":
    render_home()
elif page == "About":
    render_about()
elif page == "Contact":
    render_contact()

st.markdown(
    '<footer class="app-footer">Built with Streamlit | Dataset: Kaggle</footer>',
    unsafe_allow_html=True,
)
