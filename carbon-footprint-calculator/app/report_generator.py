"""PDF report generation for the AI Carbon Footprint Calculator."""
import tempfile
from datetime import date

from fpdf import FPDF

ACCENT = (244, 63, 94)
DARK = (30, 30, 30)

TREE_OFFSET_KG_PER_YEAR = 22
SOLAR_PANEL_OFFSET_KG_PER_YEAR = 500
CAR_FREE_DAY_KG = 12


class _ReportPDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def _section_title(pdf, text):
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(*ACCENT)
    pdf.cell(0, 12, text, new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(*DARK)
    pdf.ln(2)


def _page1_summary(pdf, predicted_co2, grade, segment):
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(*ACCENT)
    pdf.cell(0, 20, "Carbon Footprint Report", new_x="LMARGIN", new_y="NEXT", align="C")

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, f"Generated on {date.today().isoformat()}", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(15)

    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(*DARK)
    pdf.cell(0, 10, f"User Segment: {segment}", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 32)
    pdf.set_text_color(*ACCENT)
    pdf.cell(0, 20, f"{predicted_co2:,.0f} kg CO2/year", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(*DARK)
    pdf.cell(0, 16, f"Grade: {grade}", new_x="LMARGIN", new_y="NEXT", align="C")


def _page2_inputs(pdf, user_inputs):
    pdf.add_page()
    _section_title(pdf, "Your Input Summary")

    pdf.set_font("Helvetica", "", 11)
    row_height = 8
    for key, value in user_inputs.items():
        if isinstance(value, list):
            value = ", ".join(value) if value else "None"
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(90, row_height, str(key), border=1)
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, row_height, str(value), border=1, new_x="LMARGIN", new_y="NEXT")


def _page3_breakdown(pdf, breakdown):
    pdf.add_page()
    _section_title(pdf, "Category Breakdown")

    total = sum(breakdown.values()) or 1
    pdf.set_font("Helvetica", "", 12)
    for category, value in breakdown.items():
        pct = value / total * 100
        pdf.cell(0, 10, f"{category}: {value:,.0f} kg CO2/year ({pct:.1f}%)", new_x="LMARGIN", new_y="NEXT")


def _page4_recommendations(pdf, recommendations):
    pdf.add_page()
    _section_title(pdf, "Personalized Recommendations")

    if not recommendations:
        pdf.set_font("Helvetica", "", 12)
        pdf.cell(0, 10, "Great job! No high-impact recommendations were triggered.", new_x="LMARGIN", new_y="NEXT")
        return

    for rec in recommendations:
        pdf.set_font("Helvetica", "B", 12)
        pdf.multi_cell(0, 8, f"- {rec['text']}  (save ~{rec['savings']:,.0f} kg CO2/year)", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(90, 90, 90)
        pdf.multi_cell(0, 6, f"  {rec['tip']}", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(*DARK)
        pdf.ln(2)

    total_savings = sum(r["savings"] for r in recommendations)
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*ACCENT)
    pdf.multi_cell(0, 8, f"Total potential savings: {total_savings:,.0f} kg CO2/year", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(*DARK)


def _page5_offsets(pdf, predicted_co2):
    pdf.add_page()
    _section_title(pdf, "Carbon Offset Equivalents")

    trees = predicted_co2 / TREE_OFFSET_KG_PER_YEAR
    panels = predicted_co2 / SOLAR_PANEL_OFFSET_KG_PER_YEAR
    car_free_days = predicted_co2 / CAR_FREE_DAY_KG

    pdf.set_font("Helvetica", "", 12)
    pdf.multi_cell(0, 10, f"To offset your {predicted_co2:,.0f} kg CO2/year footprint, you would need:", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.cell(0, 10, f"- {trees:,.0f} trees planted and grown for one year", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 10, f"- {panels:,.1f} solar panels installed for one year", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 10, f"- {car_free_days:,.0f} car-free days", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(8)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(*ACCENT)
    pdf.cell(0, 10, "Tips to lower your footprint", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(*DARK)
    pdf.set_font("Helvetica", "", 11)
    tips = [
        "Small, consistent lifestyle changes compound over a year.",
        "Prioritize the highest-savings recommendations first.",
        "Track your footprint periodically to measure progress.",
    ]
    for tip in tips:
        pdf.multi_cell(0, 8, f"- {tip}", new_x="LMARGIN", new_y="NEXT")


def generate_report(user_inputs, predicted_co2, grade, segment, recommendations, breakdown):
    """Build the 5-page PDF report and return the path to the saved file."""
    pdf = _ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=20)

    _page1_summary(pdf, predicted_co2, grade, segment)
    _page2_inputs(pdf, user_inputs)
    _page3_breakdown(pdf, breakdown)
    _page4_recommendations(pdf, recommendations)
    _page5_offsets(pdf, predicted_co2)

    tmp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    pdf.output(tmp_file.name)
    return tmp_file.name
