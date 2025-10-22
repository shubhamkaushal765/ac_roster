import streamlit as st
from backend_algo import run_algo, plot_officer_timetable_with_labels, plot_officer_schedule_with_labels

# === Streamlit setup ===
st.set_page_config(page_title="AC roster Morning", layout="wide")
st.title("Generate AC roster (Morning)")

# === User Inputs ===
main_officers_reported = st.text_input("Main Officers Reported", value="1-18")
report_gl_counters = st.text_input("GL Counters", value="4AC1, 8AC11, 12AC21, 16AC31")
handwritten_counters = st.text_input("Handwritten Counters (30mins only)", value="3AC12,5AC13")
OT_counters = st.text_input("OT Counters (30mins only)", value="2,20,40")

sos_timings = st.text_area(
    "SOS Timings",
    value='1000-1300, 2000-2200, 1315-1430;2030-2200,1315-1430;2030-2200, 1000-1130;1315-1430;2030-2200, 1200-2200, 1400-1830, 1400-1830, 1630-1830,1330-2200,1800-2030, 1800-2030, 1730-2200, 1730-1900, 1700-1945'
)
ro_ra_officers = st.text_input("RO/RA Officers", value="3RO2100, 11RO1700,15RO2130")

# === Generate button ===
if st.button("Generate Schedule"):
    if not main_officers_reported.strip():
        st.error("⚠️ 'Main Officers Reported' is a required field. Please enter a value before submitting.")
    else:
        # Run the backend algorithm only when input is valid
        counter_matrix, final_counter_matrix, officer_schedule = run_algo(
            main_officers_reported,
            report_gl_counters,
            sos_timings,
            ro_ra_officers,
            handwritten_counters,
            OT_counters
        )

        # Display outputs
        st.subheader("Counter Timetable w/o SOS")
        fig1= plot_officer_timetable_with_labels(counter_matrix)
        st.plotly_chart(fig1, use_container_width=True, key="fig_counter_matrix")

        st.subheader("Counter Timetable w SOS")
        fig2 = plot_officer_timetable_with_labels(final_counter_matrix)
        st.plotly_chart(fig2, use_container_width=True, key="fig_counter_matrix_w_SOS")

        st.subheader("Officer Timetable")
        fig3 = plot_officer_schedule_with_labels(officer_schedule)
        st.plotly_chart(fig3, use_container_width=True, key="fig_officer_matrix")
