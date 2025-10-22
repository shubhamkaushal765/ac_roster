import streamlit as st
from backend_algo import run_algo, plot_officer_timetable_with_labels, plot_officer_schedule_with_labels

# === Streamlit setup ===
st.set_page_config(page_title="AC roster Morning", layout="wide")
st.title("Generate AC roster (Morning)")
st.markdown(
    "<p style='font-size:14px; color:gray; margin-top:-10px;'>💡 For better display on mobile, please enable <b>Desktop site</b> in your browser settings.</p>",
    unsafe_allow_html=True
)
instructions = """

<h3>How to input SOS timings</h3>

1. Different officers are separated by commas `,`.
2. If an officer has multiple SOS timings, separate them with semicolons `;`.
3. Optional pre-assigned counters must be enclosed in parentheses `()` before the time. 
   Only valid if the first SOS timing starts at `1000`.
4. Times are in 24-hour `HHMM-HHMM` format.

<h4> Example </h4> \n
`(AC22)1000-1300, 2000-2200, 1315-1430;2030-2200, (AC23)1000-1130;1315-1430;2030-2200, 1200-2200` \n

- Officer 1: pre-assigned counter 22 at `1000`, also has sos `2000-2200`. \n
- Officer 2: multiple SOS timings: `1315-1430` and `2030-2200`. \n
- Officer 3: pre-assigned counter 23 at `1000`, also has sos from `1315-1430` and `2030-2200`. \n
- Remaining officers: single shifts without pre-assigned counters. \n


<h3>Additional Note</h3> \n
- `GL counters` is only applicable for officers in main roster with S/N 4,8,12 ... \n
- `Handwritten counters` for `S/N 3` in main roster to `AC12` is written as `3AC12` \n
- `RO/RA Officers` for `S/N 11` to `RO` at `1700` is written as `11RO1700` \n

"""

st.markdown(
    f"""
    <div style="
        border: 1px solid white;
        border-radius: 8px;
        padding: 10px;
    ">
        {instructions}
    </div>
    """,
    unsafe_allow_html=True
)

# === User Inputs ===
st.markdown("<br>", unsafe_allow_html=True)

main_officers_reported = st.text_input("Main Officers Reported", value="1-18")
report_gl_counters = st.text_input("GL Counters", value="4AC1, 8AC11, 12AC21, 16AC31")
handwritten_counters = st.text_input("Handwritten Counters (30mins only)", value="3AC12,5AC13")
OT_counters = st.text_input("OT Counters (30mins only)", value="2,20,40")
ro_ra_officers = st.text_input("RO/RA Officers", value="3RA1200, 11RO1700,15RO2130")

sos_timings = st.text_area(
    "SOS Timings",
    value='(AC22)1000-1300, 2000-2200, 1315-1430;2030-2200,1315-1430;2030-2200, (AC23)1000-1130;1315-1430;2030-2200, 1200-2200, 1400-1830, 1400-1830, 1630-1830,1330-2200,1800-2030, 1800-2030, 1730-2200, 1730-1900, 1700-1945'
)

# === Generate button ===
if st.button("Generate Schedule"):
    if not main_officers_reported.strip():
        st.error("⚠️ 'Main Officers Reported' is a required field. Please enter a value before submitting.")
    else:
        # Run the backend algorithm only when input is valid
        counter_matrix, final_counter_matrix, officer_schedule, output_text = run_algo(
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

        # Display in a scrollable text area
        st.text_area("Counter Manning", value=output_text[0], height=400)

        # Optional: Provide a copy/download button
        st.download_button(
            label="Copy to Clipboard",
            data=output_text[0],
            file_name="statistics_output.txt",
            mime="text/plain",
            key = 'stats1'
        )

        st.subheader("Counter Timetable w SOS")
        fig2 = plot_officer_timetable_with_labels(final_counter_matrix)
        st.plotly_chart(fig2, use_container_width=True, key="fig_counter_matrix_w_SOS")
        # Display in a scrollable text area
        st.text_area("Counter Manning including SOS", value=output_text[1], height=400)

        # Optional: Provide a copy/download button
        st.download_button(
            label="Copy  to Clipboard",
            data=output_text[1],
            file_name="statistics_output.txt",
            mime="text/plain",
            key = 'stats2'
        )      
        st.subheader("Officer Timetable")
        fig3 = plot_officer_schedule_with_labels(officer_schedule)
        st.plotly_chart(fig3, use_container_width=True, key="fig_officer_matrix")
