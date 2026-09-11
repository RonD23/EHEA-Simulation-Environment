"""
EAHE Cooling Simulator - Streamlit app
Analytical model for a buried-pipe Earth-Air Heat Exchanger pre-cooling
system for AI facility server halls.

Run locally:
    pip install streamlit pandas
    streamlit run streamlit_app.py

Deploy for free (so the client just opens a link, no install at all):
    1. Push this file to a GitHub repo.
    2. Go to https://share.streamlit.io , sign in with GitHub.
    3. Point it at the repo/file. It builds and gives you a public URL.
"""

import math
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EAHE Cooling Simulator",
    page_icon="🌡️",
    layout="wide",
)

CP_AIR = 1005.0
RHO_AIR = 1.15
K_AIR = 0.026
PR_AIR = 0.71
MU_AIR = 1.9e-5

CLIMATE_PRESETS = {
    "Tropical": (35, 26),
    "Subtropical": (30, 22),
    "Temperate": (22, 16),
    "Arid": (42, 28),
}


# ---------------------------------------------------------------------------
# Physics (same model as eahe_simulation.py)
# ---------------------------------------------------------------------------
def mass_flow_rate(D, v, n_pipes):
    A = math.pi * D ** 2 / 4.0
    m_dot_pipe = RHO_AIR * A * v
    return m_dot_pipe, m_dot_pipe * n_pipes


def convective_h(D, v):
    Re = RHO_AIR * v * D / MU_AIR
    Nu = 0.023 * Re ** 0.8 * PR_AIR ** 0.4
    h = Nu * K_AIR / D
    return Re, h


def soil_temperature(T_soil0, k_soil, Q_load, n_pipes, days, tau_days=20.0):
    dT_max = min(10.0, 8.0 * (Q_load / 500.0) * (1.0 / k_soil) * (2.0 / max(n_pipes, 1)))
    rise = dT_max * (1 - math.exp(-days / tau_days))
    return T_soil0 + rise, rise


def run_point(L, D, z, v, n_pipes, k_soil, T_amb, T_soil0, Q_load, days):
    m_dot_pipe, m_dot_total = mass_flow_rate(D, v, n_pipes)
    T_soil, soil_rise = soil_temperature(T_soil0, k_soil, Q_load, n_pipes, days)
    Re, h = convective_h(D, v)
    NTU = h * math.pi * D * L / (m_dot_pipe * CP_AIR)
    T_out = T_soil + (T_amb - T_soil) * math.exp(-NTU)
    reduction = T_amb - T_out
    cooling_capacity_kw = m_dot_total * CP_AIR * reduction / 1000.0

    f = 0.079 * Re ** -0.25
    dP = f * (L / D) * (RHO_AIR * v ** 2 / 2.0)

    mech_backup_kw = max(0.0, Q_load - cooling_capacity_kw) / 3.0  # COP = 3
    fan_kw = dP * (m_dot_total / RHO_AIR) / 0.6 / 1000.0            # fan eff = 0.6
    pue = (Q_load + mech_backup_kw + fan_kw) / Q_load

    return {
        "T_out": T_out,
        "reduction": reduction,
        "cooling_capacity_kw": cooling_capacity_kw,
        "pressure_drop_pa": dP,
        "soil_rise": soil_rise,
        "mech_backup_kw": mech_backup_kw,
        "fan_kw": fan_kw,
        "pue": pue,
        "compliant": 18.0 <= T_out <= 27.0,
    }


# ---------------------------------------------------------------------------
# Sidebar - inputs
# ---------------------------------------------------------------------------
st.sidebar.header("Design & operating inputs")

st.sidebar.subheader("Climate preset")
preset_cols = st.sidebar.columns(2)
if "T_amb" not in st.session_state:
    st.session_state.T_amb = 35
    st.session_state.T_soil0 = 26

for i, (name, (ta, ts)) in enumerate(CLIMATE_PRESETS.items()):
    if preset_cols[i % 2].button(name, use_container_width=True):
        st.session_state.T_amb = ta
        st.session_state.T_soil0 = ts

st.sidebar.subheader("Geometry")
L = st.sidebar.slider("Pipe length (m)", 10, 200, 80, 5)
D = st.sidebar.slider("Pipe diameter (m)", 0.2, 1.0, 0.5, 0.05)
z = st.sidebar.slider("Burial depth (m)", 1.0, 5.0, 2.5, 0.5)
n_pipes = st.sidebar.slider("Parallel pipes", 1, 30, 10, 1)

st.sidebar.subheader("Airflow")
v = st.sidebar.slider("Airflow velocity (m/s)", 1.0, 10.0, 4.0, 0.5)

st.sidebar.subheader("Environment")
T_amb = st.sidebar.slider("Ambient / inlet air temp (C)", 20, 48, key="T_amb")
T_soil0 = st.sidebar.slider("Undisturbed soil temp (C)", 14, 30, key="T_soil0")
k_soil = st.sidebar.slider("Soil thermal conductivity (W/m.K)", 0.5, 3.0, 1.5, 0.1)

st.sidebar.subheader("Load & duration")
Q_load = st.sidebar.slider("AI facility heat load (kW)", 50, 1000, 400, 10)
days = st.sidebar.slider("Operating days (soil saturation)", 0, 90, 15, 1)

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("Earth-Air Heat Exchanger cooling simulator")
st.caption(
    "Analytical model of a buried-pipe pre-cooling system for AI-facility "
    "server halls: mass flow, convective heat transfer, transient soil "
    "saturation, and pressure drop."
)

r0 = run_point(L, D, z, v, n_pipes, k_soil, T_amb, T_soil0, Q_load, 0)
r = run_point(L, D, z, v, n_pipes, k_soil, T_amb, T_soil0, Q_load, days)
derate = (
    max(0.0, (r0["cooling_capacity_kw"] - r["cooling_capacity_kw"]) / r0["cooling_capacity_kw"] * 100)
    if r0["cooling_capacity_kw"] > 0 else 0.0
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Outlet air temp", f"{r['T_out']:.1f} C")
c2.metric("Temp reduction", f"{r['reduction']:.1f} C")
c3.metric("Cooling capacity", f"{r['cooling_capacity_kw']:.0f} kW")
c4.metric("Pressure drop", f"{r['pressure_drop_pa']:.0f} Pa")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Soil temp rise", f"{r['soil_rise']:.1f} C")
c6.metric("Thermal derating", f"{derate:.1f} %")
c7.metric("Estimated PUE", f"{r['pue']:.2f}")
c8.metric("Backup cooling needed", f"{r['mech_backup_kw']:.0f} kW")

if r["compliant"]:
    st.success(f"Outlet temperature {r['T_out']:.1f} C is within the 18-27 C AI hardware inlet guideline.")
else:
    st.error(f"Outlet temperature {r['T_out']:.1f} C is outside the 18-27 C AI hardware inlet guideline.")

st.subheader("Outlet temperature over the operating period")
day_range = list(range(0, 91, 3))
temps = [
    run_point(L, D, z, v, n_pipes, k_soil, T_amb, T_soil0, Q_load, d)["T_out"]
    for d in day_range
]
chart_df = pd.DataFrame({"Operating day": day_range, "Outlet air temp (C)": temps})
st.line_chart(chart_df.set_index("Operating day"))

with st.expander("Model assumptions"):
    st.markdown(
        """
- Dittus-Boelter correlation for convective heat transfer inside the pipe
- Exponential (effectiveness-NTU) model for outlet air temperature against a saturating soil sink
- Blasius friction factor for the Darcy-Weisbach pressure drop
- Soil saturation modeled as an asymptotic temperature rise with a 20-day time constant
- Backup mechanical cooling assumed at COP = 3 for any heat load the EAHE alone cannot remove
- Fan power estimated from the pressure drop at 60% fan efficiency

This is a planning-stage analytical tool, not a substitute for CFD validation
or field testing before construction.
"""
    )

st.download_button(
    "Download current run as CSV",
    data=pd.DataFrame([{
        "Pipe length (m)": L, "Pipe diameter (m)": D, "Burial depth (m)": z,
        "Airflow velocity (m/s)": v, "Parallel pipes": n_pipes,
        "Soil thermal conductivity (W/m.K)": k_soil, "Ambient temp (C)": T_amb,
        "Undisturbed soil temp (C)": T_soil0, "AI heat load (kW)": Q_load,
        "Operating days": days, "Outlet temp (C)": round(r["T_out"], 2),
        "Temp reduction (C)": round(r["reduction"], 2),
        "Cooling capacity (kW)": round(r["cooling_capacity_kw"], 2),
        "Pressure drop (Pa)": round(r["pressure_drop_pa"], 1),
        "Soil temp rise (C)": round(r["soil_rise"], 2),
        "Thermal derating (%)": round(derate, 2),
        "PUE estimate": round(r["pue"], 3),
    }]).to_csv(index=False),
    file_name="eahe_run.csv",
    mime="text/csv",
)
