import math
import numpy as np
import matplotlib.pyplot as plt
import openseespy.opensees as ops
import streamlit as st

st.set_page_config(
    page_title="OpenSees Material & Gravity Pushover Lab",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for polished dashboard look
st.markdown("""
<style>
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-info {
        background-color: #e0f2fe;
        color: #0369a1;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

if "baseline_results" not in st.session_state:
    st.session_state.baseline_results = None
if "current_results" not in st.session_state:
    st.session_state.current_results = None

st.sidebar.title("🎛️ Model Parameters")

# --- Gravity Beam Load w ---
st.sidebar.markdown("### 0. Gravity Beam Load ($w$)")
w_load = st.sidebar.slider(
    "Beam Uniform Load w (kN/m)",
    min_value=-60.0,
    max_value=0.0,
    value=-20.0,
    step=2.5,
    help="Applied downward uniformly along the roof beam ele 2 during gravity phase."
)

# --- Concrete02 Parameters ---
st.sidebar.markdown("### 1. Concrete (`Concrete02`)")
c_col1, c_col2 = st.sidebar.columns(2)
with c_col1:
    fck = st.slider("fck (MPa)", 15.0, 70.0, 40.0, 2.5, help="Compressive peak strength")
    fcu = st.slider("fcu (MPa)", 0.0, 25.0, 10.0, 1.0, help="Residual crushing strength")
    Lambda = st.slider("Lambda (E_unl/E0)", 0.05, 0.40, 0.10, 0.01, help="Ratio of unloading slope to initial E0")
    Ets = st.slider("Ets (MPa)", 0.0, 10000.0, 0.0, 500.0, help="Tension softening modulus")
with c_col2:
    epsc0 = st.slider("epsc0", 0.0015, 0.0050, 0.0030, 0.0002, format="%.4f", help="Strain at peak strength")
    epscu = st.slider("epscu", 0.010, 0.080, 0.040, 0.005, format="%.3f", help="Ultimate crushing strain")
    ft = st.slider("ft (MPa)", 0.0, 4.5, 0.0, 0.25, help="Tensile strength")

# --- Steel02 Parameters ---
st.sidebar.markdown("### 2. Longitudinal Rebar (`Steel02`)")
s_col1, s_col2 = st.sidebar.columns(2)
with s_col1:
    fy = st.slider("fy (MPa)", 300.0, 650.0, 550.0, 10.0, help="Yield stress")
    b_steel = st.slider("b (Hardening)", 0.0005, 0.05, 0.001, 0.0005, format="%.4f", help="Strain-hardening ratio")
    cR1 = st.slider("cR1", 0.70, 0.99, 0.925, 0.025, help="Curvature transition degradation 1")
with s_col2:
    E0 = st.slider("E0 (MPa)", 180000.0, 230000.0, 210000.0, 5000.0, help="Elastic modulus")
    R0 = st.slider("R0 (Curvature)", 8.0, 25.0, 15.0, 1.0, help="Elastic-to-plastic transition exponent")
    cR2 = st.slider("cR2", 0.05, 0.35, 0.15, 0.025, help="Curvature transition degradation 2")

# --- Hysteretic Brace Parameters ---
st.sidebar.markdown("### 3. Diagonal Brace (`Hysteretic`)")
h_col1, h_col2 = st.sidebar.columns(2)
with h_col1:
    px = st.slider("px (Pinch X)", 0.1, 1.0, 0.6, 0.05, help="Pinching limit along displacement axis")
    damage1 = st.slider("damage1", 0.0, 0.5, 0.0, 0.05, help="Ductility-based damage")
    beta = st.slider("beta", 0.0, 0.5, 0.0, 0.05, help="Degradation based on energy")
with h_col2:
    py = st.slider("py (Pinch Y)", 0.1, 1.0, 0.4, 0.05, help="Pinching limit along force axis")
    damage2 = st.slider("damage2", 0.0, 0.5, 0.0, 0.05, help="Energy-based damage")
    A_brace = st.slider("Area (mm²)", 200.0, 2500.0, 1000.0, 100.0)

# --- Analysis Controls ---
st.sidebar.markdown("### 4. Pushover Controls")
target_disp = st.sidebar.slider("Target Displacement (mm)", 20.0, 150.0, 90.0, 5.0)
dU = st.sidebar.slider("Step size dU (mm)", 0.1, 1.0, 0.5, 0.1)

def run_frame_pushover(
    w_val, fck_v, epsc0_v, fcu_v, epscu_v, Lam_v, ft_v, Ets_v,
    fy_v, E0_v, b_v, R0_v, cR1_v, cR2_v,
    px_v, py_v, dmg1_v, dmg2_v, beta_v, A_br_v,
    t_disp, du_step
):
    """
    Executes 2D Portal Frame Pushover with gravity beam loading w.
    Returns (disp_list, base_shear_list, brace_force_list, status_msg)
    """
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)

    L = 5000.0
    H = 4000.0
    Bc, Hc = 300.0, 200.0
    Hb, Bb = 300.0, 500.0

    ops.node(1, 0.0, 0.0)
    ops.node(2, 0.0, H)
    ops.node(3, L, H)
    ops.node(4, L, 0.0)

    ops.fix(1, 1, 1, 1)
    ops.fix(4, 1, 1, 1)

    # 1. Concrete02 Material
    ops.uniaxialMaterial(
        "Concrete02", 1,
        -abs(fck_v), -abs(epsc0_v), -abs(fcu_v), -abs(epscu_v),
        Lam_v, ft_v, Ets_v
    )

    # 2. Steel02 Material
    ops.uniaxialMaterial("Steel02", 3, fy_v, E0_v, b_v, R0_v, cR1_v, cR2_v)

    # 3. Column Section (Fiber)
    cover = 10.0
    col_reinf_dia = 12.0
    As_col = math.pi * col_reinf_dia**2 / 4.0
    y_col, z_col = Hc / 2.0, Bc / 2.0

    ops.section("Fiber", 1)
    ops.patch("rect", 1, 10, 10, -y_col, -z_col, y_col, z_col)
    ops.layer("straight", 3, 3, As_col, -y_col + cover, -z_col + cover, -y_col + cover, z_col - cover)
    ops.layer("straight", 3, 3, As_col,  y_col - cover, -z_col + cover,  y_col - cover,  z_col - cover)

    # 4. Beam Section (Fiber)
    beam_reinf_dia = 12.0
    As_beam = math.pi * beam_reinf_dia**2 / 4.0
    y_beam, z_beam = Hb / 2.0, Bb / 2.0

    ops.section("Fiber", 2)
    ops.patch("rect", 1, 10, 10, -y_beam, -z_beam, y_beam, z_beam)
    ops.layer("straight", 3, 3, As_beam, -y_beam + cover, -z_beam + cover, -y_beam + cover, z_beam - cover)
    ops.layer("straight", 3, 3, As_beam,  y_beam - cover, -z_beam + cover,  y_beam - cover,  z_beam - cover)

    # 5. Hysteretic Brace Material & Truss Element
    fy_b = 200.0
    e1, e2, e3 = 0.002, 0.005, 0.010
    ops.uniaxialMaterial(
        "Hysteretic", 4,
        fy_b, e1, fy_b * 1.1, e2, fy_b * 0.2, e3,
        -fy_b, -e1, -fy_b * 1.1, -e2, -fy_b * 0.2, -e3,
        px_v, py_v, dmg1_v, dmg2_v, beta_v
    )
    ops.element("corotTruss", 4, 2, 4, A_br_v, 4)

    # 6. Nonlinear Frame Elements
    ops.geomTransf("Linear", 1)
    ops.element("nonlinearBeamColumn", 1, 1, 2, 5, 1, 1)
    ops.element("nonlinearBeamColumn", 2, 2, 3, 5, 2, 1)
    ops.element("nonlinearBeamColumn", 3, 3, 4, 5, 1, 1)

    # 7. Gravity Analysis with parameter w
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    # Apply user defined uniform beam gravity load (N/mm = kN/m)
    ops.eleLoad("-ele", 2, "-type", "beamUniform", w_val)

    ops.system("BandGeneral")
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.test("NormDispIncr", 1.0e-5, 500, 0)
    ops.algorithm("NewtonLineSearch", 0.8)
    ops.integrator("LoadControl", 0.01)
    ops.analysis("Static")

    ok = ops.analyze(100)
    if ok != 0:
        ops.test("NormDispIncr", 1.0e-4, 300)
        ops.algorithm("ModifiedNewton")
        ok = ops.analyze(100)

    if ok != 0:
        return None, None, None, "Gravity phase failed to converge. Check w load or section capacity."

    # Fix gravity loads as constant state
    ops.loadConst("-time", 0.0)
    ops.wipeAnalysis()

    # 8. Pushover Phase
    ops.timeSeries("Linear", 2)
    ops.pattern("Plain", 2, 2)
    ops.load(2, 1.0, 0.0, 0.0)

    ops.system("BandGeneral")
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.test("NormDispIncr", 1.0e-4, 300)
    ops.algorithm("Newton")
    ops.integrator("DisplacementControl", 2, 1, du_step)
    ops.analysis("Static")

    D_hist = [0.0]
    V_hist = [0.0]
    N_hist = [0.0]
    num_steps = int(t_disp / du_step)

    for _ in range(num_steps):
        ok = ops.analyze(1)
        if ok != 0:
            ops.algorithm("ModifiedNewton")
            ok = ops.analyze(1)
        if ok != 0:
            ops.algorithm("KrylovNewton")
            ok = ops.analyze(1)
        if ok != 0:
            break

        ops.reactions()
        roof_disp = ops.nodeDisp(2, 1)
        base_shear = -(ops.nodeReaction(1, 1) + ops.nodeReaction(4, 1)) / 1e3
        
        # CorotTruss axial force (N -> kN)
        brace_axial = -(ops.eleResponse(4, "axialForce")[0] / 1e3)

        D_hist.append(roof_disp)
        V_hist.append(base_shear)
        N_hist.append(brace_axial)

    return D_hist, V_hist, N_hist, "Success"

def run_cyclic_material_test(material_name):
    """
    Simulates a 1D unit truss under cyclic displacement protocols
    to demonstrate cyclic deterioration and unloading laws.
    """
    ops.wipe()
    ops.model("basic", "-ndm", 1, "-ndf", 1)
    ops.node(1, 0.0)
    ops.node(2, 1.0)
    ops.fix(1, 1)

    if material_name == "Hysteretic Brace":
        fy_b = 200.0
        e1, e2, e3 = 0.002, 0.005, 0.010
        ops.uniaxialMaterial(
            "Hysteretic", 10,
            fy_b, e1, fy_b * 1.1, e2, fy_b * 0.2, e3,
            -fy_b, -e1, -fy_b * 1.1, -e2, -fy_b * 0.2, -e3,
            px, py, damage1, damage2, beta
        )
        strains = np.concatenate([
            np.linspace(0, 0.006, 35),
            np.linspace(0.006, -0.006, 70),
            np.linspace(-0.006, 0.009, 85),
            np.linspace(0.009, -0.009, 100),
            np.linspace(-0.009, 0.0, 50)
        ])
    elif material_name == "Steel02 Rebar":
        ops.uniaxialMaterial("Steel02", 10, fy, E0, b_steel, R0, cR1, cR2)
        ey = fy / E0
        strains = np.concatenate([
            np.linspace(0, 3.0 * ey, 35),
            np.linspace(3.0 * ey, -2.5 * ey, 60),
            np.linspace(-2.5 * ey, 5.0 * ey, 85),
            np.linspace(5.0 * ey, -4.0 * ey, 90),
            np.linspace(-4.0 * ey, 0.0, 40)
        ])
    else:  # Concrete02
        ops.uniaxialMaterial(
            "Concrete02", 10,
            -abs(fck), -abs(epsc0), -abs(fcu), -abs(epscu),
            Lambda, ft, Ets
        )
        strains = np.concatenate([
            np.linspace(0, -epsc0 * 1.1, 40),
            np.linspace(-epsc0 * 1.1, 0.0003, 35),
            np.linspace(0.0003, -epscu * 0.65, 75),
            np.linspace(-epscu * 0.65, 0.0001, 35),
            np.linspace(0.0001, -epscu * 0.95, 45)
        ])

    ops.element("truss", 1, 1, 2, 1.0, 10)

    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(2, 1.0)

    ops.system("BandGeneral")
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.test("NormDispIncr", 1.0e-6, 150)
    ops.algorithm("Newton")
    ops.analysis("Static")

    strain_trace = [0.0]
    stress_trace = [0.0]
    prev_d = 0.0

    for d in strains:
        incr = d - prev_d
        ops.integrator("DisplacementControl", 2, 1, incr)
        ops.analyze(1)
        prev_d = d
        ops.reactions()
        stress = -ops.nodeReaction(1, 1)
        strain_trace.append(d)
        stress_trace.append(stress)

    return strain_trace, stress_trace

tab_pushover, tab_playground, tab_theory = st.tabs([
    "📈 Frame Pushover & Gravity Analysis",
    "🔬 Cyclic Material Law Playground",
    "📚 Engineering Sensitivity Guide (w & Parameters)"
])

with tab_pushover:
    st.subheader("Global Pushover & Diagonal Demand")
    st.caption("Investigate how beam gravity load $w$, reinforcement, and bracing stiffness interact to govern the global capacity.")

    b_col1, b_col2, b_col3 = st.columns([1.5, 1.5, 3])
    with b_col1:
        run_sim = st.button("🚀 Run Frame Pushover", type="primary", use_container_width=True)
    with b_col2:
        set_base = st.button("📌 Store Current as Baseline", use_container_width=True)
    with b_col3:
        clear_base = st.button("🗑️ Clear Baseline", use_container_width=True)

    if set_base and st.session_state.current_results:
        st.session_state.baseline_results = st.session_state.current_results
        st.success("Current run locked as Baseline.")

    if clear_base:
        st.session_state.baseline_results = None
        st.info("Baseline removed.")

    if run_sim:
        with st.spinner("Computing nonlinear static equilibrium across load steps..."):
            D, V, N, status = run_frame_pushover(
                w_load, fck, epsc0, fcu, epscu, Lambda, ft, Ets,
                fy, E0, b_steel, R0, cR1, cR2,
                px, py, damage1, damage2, beta, A_brace,
                target_disp, dU
            )
        if D is None:
            st.error(status)
        else:
            st.session_state.current_results = {
                "D": D, "V": V, "N": N, "w": w_load,
                "fck": fck, "fy": fy, "A_brace": A_brace
            }

    if st.session_state.current_results is not None:
        curr = st.session_state.current_results
        base = st.session_state.baseline_results

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # 1. Base Shear vs Roof Disp
        ax1.plot(curr["D"], curr["V"], color="#1d4ed8", lw=2.4, label=f"Current (w={curr['w']} kN/m)")
        if base is not None:
            ax1.plot(base["D"], base["V"], color="#64748b", lw=1.8, linestyle="--", label=f"Baseline (w={base['w']} kN/m)")
        ax1.set_title("Global Pushover Curve (Base Shear vs Roof Drift)", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Roof Lateral Displacement (mm)")
        ax1.set_ylabel("Total Base Shear (kN)")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="lower right")

        # 2. Brace Axial Force vs Roof Disp
        ax2.plot(curr["D"], curr["N"], color="#dc2626", lw=2.2, label="Current Brace Axial Force")
        if base is not None:
            ax2.plot(base["D"], base["N"], color="#94a3b8", lw=1.8, linestyle="--", label="Baseline Brace Force")
        ax2.set_title("Diagonal Brace Axial Force Demand", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Roof Lateral Displacement (mm)")
        ax2.set_ylabel("Brace Axial Force (kN, Compression < 0)")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="best")

        plt.tight_layout()
        st.pyplot(fig)

        # Performance Metrics
        m1, m2, m3, m4 = st.columns(4)
        peak_V = max(curr["V"])
        disp_peak = curr["D"][np.argmax(curr["V"])]
        max_brace_force = min(curr["N"])  # Compression peak

        with m1:
            st.metric("Peak Base Shear", f"{peak_V:.1f} kN", delta=f"{peak_V - max(base['V']):.1f} kN" if base else None)
        with m2:
            st.metric("Drift at Peak Shear", f"{disp_peak:.1f} mm")
        with m3:
            st.metric("Peak Brace Compression", f"{max_brace_force:.1f} kN")
        with m4:
            st.metric("Gravity Beam Load w", f"{curr['w']} kN/m")

with tab_playground:
    st.subheader("🔬 Constitutive Mechanics Lab (Cyclic Protocols)")
    st.markdown("""
    Explore how the advanced degradation and transition parameters (`px`, `py`, `damage1`, `damage2`, `beta`, `R0`, `cR1`, `cR2`, `Lambda`) 
    actively govern hysteretic damping, pinch shapes, and stiffness decay.
    """)

    sel_mat = st.radio(
        "Select Material Model to Test Cyclically:",
        ["Hysteretic Brace", "Steel02 Rebar", "Concrete02"],
        horizontal=True
    )

    with st.spinner("Executing cyclic strain history..."):
        eps_vec, sig_vec = run_cyclic_material_test(sel_mat)

    fig_mat, ax_mat = plt.subplots(figsize=(9, 4.8))
    
    if sel_mat == "Hysteretic Brace":
        ax_mat.plot(eps_vec, sig_vec, color="#9333ea", lw=2.0, label="Hysteretic Loop")
        ax_mat.set_ylabel("Axial Stress / Force")
        ax_mat.set_title(f"Hysteretic: px={px}, py={py}, dmg1={damage1}, dmg2={damage2}, beta={beta}", fontsize=11)
    elif sel_mat == "Steel02 Rebar":
        ax_mat.plot(eps_vec, sig_vec, color="#ea580c", lw=2.0, label="Steel02 Cyclic Curve")
        ax_mat.set_ylabel("Stress (MPa)")
        ax_mat.set_title(f"Steel02: fy={fy} MPa, R0={R0}, cR1={cR1}, cR2={cR2}, b={b_steel}", fontsize=11)
    else:
        ax_mat.plot(eps_vec, sig_vec, color="#15803d", lw=2.0, label="Concrete02 Cyclic Curve")
        ax_mat.set_ylabel("Stress (MPa, Compressive < 0)")
        ax_mat.set_title(f"Concrete02: fck={fck} MPa, epsc0={epsc0}, fcu={fcu}, epscu={epscu}, Lambda={Lambda}", fontsize=11)

    ax_mat.set_xlabel("Strain ε")
    ax_mat.axhline(0, color="black", lw=0.8, alpha=0.7)
    ax_mat.axvline(0, color="black", lw=0.8, alpha=0.7)
    ax_mat.grid(True, linestyle="--", alpha=0.6)
    ax_mat.legend(loc="best")
    plt.tight_layout()
    st.pyplot(fig_mat)

with tab_theory:
    st.subheader("📚 Structural Parameter Sensitivity Matrix")
    st.markdown("""
    This table breaks down how each argument influences both a **monotonic pushover** and **cyclic seismic loading**:
    """)

    st.markdown("""
| Model | Parameter | Physical Meaning | Effect on Monotonic Pushover | Effect on Cyclic Response |
| :--- | :--- | :--- | :--- | :--- |
| **Gravity** | **`w`** | Uniform vertical load on beam | **High**: Pre-compresses columns, creates initial sagging beam moment, accelerates column plastic hinging via P-M interaction and reduces lateral capacity. | Pre-loads beam plastic hinges, causing asymmetric cyclic loops. |
| **`Concrete02`**| `fck` | Peak compressive strength | Scales column axial and flexural capacity directly. | Expands bounding compression envelope. |
| | `epsc0` | Strain at peak strength | Alters initial flexural stiffness ($E_c \approx 2f_{ck}/\epsilon_0$). | Governs strain at start of compression softening. |
| | `fcu` | Residual compressive strength | Sustains residual post-peak capacity of columns. | Sets minimum residual loop size at large drifts. |
| | `epscu` | Ultimate crushing strain | Controls ductility before abrupt column strength loss. | Dictates core concrete spalling limit. |
| | `Lambda` | Unloading stiffness ratio | **Zero effect** (only active during unload). | Prevents numerical divergence during crack closing. |
| | `ft` | Tensile cracking strength | Modulates initial cracking elastic stiffness. | Shifts crack opening thresholds. |
| | `Ets` | Tension softening modulus | Softens tensile crack opening, prevents shocks. | Controls stiffness degradation in tension. |
| **`Steel02`** | `fy` | Steel yield stress | Dominates lateral capacity and yield displacement. | Sets cyclic yield limits in tension/compression. |
| | `E0` | Modulus of elasticity | Governs initial elastic frame stiffness. | Governs reload/unload elastic slope. |
| | `b` | Strain-hardening ratio | Dictates post-yield positive slope of the curve. | Controls tangent hardening under large cycles. |
| | `R0` | Transition parameter | Governs roundness of elastic-to-plastic knee. | Governs Bauschinger curvature roundness. |
| | `cR1`, `cR2`| Curvature degradation | **Zero effect** (monotonic only). | Flattens Menegotto-Pinto curve under plastic cycling. |
| **`Hysteretic`**| `px`, `py` | Displacement & force pinching | **Zero effect** (monotonic envelope governs). | Controls pinched waist size (crack re-closure / slip). |
| | `damage1`, `2`| Ductility & energy damage | **Zero effect** (monotonic envelope governs). | Shrinks backbone envelope after each yield cycle. |
| | `beta` | Stiffness degradation | **Zero effect** (monotonic envelope governs). | Progressively degrades reloading stiffness. |
""")

    st.info("""
    💡 **Key Insight on Monotonic Pushover Mechanics:**
    In a monotonic push, displacement increases strictly in one direction. Parameters governing **unloading, pinching, and cyclic damage** (`px`, `py`, `damage1`, `damage2`, `beta`, `Lambda`, `cR1`, `cR2`) remain dormant in monotonic curves. To evaluate them, use the **Cyclic Material Law Playground** tab.
    """)
