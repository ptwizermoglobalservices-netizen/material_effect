import math
import matplotlib.pyplot as plt
import numpy as np
import openseespy.opensees as ops
import streamlit as st

st.set_page_config(page_title="Material Mechanics & Pushover Lab", layout="wide")
st.title("Constitutive Mechanics & Frame Pushover Sensitivity Lab")

# ==============================================================================
# SIDEBAR CONTROLS: ALL REQUESTED PARAMETERS
# ==============================================================================

# --- 1. Concrete02 ---
st.sidebar.markdown("### 1. Concrete (`Concrete02`)")
fck = st.sidebar.slider("fck: Peak compressive strength (MPa)", 15.0, 80.0, 40.0, 2.5)
epsc0 = st.sidebar.slider("epsc0: Strain at peak strength", 0.001, 0.006, 0.003, 0.0005, format="%.4f")
fcu = st.sidebar.slider("fcu: Residual strength (MPa)", 0.0, 30.0, 10.0, 1.0)
epscu = st.sidebar.slider("epscu: Ultimate strain", 0.005, 0.08, 0.04, 0.005, format="%.3f")
Lambda = st.sidebar.slider("Lambda: Ratio of unloading slope to initial E0", 0.05, 0.50, 0.10, 0.05)
ft = st.sidebar.slider("ft: Tensile strength (MPa)", 0.0, 5.0, 0.0, 0.2)
Ets = st.sidebar.slider("Ets: Tension softening modulus (MPa)", 0.0, 10000.0, 0.0, 500.0)

# --- 2. Steel02 ---
st.sidebar.markdown("### 2. Rebar (`Steel02`)")
fy = st.sidebar.slider("fy: Yield strength (MPa)", 300.0, 650.0, 550.0, 10.0)
E0 = st.sidebar.slider("E0: Modulus of Elasticity (MPa)", 180000.0, 230000.0, 210000.0, 5000.0)
b = st.sidebar.slider("b: Strain-hardening ratio", 0.0005, 0.05, 0.001, 0.0005, format="%.4f")
R0 = st.sidebar.slider("R0: Elastic-to-plastic transition exponent", 10.0, 30.0, 15.0, 1.0)
cR1 = st.sidebar.slider("cR1: Curvature degradation parameter", 0.70, 0.99, 0.925, 0.025)
cR2 = st.sidebar.slider("cR2: Curvature degradation parameter", 0.05, 0.35, 0.15, 0.025)

# --- 3. Hysteretic Brace ---
st.sidebar.markdown("### 3. Brace (`Hysteretic`)")
px = st.sidebar.slider("px: Pinching limit along displacement axis", 0.1, 1.0, 0.6, 0.05)
py = st.sidebar.slider("py: Pinching limit along force axis", 0.1, 1.0, 0.4, 0.05)
damage1 = st.sidebar.slider("damage1: Ductility damage parameter", 0.0, 0.5, 0.0, 0.05)
damage2 = st.sidebar.slider("damage2: Energy damage parameter", 0.0, 0.5, 0.0, 0.05)
beta = st.sidebar.slider("beta: Stiffness degradation parameter", 0.0, 0.5, 0.0, 0.05)

# --- 4. Pushover Settings ---
st.sidebar.markdown("### 4. Pushover Displacement")
target_disp = st.sidebar.slider("Max Pushover Disp (mm)", 20.0, 150.0, 80.0, 10.0)
dU = 0.5


# ==============================================================================
# SIMULATION WORKFLOWS
# ==============================================================================

def run_pushover():
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)

    L, H = 5000.0, 4000.0
    Bc, Hc = 300.0, 200.0
    Hb, Bb = 300.0, 500.0

    ops.node(1, 0.0, 0.0)
    ops.node(2, 0.0, H)
    ops.node(3, L, H)
    ops.node(4, L, 0.0)

    ops.fix(1, 1, 1, 1)
    ops.fix(4, 1, 1, 1)

    # 1. Concrete02
    ops.uniaxialMaterial(
        "Concrete02", 1, -abs(fck), -abs(epsc0), -abs(fcu), -abs(epscu), Lambda, ft, Ets
    )

    # 2. Steel02
    ops.uniaxialMaterial("Steel02", 3, fy, E0, b, R0, cR1, cR2)

    # 3. Sections
    cover = 10.0
    col_reinf_dia = 12.0
    As_col = math.pi * col_reinf_dia ** 2 / 4.0
    y_col, z_col = Hc / 2.0, Bc / 2.0

    ops.section("Fiber", 1)
    ops.patch("rect", 1, 10, 10, -y_col, -z_col, y_col, z_col)
    ops.layer("straight", 3, 3, As_col, -y_col + cover, -z_col + cover, -y_col + cover, z_col - cover)
    ops.layer("straight", 3, 3, As_col, y_col - cover, -z_col + cover, y_col - cover, z_col - cover)

    beam_reinf_dia = 12.0
    As_beam = math.pi * beam_reinf_dia ** 2 / 4.0
    y_beam, z_beam = Hb / 2.0, Bb / 2.0

    ops.section("Fiber", 2)
    ops.patch("rect", 1, 10, 10, -y_beam, -z_beam, y_beam, z_beam)
    ops.layer("straight", 3, 3, As_beam, -y_beam + cover, -z_beam + cover, -y_beam + cover, z_beam - cover)
    ops.layer("straight", 3, 3, As_beam, y_beam - cover, -z_beam + cover, y_beam - cover, z_beam - cover)

    # 4. Hysteretic Brace
    fy_b = 200.0
    e1, e2, e3 = 0.002, 0.005, 0.010
    ops.uniaxialMaterial(
        "Hysteretic", 4,
        fy_b, e1, fy_b * 1.1, e2, fy_b * 0.2, e3,
        -fy_b, -e1, -fy_b * 1.1, -e2, -fy_b * 0.2, -e3,
        px, py, damage1, damage2, beta
    )
    A_brace = 1000.0
    ops.element("corotTruss", 4, 2, 4, A_brace, 4)

    # 5. Frame Elements
    ops.geomTransf("Linear", 1)
    ops.element("nonlinearBeamColumn", 1, 1, 2, 5, 1, 1)
    ops.element("nonlinearBeamColumn", 2, 2, 3, 5, 2, 1)
    ops.element("nonlinearBeamColumn", 3, 3, 4, 5, 1, 1)

    # 6. Gravity Load
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.eleLoad("-ele", 2, "-type", "beamUniform", -20.0)

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
        return None, None, None, "Gravity phase failed."

    ops.loadConst("-time", 0.0)
    ops.wipeAnalysis()

    # 7. Pushover Analysis
    ops.timeSeries("Linear", 2)
    ops.pattern("Plain", 2, 2)
    ops.load(2, 1.0, 0.0, 0.0)

    ops.system("BandGeneral")
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.test("NormDispIncr", 1.0e-4, 300)
    ops.algorithm("Newton")
    ops.integrator("DisplacementControl", 2, 1, dU)
    ops.analysis("Static")

    D_hist, V_hist, N_hist = [0.0], [0.0], [0.0]
    steps = int(target_disp / dU)

    for _ in range(steps):
        ok = ops.analyze(1)
        if ok != 0:
            ops.algorithm("ModifiedNewton")
            ok = ops.analyze(1)
        if ok != 0:
            break

        ops.reactions()
        top_disp = ops.nodeDisp(2, 1)
        base_shear = -(ops.nodeReaction(1, 1) + ops.nodeReaction(4, 1)) / 1e3
        brace_axial = ops.eleResponse(4, "axialForce")[0] / 1e3

        D_hist.append(top_disp)
        V_hist.append(base_shear)
        N_hist.append(brace_axial)

    return D_hist, V_hist, N_hist, "Completed"


def run_single_material_test(mat_type):
    """Subject a single 1D link to cyclic loading to directly visualize material parameters."""
    ops.wipe()
    ops.model("basic", "-ndm", 1, "-ndf", 1)
    ops.node(1, 0.0)
    ops.node(2, 1.0)
    ops.fix(1, 1)

    if mat_type == "Hysteretic":
        fy_b = 200.0
        e1, e2, e3 = 0.002, 0.005, 0.010
        ops.uniaxialMaterial(
            "Hysteretic", 1,
            fy_b, e1, fy_b * 1.1, e2, fy_b * 0.2, e3,
            -fy_b, -e1, -fy_b * 1.1, -e2, -fy_b * 0.2, -e3,
            px, py, damage1, damage2, beta
        )
        strains = np.concatenate([
            np.linspace(0, 0.006, 30),
            np.linspace(0.006, -0.006, 60),
            np.linspace(-0.006, 0.009, 75),
            np.linspace(0.009, -0.009, 90),
            np.linspace(-0.009, 0.0, 45)
        ])
    elif mat_type == "Steel02":
        ops.uniaxialMaterial("Steel02", 1, fy, E0, b, R0, cR1, cR2)
        ey = fy / E0
        strains = np.concatenate([
            np.linspace(0, 3 * ey, 30),
            np.linspace(3 * ey, -3 * ey, 60),
            np.linspace(-3 * ey, 5 * ey, 80),
            np.linspace(5 * ey, 0.0, 50)
        ])
    elif mat_type == "Concrete02":
        ops.uniaxialMaterial(
            "Concrete02", 1, -abs(fck), -abs(epsc0), -abs(fcu), -abs(epscu), Lambda, ft, Ets
        )
        strains = np.concatenate([
            np.linspace(0, -epsc0 * 1.2, 40),
            np.linspace(-epsc0 * 1.2, 0.0005, 30),
            np.linspace(0.0005, -epscu * 0.7, 60),
            np.linspace(-epscu * 0.7, 0.0, 30)
        ])

    ops.element("truss", 1, 1, 2, 1.0, 1)

    strain_out, stress_out = [0.0], [0.0]
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(2, 1.0)

    ops.system("BandGeneral")
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.test("NormDispIncr", 1e-6, 100)
    ops.algorithm("Newton")
    ops.analysis("Static")

    prev_disp = 0.0
    for target in strains:
        dD = target - prev_disp
        ops.integrator("DisplacementControl", 2, 1, dD)
        ops.analyze(1)
        prev_disp = target
        ops.reactions()
        stress = -ops.nodeReaction(1, 1)
        strain_out.append(target)
        stress_out.append(stress)

    return strain_out, stress_out


# ==============================================================================
# UI TABS
# ==============================================================================
tab_frame, tab_material, tab_guide = st.tabs([
    "Global Frame Pushover",
    "Cyclic Constitutive Playground",
    "Parameter Guide & Sensitivity Matrix"
])

with tab_frame:
    st.subheader("Global Pushover & Internal Brace Demand")
    if st.button("Run Frame Pushover", type="primary"):
        with st.spinner("Analyzing Frame..."):
            D, V, N, msg = run_pushover()

        if D is None:
            st.error(msg)
        else:
            fig, ax1 = plt.subplots(figsize=(9, 4.5))
            ax1.plot(D, V, color="royalblue", lw=2.2, label="Base Shear (kN)")
            ax1.set_xlabel("Roof Lateral Displacement (mm)")
            ax1.set_ylabel("Total Base Shear (kN)", color="royalblue")
            ax1.grid(True, linestyle="--", alpha=0.5)

            ax2 = ax1.twinx()
            ax2.plot(D, N, color="crimson", lw=1.8, linestyle="--", label="Brace Axial Force (kN)")
            ax2.set_ylabel("Brace Axial Demand (kN)", color="crimson")

            lines = ax1.get_lines() + ax2.get_lines()
            ax1.legend(lines, [l.get_label() for l in lines], loc="best")
            st.pyplot(fig)

            c1, c2, c3 = st.columns(3)
            c1.metric("Peak Base Shear", f"{max(V):.1f} kN")
            c2.metric("Disp at Peak", f"{D[np.argmax(V)]:.1f} mm")
            c3.metric("Peak Brace Demand", f"{min(N):.1f} kN" if abs(min(N)) > abs(max(N)) else f"{max(N):.1f} kN")

with tab_material:
    st.subheader("Constitutive Law Stress-Strain Response (Cyclic Protocol)")
    st.caption(
        "Inspect how the cyclic unloading, pinching, and curvature parameters shape each material's hysteretic loop.")

    mat_selected = st.radio("Select Material to Test", ["Hysteretic", "Steel02", "Concrete02"], horizontal=True)
    eps_vals, sig_vals = run_single_material_test(mat_selected)

    fig_mat, ax_mat = plt.subplots(figsize=(8, 4))
    ax_mat.plot(eps_vals, sig_vals, color="darkorange" if mat_selected == "Steel02" else (
        "forestgreen" if mat_selected == "Concrete02" else "purple"), lw=2.0)
    ax_mat.set_xlabel("Strain")
    ax_mat.set_ylabel("Stress / Force")
    ax_mat.grid(True, linestyle="--", alpha=0.6)
    ax_mat.axhline(0, color="black", lw=0.8)
    ax_mat.axvline(0, color="black", lw=0.8)
    st.pyplot(fig_mat)

with tab_guide:
    st.markdown("""
| Material | Parameter | Physical Meaning | Effect on Monotonic Pushover | Effect on Cyclic Response |
| :--- | :--- | :--- | :--- | :--- |
| **`Concrete02`** | `fck` | Compressive peak strength | Scales initial peak capacity | Scales compression envelop |
| | `epsc0` | Peak strain | Shifts displacement of peak capacity | Controls initial stiffness $E_c = 2f_c/\epsilon_0$ |
| | `fcu` | Residual strength | Sets post-crushing plateau | Sustains minimum cyclic capacity |
| | `epscu` | Crushing strain | Determines softening slope termination | Controls strain limit before crushing |
| | `Lambda` | Unloading stiffness ratio | **Zero effect** (monotonic only) | Controls degradation of unloading slope |
| | `ft` | Tensile strength | Increases cracking threshold slightly | Shifts tension cracking envelope |
| | `Ets` | Tension softening slope | Prevents sharp convergence cuts at crack | Governs tension stiffness degradation |
| **`Steel02`** | `fy` | Steel yield stress | Sets global yield plateau of the frame | Sets cyclic yield limits |
| | `E0` | Elastic modulus | Modulates elastic frame stiffness | Controls elastic stiffness during reversals |
| | `b` | Strain-hardening ratio | Governs post-yield slope | Controls tangent hardening under large cycles |
| | `R0` | Transition parameter | Governs sharpness from elastic to plastic | Controls initial Bauschinger roundness |
| | `cR1`, `cR2`| Curvature degradation | **Zero effect** (monotonic only) | Flattens elastic-plastic curve with plastic strain |
| **`Hysteretic`** | `px`, `py` | Pinching parameters | **Zero effect** (monotonic only) | Controls slip and crack closing during load reversal |
| | `damage1`, `2`| Ductility & energy damage| **Zero effect** (monotonic only) | Degrades strength and stiffness after plastic excursion |
| | `beta` | Stiffness degradation | **Zero effect** (monotonic only) | Reduces reloading stiffness based on displacement |
""")