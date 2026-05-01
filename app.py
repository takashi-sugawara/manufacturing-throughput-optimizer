import streamlit as st
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, TerminationCondition
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import io

# ─── Page Config ─────────────────────────────────────────
st.set_page_config(
    page_title="Manufacturing Throughput Optimizer",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.kpi-card {
    background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%);
    border: 1px solid #3a3a5c;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    margin: 4px;
}
.kpi-label { color: #a0a0c0; font-size: 13px; margin-bottom: 6px; }
.kpi-value { color: #7c6af7; font-size: 36px; font-weight: 700; }
.kpi-sub   { color: #606080; font-size: 12px; margin-top: 4px; }

.story-box {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-left: 4px solid #7c6af7;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 12px 0;
}
.constraint-box {
    background: #12121f;
    border: 1px solid #2a2a4a;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)


# ─── Optimization Model ─────────────────────────────────
def solve_model(T, M, x_max, c3_limit, c4_limit):
    model = pyo.ConcreteModel()
    model.x = pyo.Var(range(1, M+1), range(1, T+1), within=pyo.Integers, bounds=(0, x_max))
    x = model.x

    model.obj = pyo.Objective(
        expr=sum(x[m, t] for m in range(1, M+1) for t in range(1, T+1)),
        sense=pyo.maximize
    )

    model.C1 = pyo.ConstraintList()
    for t in range(1, T+1):
        model.C1.add(expr=x[2, t] - 4*x[3, t] <= 0)

    model.C2 = pyo.ConstraintList()
    for t in range(3, T+1):
        model.C2.add(expr=x[2, t] - 2*x[3, t-2] + x[4, t] >= 1)

    model.C3 = pyo.ConstraintList()
    for t in range(1, T+1):
        model.C3.add(expr=sum(x[m, t] for m in range(1, M+1)) <= c3_limit)

    model.C4 = pyo.ConstraintList()
    for t in range(2, T+1):
        model.C4.add(expr=x[1, t] + x[2, t-1] + x[3, t] + x[4, t] <= c4_limit)

    opt = SolverFactory('cbc')
    if not opt.available():
        msg = (
            "❌ CBC solver not found.\n\n"
            "Install instructions:\n"
            "- Mac/Linux: `conda install -c conda-forge coincbc`\n"
            "- Windows: download prebuilt from https://github.com/coin-or/Cbc"
        )
        return None, msg
    try:
        results = opt.solve(model, tee=False)
    except Exception as e:
        return None, f"Solver error: {e}"

    status = results.solver.termination_condition
    if status != TerminationCondition.optimal:
        return None, f"Infeasible or unbounded (status: {status})"

    df = pd.DataFrame(
        {f"Process {m}": [int(round(pyo.value(x[m, t]))) for t in range(1, T+1)]
         for m in range(1, M+1)},
        index=[f"t={t}" for t in range(1, T+1)]
    )
    total = int(round(pyo.value(model.obj)))
    return df, total


# ─── Sensitivity Analysis ────────────────────────────────
@st.cache_data(ttl=600)
def sensitivity_analysis(T, M, x_max, c3_limit, c4_limit, param_name, param_min, param_max, param_step):
    """Sweep one parameter while holding the other fixed at its current value.
    param_name: 'C4_limit' or 'C3_limit'
    """
    param_range = range(param_min, param_max + 1, param_step)
    results = []
    for val in param_range:
        c4 = val if param_name == "C4_limit" else c4_limit
        c3 = val if param_name == "C3_limit" else c3_limit
        df, total = solve_model(T, M, x_max, c3, c4)
        if df is not None:  # skip infeasible points
            results.append({"Parameter Value": val, "Total Throughput": total})
    return pd.DataFrame(results)


# ─── Sidebar ───────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏭 Parameter Settings")
    st.markdown("---")

    # Presets
    presets = {
        "Custom": None,
        "Small (T=5)": {"T": 5, "x_max": 10, "c3": 30, "c4": 8},
        "Medium (T=10)": {"T": 10, "x_max": 10, "c3": 50, "c4": 10},
        "Large (T=20)": {"T": 20, "x_max": 10, "c3": 100, "c4": 15},
    }
    preset_name = st.selectbox("🎛️ Preset", list(presets.keys()))
    preset = presets[preset_name]
    st.markdown("---")

    T = st.slider("⏱️ Operating Hours T", min_value=5, max_value=20,
                  value=preset["T"] if preset else 10, step=1)
    st.markdown("---")

    st.markdown("**Constraint Parameters**")
    x_max = st.slider("Max output per process (C5)", min_value=5, max_value=20,
                       value=preset["x_max"] if preset else 10)
    c3_limit = st.slider("Line capacity ceiling (C3)", min_value=10, max_value=100,
                          value=preset["c3"] if preset else 50, step=5)
    c4_limit = st.slider("Bottleneck ceiling (C4)", min_value=5, max_value=30,
                          value=preset["c4"] if preset else 10, step=1)
    st.markdown("---")

    run_btn = st.button("▶ Run Optimization", type="primary", width="stretch")

M = 4  # number of processes (fixed)
current_params = {"T": T, "M": M, "x_max": x_max, "c3": c3_limit, "c4": c4_limit}

if run_btn or "result_df" not in st.session_state:
    import time
    with st.spinner("Solving with CBC..."):
        _t0 = time.time()
        df, total = solve_model(T, M, x_max, c3_limit, c4_limit)
        _elapsed = time.time() - _t0
    if df is None:
        st.error(f"⚠️ Solver failed: {total}")
        st.info("💡 Try adjusting parameters or reinstalling the CBC solver.")
        st.stop()
    st.session_state["elapsed"] = _elapsed
    st.session_state["result_df"] = df
    st.session_state["total"] = total
    st.session_state["params"] = current_params

df: pd.DataFrame = st.session_state["result_df"]
total: int = st.session_state["total"]
params = st.session_state["params"]

# Warn if parameters have changed since last solve
if current_params != params:
    col_warn, col_btn = st.columns([3, 1])
    with col_warn:
        st.warning("⚠️ Parameters have changed. Results shown may be outdated.", icon="🔄")
    with col_btn:
        if st.button("🔄 Re-run Now", key="forced_rerun"):
            st.session_state.clear()
            st.rerun()

elapsed = st.session_state.get("elapsed", None)
if elapsed is not None:
    st.caption(f"⏱️ Last solve time: {elapsed:.2f}s")


# ─── Main Area ───────────────────────────────────────────────────────────────
st.title("🏭 Manufacturing Process Throughput Optimizer")
st.markdown("Solves a multi-stage, time-dependent Integer Linear Program with the CBC solver.")

tab1, tab2, tab3, tab4 = st.tabs(["📖 Story & Formulation", "📊 Results", "📈 Sensitivity Analysis", "🔧 Model Details"])


# ════════════════════════════════════════════════════════
# TAB 1: Story & Formulation
# ════════════════════════════════════════════════════════
with tab1:
    st.header("Factory Manufacturing Story")

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
<div class="story-box">
<b>🏭 Scenario</b><br>
A factory has four manufacturing processes (machines).<br>
Each process handles a certain number of units every hour.<br><br>
<b>Goal:</b> Maximize total throughput over T hours<br>
while respecting inter-process dependencies and lead-time constraints.
</div>
""", unsafe_allow_html=True)

        st.markdown("#### Process Roles")
        roles = {
            "Process 1 (x₁)": "Finishing & packaging. Fewest constraints, most flexible.",
            "Process 2 (x₂)": "Intermediate assembly. Depends on Process 3 supply (C1).",
            "Process 3 (x₃)": "Raw-material refining. Supply source for downstream. 2-hour lead time.",
            "Process 4 (x₄)": "Final inspection & shipping. Tied to Process 3 output (C2).",
        }
        for k, v in roles.items():
            st.markdown(f"""
<div class="constraint-box">
<b>{k}</b><br>
<span style="color:#a0a0c0">{v}</span>
</div>
""", unsafe_allow_html=True)

    with col2:
        # 工程フロー図
        fig_flow = go.Figure()
        nodes = ["Process 3\n(Raw Mat.)", "Process 2\n(Assembly)", "Process 1\n(Finishing)", "Process 4\n(Shipping)"]
        y_pos = [0.5, 0.5, 0.8, 0.2]
        x_pos = [0.1, 0.55, 0.9, 0.9]
        colors = ["#f59e0b", "#7c6af7", "#10b981", "#3b82f6"]

        for i, (node, xp, yp, c) in enumerate(zip(nodes, x_pos, y_pos, colors)):
            fig_flow.add_annotation(
                x=xp, y=yp, text=node.replace("\n", "<br>"),
                showarrow=False, font=dict(size=12, color="white"),
                bgcolor=c, bordercolor=c, borderpad=8,
                borderwidth=2, xref="paper", yref="paper"
            )

        # 矢印（shape で描画）
        arrow_shapes = [
            dict(type="line", x0=0.18, y0=0.5, x1=0.44, y1=0.5,
                 xref="paper", yref="paper",
                 line=dict(color="#7c6af7", width=2)),
            dict(type="line", x0=0.65, y0=0.55, x1=0.81, y1=0.75,
                 xref="paper", yref="paper",
                 line=dict(color="#7c6af7", width=2)),
            dict(type="line", x0=0.2, y0=0.45, x1=0.8, y1=0.2,
                 xref="paper", yref="paper",
                 line=dict(color="#7c6af7", width=2)),
        ]
        fig_flow.update_layout(
            shapes=arrow_shapes,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor="#12121f", paper_bgcolor="#12121f",
            margin=dict(l=0, r=0, t=0, b=0),
            height=250
        )
        st.plotly_chart(fig_flow, width="stretch")

    st.markdown("---")
    st.header("Mathematical Formulation")

    formulas = [
        ("🎯 Objective Function (Maximize)",
         r"\max \sum_{m=1}^{M} \sum_{t=1}^{T} x_{m,t}",
         "Maximize the total production output across all processes and all time periods."),
        ("C1: Process 2 ≤ 4 × Process 3",
         r"x_{2,t} \leq 4 x_{3,t} \quad \forall t",
         "Process 3 is the supply source. Process 2 throughput is capped at 4× the output of Process 3."),
        ("C2: Lead-Time Constraint (2-hour delivery lag)",
         r"x_{2,t} - 2x_{3,t-2} + x_{4,t} \geq 1 \quad \forall t \geq 3",
         "Output from Process 3 reaches Processes 2 and 4 after a 2-hour delay. Not applied at t=1, 2."),
        ("C3: Line-Wide Throughput Ceiling",
         r"\sum_{m=1}^{M} x_{m,t} \leq " + str(params["c3"]) + r" \quad \forall t",
         f"Total output across all processes at each hour must not exceed {params['c3']} units."),
        ("C4: Bottleneck Constraint (1-hour lag)",
         r"x_{1,t} + x_{2,t-1} + x_{3,t} + x_{4,t} \leq " + str(params["c4"]) + r" \quad \forall t \geq 2",
         f"Process 2 references its value from the previous hour (t−1). Upper limit: {params['c4']} units."),
        ("C5: Individual Process Bounds",
         r"0 \leq x_{m,t} \leq " + str(params["x_max"]) + r" \quad \forall m, t \quad (\text{integer})",
         f"Each process output at each hour must be an integer between 0 and {params['x_max']}."),
    ]

    for title, latex_expr, desc in formulas:
        with st.expander(f" {title}", expanded=True):
            st.latex(latex_expr)
            st.caption(desc)


# ════════════════════════════════════════════════════════
# TAB 2: Results
# ════════════════════════════════════════════════════════
with tab2:

    # KPI cards
    total_by_process = df.sum()
    active_hours = (df.sum(axis=1) > 0).sum()
    row_sums = df.sum(axis=1)
    active_rows = row_sums[row_sums > 0]
    if not active_rows.empty:
        bottleneck_t = active_rows.idxmin()
        bottleneck_val = int(active_rows.min())
    else:
        bottleneck_t, bottleneck_val = "N/A", 0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
<div class="kpi-card">
<div class="kpi-label">🏆 Total Throughput</div>
<div class="kpi-value">{total}</div>
<div class="kpi-sub">units over T={params['T']} hours</div>
</div>""", unsafe_allow_html=True)
    with k2:
        avg = total / params["T"]
        st.markdown(f"""
<div class="kpi-card">
<div class="kpi-label">⚡ Avg Hourly Throughput</div>
<div class="kpi-value">{avg:.1f}</div>
<div class="kpi-sub">units / hour</div>
</div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
<div class="kpi-card">
<div class="kpi-label">🔴 Bottleneck Hour</div>
<div class="kpi-value">{bottleneck_t}</div>
<div class="kpi-sub">Min output (active hrs): {bottleneck_val} units</div>
</div>""", unsafe_allow_html=True)
    with k4:
        top_process = total_by_process.idxmax()
        st.markdown(f"""
<div class="kpi-card">
<div class="kpi-label">🥇 Top Contributing Process</div>
<div class="kpi-value">{top_process}</div>
<div class="kpi-sub">Total: {int(total_by_process.max())} units</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("📊 Output by Process & Hour (Stacked Bar)")
        process_colors = ["#818cf8", "#a78bfa", "#f59e0b", "#38bdf8"]
        fig_bar = go.Figure()
        for i, col in enumerate(df.columns):
            fig_bar.add_trace(go.Bar(
                name=col, x=df.index, y=df[col],
                marker_color=process_colors[i],
                text=df[col].where(df[col] > 0),
                textposition="inside",
                textfont=dict(color="white", size=11),
            ))
        fig_bar.update_layout(
            barmode="stack",
            plot_bgcolor="#12121f", paper_bgcolor="#12121f",
            font=dict(color="#c0c0d0"),
            legend=dict(bgcolor="#1e1e2e", bordercolor="#3a3a5c"),
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis=dict(gridcolor="#2a2a3a"),
            yaxis=dict(gridcolor="#2a2a3a", title="Output (units)"),
            height=380,
        )
        st.plotly_chart(fig_bar, width="stretch")

    with col_r:
        st.subheader("🌡️ Utilization Heatmap (Process × Hour)")
        heat_data = df.T
        fig_heat = px.imshow(
            heat_data,
            color_continuous_scale="Viridis",
            text_auto=True,
            aspect="auto",
            labels=dict(x="Hour", y="Process", color="Output"),
        )
        fig_heat.update_traces(textfont=dict(size=12))
        fig_heat.update_layout(
            plot_bgcolor="#12121f", paper_bgcolor="#12121f",
            font=dict(color="#c0c0d0"),
            margin=dict(l=10, r=10, t=20, b=10),
            height=380,
            coloraxis_colorbar=dict(tickfont=dict(color="#c0c0d0")),
        )
        st.plotly_chart(fig_heat, width="stretch")

    st.subheader("📋 Optimal Solution Table")
    display_df = df.copy()
    display_df["Hour Total"] = display_df.sum(axis=1)
    total_row = display_df.sum().rename("Process Total")
    display_df = pd.concat([display_df, total_row.to_frame().T])
    st.dataframe(
        display_df.style.background_gradient(cmap="Purples", axis=None).format("{:.0f}"),
        width="stretch",
    )

    csv_buf = io.StringIO()
    display_df.to_csv(csv_buf, encoding="utf-8-sig")
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv_buf.getvalue().encode("utf-8-sig"),
        file_name="optimization_result.csv",
        mime="text/csv",
        key="csv_download_tab2",
    )

    st.markdown("---")
    st.subheader("🧠 Auto-Commentary")
    max_t = df.sum(axis=1).idxmax()
    min_t = df.sum(axis=1).idxmin()
    st.info(f"""
**Peak hour: {max_t}** ({int(df.sum(axis=1).max())} units)  
**Lowest-output hour: {min_t}** ({int(df.sum(axis=1).min())} units)  

Due to the C2 lead-time constraint (2 hours), Process 2 is zero at t=1 and t=2.
Output from Process 3 at t=1 arrives at t=3, after which production ramps up.
The C4 bottleneck propagates to subsequent hours, shaping the throughput pattern.
    """)
# ════════════════════════════════════════════════════════
# TAB 3: Sensitivity Analysis
# ════════════════════════════════════════════════════════
with tab3:
    st.header("📈 Parameter Sensitivity Analysis")
    st.markdown("Visualize how total throughput changes as constraint parameters vary.")

    # Run both C4 and C3 sweeps at once so switching the radio never resets the chart
    sens_ready = ("sens_df_C4_limit" in st.session_state and
                  "sens_df_C3_limit" in st.session_state)

    if st.button("▶ Run Sensitivity Analysis (C4 & C3)", key="sens_run_btn", type="primary"):
        with st.spinner("Running sensitivity analysis — computing all C4 & C3 combinations..."):
            df_c4 = sensitivity_analysis(
                params["T"], M, params["x_max"], params["c3"], params["c4"],
                "C4_limit", 5, 30, 1
            )
            df_c3 = sensitivity_analysis(
                params["T"], M, params["x_max"], params["c3"], params["c4"],
                "C3_limit", 10, 100, 5
            )
        st.session_state["sens_df_C4_limit"] = df_c4
        st.session_state["sens_df_C3_limit"] = df_c3
        st.rerun()

    if not sens_ready:
        st.info("💡 Press **▶ Run Sensitivity Analysis** above to compute both C4 and C3 sweeps at once.")
    else:
        # Radio only switches the displayed chart — no recomputation
        sens_param = st.radio(
            "Display parameter",
            ["C4 Ceiling (Bottleneck)", "C3 Ceiling (Line-wide)"],
            horizontal=True,
            key="sens_param_radio",
        )

        if sens_param == "C4 Ceiling (Bottleneck)":
            sens_df = st.session_state["sens_df_C4_limit"]
            param_label = "C4 Ceiling (Bottleneck Constraint)"
            current_val = params["c4"]
        else:
            sens_df = st.session_state["sens_df_C3_limit"]
            param_label = "C3 Ceiling (Line-wide Constraint)"
            current_val = params["c3"]

        fig_sens = go.Figure()
        fig_sens.add_trace(go.Scatter(
            x=sens_df["Parameter Value"], y=sens_df["Total Throughput"],
            mode="lines+markers",
            line=dict(color="#7c6af7", width=3),
            marker=dict(size=7, color="#7c6af7"),
            name="Total Throughput",
        ))
        fig_sens.add_vline(
            x=current_val, line_dash="dash", line_color="#f59e0b",
            annotation_text=f"Current: {current_val}",
            annotation_font_color="#f59e0b",
        )
        fig_sens.update_layout(
            plot_bgcolor="#12121f", paper_bgcolor="#12121f",
            font=dict(color="#c0c0d0"),
            xaxis=dict(title=param_label, gridcolor="#2a2a3a"),
            yaxis=dict(title="Total Throughput (units)", gridcolor="#2a2a3a"),
            margin=dict(l=10, r=10, t=20, b=10),
            height=420,
        )
        st.plotly_chart(fig_sens, width="stretch")

        # Detect the largest single-step improvement
        if len(sens_df) > 1:
            diff = sens_df["Total Throughput"].diff()
            max_diff_idx = diff.idxmax()
            max_improvement = diff.max()
            if max_improvement > 0:
                suggested_val = int(sens_df.loc[max_diff_idx, "Parameter Value"])
                st.success(
                    f"📍 **Best improvement point**: Setting {param_label} to "
                    f"**{suggested_val}** increases throughput by "
                    f"**{int(max_improvement)} units** (largest single-step gain)."
                )



# ════════════════════════════════════════════════════════
# TAB 4: Model Details
# ════════════════════════════════════════════════════════
with tab4:
    st.header("🔧 Model Details (Expanded Constraints)")
    st.markdown("A full listing of every constraint Pyomo generates internally. Useful for learning and debugging.")

    st.subheader("Variable Definition")
    st.markdown(f"""
| Symbol | Description | Domain |
|---|---|---|
| `x[m,t]` | Units processed by process m at hour t | Integer, 0 ≤ x ≤ {params['x_max']} |
| m | Process index | 1, 2, 3, 4 |
| t | Time period (hour) | 1, 2, ..., {params['T']} |
    """)

    st.subheader("Expanded Constraint List")

    constraint_data = []

    for t in range(1, params["T"]+1):
        constraint_data.append({
            "Constraint": "C1", "Hour": f"t={t}",
            "Expression": f"x[2,{t}] - 4·x[3,{t}] ≤ 0",
            "Meaning": "Process 2 ≤ 4 × Process 3"
        })
    for t in range(3, params["T"]+1):
        constraint_data.append({
            "Constraint": "C2", "Hour": f"t={t}",
            "Expression": f"x[2,{t}] - 2·x[3,{t-2}] + x[4,{t}] ≥ 1",
            "Meaning": f"Lead-time (references output at t-2={t-2})"
        })
    for t in range(1, params["T"]+1):
        constraint_data.append({
            "Constraint": "C3", "Hour": f"t={t}",
            "Expression": f"Σm x[m,{t}] ≤ {params['c3']}",
            "Meaning": "Line-wide total ceiling"
        })
    for t in range(2, params["T"]+1):
        constraint_data.append({
            "Constraint": "C4", "Hour": f"t={t}",
            "Expression": f"x[1,{t}] + x[2,{t-1}] + x[3,{t}] + x[4,{t}] ≤ {params['c4']}",
            "Meaning": f"Bottleneck (x[2] references t-1={t-1})"
        })

    c_df = pd.DataFrame(constraint_data)
    filter_c = st.multiselect("Filter by constraint", ["C1", "C2", "C3", "C4"], default=["C1", "C2"])
    st.dataframe(c_df[c_df["Constraint"].isin(filter_c)], width="stretch", height=400)

    st.markdown("---")
    st.subheader("Component Declaration Count")
    st.markdown(f"""
| Component | Name | Size (expanded) |
|---|---|---|
| Var | `model.x` | {M}×{params['T']} = **{M*params['T']} variables** |
| Objective | `model.obj` | **1** |
| ConstraintList | `model.C1` | **{params['T']}** |
| ConstraintList | `model.C2` | **{params['T']-2}** |
| ConstraintList | `model.C3` | **{params['T']}** |
| ConstraintList | `model.C4` | **{params['T']-1}** |
| Bounds (C5) | `bounds=(0, {params['x_max']})` | Built into variable declaration |
| **Total** | 6 Declarations | **{params['T'] + (params['T']-2) + params['T'] + (params['T']-1)} constraints** |
    """)
