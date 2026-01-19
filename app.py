import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from generator import SimulationGenerator
from stats import FrequentistTest, BayesianTest

# Page Configuration
st.set_page_config(
    page_title="Experimentation Decision Engine",
    page_icon="🧪",
    layout="wide"
)

# Custom CSS for "Premium" feel
st.markdown("""
<style>
    .block-container {padding-top: 1rem; padding-bottom: 2rem;}
    .reportview-container { background: #0e1117; }
    div[data-testid="stMetricValue"] { font-size: 2rem; }
    h1, h2, h3 { color: #f0f2f6; }
    .stAlert { font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Application Title
st.title("🧪 The Experimentation Decision Engine (V1)")
st.markdown("*Simulating hostile environments to train decision making.*")

# --- SIDEBAR: INPUTS ---
with st.sidebar:
    st.header("🎛️ Experiment Setup")
    
    # Section 1: Parameters
    st.subheader("1. parameters")
    n_users = st.slider("Total Users", min_value=1000, max_value=50000, value=10000, step=1000)
    baseline_rate = st.slider("Baseline Conversion Rate (%)", 1.0, 50.0, 12.0, 0.5) / 100.0
    lift_percent = st.slider("Expected Lift (%)", -20.0, 50.0, 10.0, 1.0)
    lift = lift_percent / 100.0
    
    st.divider()
    
    # Section 2: Chaos Mode
    st.subheader("2. Chaos Mode ( The Juice )")
    inject_srm = st.toggle("Inject SRM (Sample Ratio Mismatch)", value=False, help="Breaks the 50/50 randomization.")
    inject_simpson = st.toggle("Inject Simpson's Bias", value=False, help="Inverts trends between Device segments vs Global.")

    st.divider()
    
    if st.button("🚀 Run Simulation", type="primary"):
        st.session_state['run'] = True
    else:
        if 'run' not in st.session_state:
            st.session_state['run'] = False

# --- MAIN LOGIC ---
if st.session_state['run']:
    
    # Generate Data
    gen = SimulationGenerator()
    df = gen.generate_data(n_users, baseline_rate, lift, inject_srm, inject_simpson)
    
    # Calculate Stats
    freq = FrequentistTest()
    bayes = BayesianTest()
    
    srm_res = freq.check_srm(df)
    freq_res = freq.analyze(df)
    bayes_res = bayes.analyze(df)
    
    # --- ZONE 1: HEALTH CHECK (SRM) ---
    st.header("1. Health Check & Validation")
    
    col_health1, col_health2 = st.columns([1, 3])
    
    with col_health1:
        if srm_res['srm_detected']:
            st.error(f"❌ SRM DETECTED (p={srm_res['p_value']:.4f})")
            st.markdown("**CRITICAL:** The sample ratio is flawed. Results are invalid.")
        else:
            st.success(f"✅ Sample Ratio Valid (p={srm_res['p_value']:.4f})")
            
    with col_health2:
        if n_users < 2000:
             st.warning("⚠️ Low Sample Size. False Positives risk increased.")
        elif freq_res['significant'] and not srm_res['srm_detected']:
             st.info("💡 Statistically Significant Result found in a valid test.")
        else:
             st.markdown("Experiment looks technically sound (or insufficient data). Proceed to Analysis.")

    st.divider()

    # --- ZONE 2: EXECUTIVE RESULTS ---
    st.header("2. Executive Results")
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    # Determine Winner
    conv_a = freq_res['stats_a']['cr']
    conv_b = freq_res['stats_b']['cr']
    obs_lift = (conv_b - conv_a) / conv_a if conv_a > 0 else 0
    
    winner_color = "normal"
    if freq_res['significant']:
        if obs_lift > 0: winner_color = "g" # Green
        else: winner_color = "r" # Red
        
    kpi1.metric("Observed Lift", f"{obs_lift*100:.2f}%", 
                delta=f"{obs_lift*100:.2f} pts" if obs_lift != 0 else None,
                delta_color=winner_color)
    
    kpi2.metric("Confidence (Frequentist)", f"{(1 - freq_res['p_value']) * 100:.2f}%",
                help=f"P-value: {freq_res['p_value']:.4f}")
    
    kpi3.metric("Prob. B is Best (Bayesian)", f"{bayes_res['prob_b_wins']*100:.2f}%")
    
    kpi4.metric("Conversions (B)", f"{freq_res['stats_b']['n']} Visitors", f"{freq_res['stats_b']['cr']*100:.2f}% CR")

    st.divider()

    # --- ZONE 3: VISUALIZATION ---
    st.header("3. Visual Insights")
    
    tab1, tab2 = st.tabs(["📈 Time & Cumulative", "🔔 Bayesian Posteriors"])
    
    with tab1:
        # Prepare Time Series Data
        # Calculate daily cumulative conversion rate
        daily = df.groupby(['day_index', 'group'])['converted'].agg(['count', 'sum']).reset_index()
        daily = daily.sort_values(['group', 'day_index'])
        
        # Cumulative Sums
        daily['cum_users'] = daily.groupby('group')['count'].cumsum()
        daily['cum_conv'] = daily.groupby('group')['sum'].cumsum()
        daily['cum_cr'] = daily['cum_conv'] / daily['cum_users']
        
        fig_ts = px.line(daily, x='day_index', y='cum_cr', color='group', 
                         title='Cumulative Conversion Rate over Time',
                         labels={'cum_cr': 'Conversion Rate', 'day_index': 'Day'},
                         markers=True)
        fig_ts.update_layout(yaxis=dict(tickformat=".1%"))
        st.plotly_chart(fig_ts, use_container_width=True)
        
    with tab2:
        # Plot Beta Distributions based on Posteriors
        x = np.linspace(0, max(conv_a, conv_b)*1.5, 500)
        
        y_a = stats.beta.pdf(x, bayes_res['posterior_a']['alpha'], bayes_res['posterior_a']['beta'])
        y_b = stats.beta.pdf(x, bayes_res['posterior_b']['alpha'], bayes_res['posterior_b']['beta'])
        
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Scatter(x=x, y=y_a, mode='lines', name='Group A', fill='tozeroy'))
        fig_dist.add_trace(go.Scatter(x=x, y=y_b, mode='lines', name='Group B', fill='tozeroy'))
        
        fig_dist.update_layout(title="Posterior Distributions (Belief in True CR)",
                               xaxis_title="Conversion Rate", yaxis_title="Density")
        st.plotly_chart(fig_dist, use_container_width=True)

    # --- ZONE 4: SEGMENTATION (The Detective) ---
    st.header("4. Segmentation (Simpson's Check)")
    
    # Calculate Segmented Stats
    seg_stats = df.groupby(['device', 'group'])['converted'].mean().reset_index()
    seg_stats['cr_percent'] = seg_stats['converted'] * 100
    
    col_seg1, col_seg2 = st.columns([2, 1])
    
    with col_seg1:
        fig_seg = px.bar(seg_stats, x='device', y='cr_percent', color='group', barmode='group',
                         title="Conversion Rate by Device", text_auto='.2f')
        st.plotly_chart(fig_seg, use_container_width=True)
        
    with col_seg2:
        st.info("ℹ️ **Simpson's Paradox:** Look for cases where the Global Winner is different from the Segment Winners.")
        
        # Check logic for Simpson's warning in UI
        # (Simplified check: Just comparing means)
        desktop = seg_stats[seg_stats['device'] == 'Desktop']
        mobile = seg_stats[seg_stats['device'] == 'Mobile']
        
        if not desktop.empty and not mobile.empty:
            d_win = 'B' if desktop[desktop['group'] == 'B']['converted'].values[0] > desktop[desktop['group'] == 'A']['converted'].values[0] else 'A'
            m_win = 'B' if mobile[mobile['group'] == 'B']['converted'].values[0] > mobile[mobile['group'] == 'A']['converted'].values[0] else 'A'
            
            # Global winner is based on earlier calculation
            global_win = 'B' if freq_res['stats_b']['cr'] > freq_res['stats_a']['cr'] else 'A'
            
            if d_win == m_win and d_win != global_win:
                st.error("🚨 **SIMPSON'S PARADOX DETECTED!**")
                st.markdown(f"Winner by Device: **{d_win}**")
                st.markdown(f"Global Winner: **{global_win}**")
                st.markdown("The global result is misleading due to traffic mix!")
            else:
                 st.markdown(f"Winner by Device: **{d_win}** (D), **{m_win}** (M)")
                 st.markdown(f"Global Winner: **{global_win}**")

else:
    st.info("👈 Set your parameters and click 'Run Simulation' to start.")

