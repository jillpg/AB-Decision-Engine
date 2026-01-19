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
st.title("🧪 The Experimentation Decision Engine")
st.markdown("*Simulating hostile environments to train decision making.*")

# --- SIDEBAR: INPUTS ---
with st.sidebar:
    st.header("🎛️ Configuration")
    
    # Section 1: Parameters
    st.subheader("Simulation Parameters")
    n_users = st.slider("Traffic Volume", min_value=1000, max_value=50000, value=10000, step=1000)
    baseline_rate = st.slider("Baseline Conversion Rate (%)", 1.0, 50.0, 12.0, 0.5) / 100.0
    lift_percent = st.slider("Expected Lift (%)", -20.0, 50.0, 10.0, 1.0)
    lift = lift_percent / 100.0
    
    st.divider()
    
    # Section 1.5: Advanced Config
    st.subheader("Advanced Settings")
    metric_type = st.radio("Primary Metric", ["Conversion Rate", "Revenue (RPV)"])
    if metric_type == "Revenue (RPV)":
        aov = st.number_input("Average Order Value ($)", value=50.0)
        variance = st.slider("Variance (Lognormal Sigma)", 0.1, 2.0, 1.0, help="Controls the skew of the revenue distribution.")
    else:
        aov = 50.0
        variance = 1.0
        
    analysis_type = st.radio("Inference Framework", ["Fixed Horizon", "Sequential Testing"])
    
    st.divider()
    
    # Section 2: Chaos Mode
    st.subheader("Stress Testing (Chaos)")
    inject_srm = st.toggle("Simulate SRM (Sample Ratio Mismatch)", value=False, help="Artificially biases the traffic allocation (e.g., 30/70 split).")
    inject_simpson = st.toggle("Simulate Simpson's Paradox", value=False, help="Inverts segment-level performance vs global performance.")

    st.divider()
    
    if st.button("🚀 Run Simulation", type="primary"):
        st.session_state['run'] = True
    else:
        if 'run' not in st.session_state:
            st.session_state['run'] = False

# --- MAIN LOGIC ---
if st.session_state['run']:
    
    # Generate Data
    metric_key = 'revenue' if metric_type == "Revenue (RPV)" else 'conversion'
    metric_col = 'revenue' if metric_key == 'revenue' else 'converted'
    
    gen = SimulationGenerator()
    df = gen.generate_data(n_users, baseline_rate, lift, inject_srm, inject_simpson, 
                          metric_type=metric_key, aov=aov, variance=variance)
    
    # Calculate Stats
    freq = FrequentistTest()
    bayes = BayesianTest()
    
    srm_res = freq.check_srm(df)
    
    if metric_key == 'conversion':
        freq_res = freq.analyze_conversion(df)
        bayes_res = bayes.analyze_conversion(df)
        val_a = freq_res['stats_a']['mean'] * 100
        val_b = freq_res['stats_b']['mean'] * 100
        val_label = "%"
    else:
        freq_res = freq.analyze_revenue(df)
        bayes_res = bayes.analyze_revenue(df)
        val_a = freq_res['stats_a']['mean']
        val_b = freq_res['stats_b']['mean']
        val_label = "$"
    
    # --- ZONE 1: HEALTH CHECK (SRM) ---
    st.header("1. Data Integrity & Health")
    
    col_health1, col_health2 = st.columns([1, 3])
    
    with col_health1:
        if srm_res['srm_detected']:
            st.error(f"❌ SRM DETECTED (p={srm_res['p_value']:.4f})")
            st.markdown("**CRITICAL:** Traffic allocation is biased. Results are statistically invalid.")
        else:
            st.success(f"✅ Sample Ratio Valid (p={srm_res['p_value']:.4f})")
            
    with col_health2:
        if srm_res['srm_detected']:
             st.warning("⚠️ **Warning:** Sample ratio mismatch prevents reliable analysis. Investigate traffic allocation.")
        elif n_users < 2000:
             st.warning("⚠️ Low Sample Size. Increased risk of false positives/negatives.")
        elif freq_res['significant']:
             st.info("💡 Statistically Significant Result observed.")
        else:
             st.markdown("Experiment design appears sound. Proceeding to detailed analysis.")

    st.divider()

    # --- ZONE 2: EXECUTIVE RESULTS ---
    st.header("2. Performance Overview")
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    # Determine Winner
    obs_lift = freq_res['lift']
    
    winner_color = "normal"
    if freq_res['significant']:
        if obs_lift > 0: winner_color = "green" 
        else: winner_color = "red" 
        
    kpi1.metric("Observed Lift", f"{obs_lift*100:.2f}%", 
                delta=f"{obs_lift*100:.2f} pts" if obs_lift != 0 else None,
                delta_color=winner_color)
    
    kpi2.metric(f"Confidence ({'T-Test' if metric_key == 'revenue' else 'Z-Test'})", 
                f"{(1 - freq_res['p_value']) * 100:.2f}%",
                help=f"P-value: {freq_res['p_value']:.4f}")
    
    kpi3.metric("Prob. B is Better (Bayesian)", f"{bayes_res['prob_b_wins']*100:.2f}%",
                help=f"Expected Loss: {bayes_res['expected_loss']:.4f}")
    
    kpi4.metric(f"Avg {metric_type}", f"{val_b:.2f}{val_label}", f"vs A: {val_a:.2f}{val_label}")

    st.divider()

    # --- ZONE 3: VISUALIZATION ---
    st.header("3. Analytical Insights")
    
    tab1, tab2, tab3 = st.tabs(["📈 Cumulative Trend", "🔔 Posterior Distributions", "🚦 Sequential Analysis"])
    
    with tab1:
        # Prepare Time Series Data
        daily = df.groupby(['day_index', 'group']).agg(
            users=('user_id', 'count'),
            converted=('converted', 'sum'),
            revenue=('revenue', 'sum')
        ).reset_index()
        daily = daily.sort_values(['group', 'day_index'])
        
        daily['cum_users'] = daily.groupby('group')['users'].cumsum()
        daily['cum_value'] = daily.groupby('group')[metric_col].cumsum()
        daily['cum_metric'] = daily['cum_value'] / daily['cum_users']
        
        fig_ts = px.line(daily, x='day_index', y='cum_metric', color='group', 
                         title=f'Cumulative {metric_type} Evolution',
                         labels={'cum_metric': metric_type, 'day_index': 'Day of Experiment'},
                         markers=True)
        st.plotly_chart(fig_ts, use_container_width=True)
        
    with tab2:
        if metric_key == 'conversion':
            # Beta Distributions
            x = np.linspace(0, max(val_a/100, val_b/100)*1.5, 500)
            y_a = stats.beta.pdf(x, bayes_res['posterior_a']['alpha'], bayes_res['posterior_a']['beta'])
            y_b = stats.beta.pdf(x, bayes_res['posterior_b']['alpha'], bayes_res['posterior_b']['beta'])
            
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Scatter(x=x, y=y_a, mode='lines', name='Group A', fill='tozeroy'))
            fig_dist.add_trace(go.Scatter(x=x, y=y_b, mode='lines', name='Group B', fill='tozeroy'))
            fig_dist.update_layout(title="Posterior Belief Distributions (Beta)", xaxis_title="Conversion Rate")
        else:
            # Bootstrap Histograms
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(x=bayes_res['samples_a'], name='A (Bootstrap)', opacity=0.7))
            fig_dist.add_trace(go.Histogram(x=bayes_res['samples_b'], name='B (Bootstrap)', opacity=0.7))
            fig_dist.update_layout(title="Bootstrap Evaluation of Mean RPV", barmode='overlay')
            
        st.plotly_chart(fig_dist, use_container_width=True)

    with tab3:
        if analysis_type == "Sequential Testing":
            st.markdown("### Sequential Analysis (O'Brien-Fleming)")
            st.markdown("Dynamic decision boundaries that allow continuous monitoring while controlling Type I error.")
            
            # Reconstruct Z-Score path
            days = sorted(df['day_index'].unique())
            day_data = []
            n_total = len(df)
            
            for d in days:
                sub_df = df[df['day_index'] <= d]
                check = freq.analyze_conversion(sub_df) if metric_key == 'conversion' else freq.analyze_revenue(sub_df)
                n_curr = len(sub_df)
                
                p = check['p_value']
                z = stats.norm.ppf(1 - p/2) # two-sided approximation
                if check['lift'] < 0: z = -z
                
                bound = freq.get_sequential_boundary(n_curr, n_total)
                day_data.append({'day': d, 'z_score': z, 'boundary': bound})
            
            seq_df = pd.DataFrame(day_data)
            
            fig_seq = go.Figure()
            fig_seq.add_trace(go.Scatter(x=seq_df['day'], y=seq_df['z_score'], mode='lines+markers', name='Test Statistic (Z)'))
            fig_seq.add_trace(go.Scatter(x=seq_df['day'], y=seq_df['boundary'], mode='lines', name='Upper Boundary', line=dict(dash='dash', color='red')))
            fig_seq.add_trace(go.Scatter(x=seq_df['day'], y=-seq_df['boundary'], mode='lines', name='Lower Boundary', line=dict(dash='dash', color='red')))
            
            # Add Fill for Decision Zone
            fig_seq.add_hrect(y0=-1.96, y1=1.96, annotation_text="Standard Significance Zone (Z=1.96)", annotation_position="top left", fillcolor="yellow", opacity=0.1, line_width=0)
            
            fig_seq.update_layout(title="Sequential Test Trajectory", xaxis_title="Day", yaxis_title="Z-Score")
            st.plotly_chart(fig_seq, use_container_width=True)
            
            # Decision
            last_day = seq_df.iloc[-1]
            if abs(last_day['z_score']) > last_day['boundary']:
                st.success(f"✅ **Statistical Significance Reached:** Stopping criterion met at Day {int(last_day['day'])}.")
            else:
                st.info("⏳ **Status:** Insufficient evidence to reject null hypothesis. Continue testing.")

        else:
            st.warning("Enable 'Sequential Testing' in the configuration panel to view this analysis.")

    # --- ZONE 4: SEGMENTATION (The Detective) ---
    st.header("4. Segmentation & Bias Detection")
    
    # Calculate Segmented Stats
    seg_stats = df.groupby(['device', 'group'])[metric_col].mean().reset_index()
    seg_stats['metric_val'] = seg_stats[metric_col]
    if metric_key == 'conversion': seg_stats['metric_val'] *= 100
    
    col_seg1, col_seg2 = st.columns([2, 1])
    
    with col_seg1:
        fig_seg = px.bar(seg_stats, x='device', y='metric_val', color='group', barmode='group',
                         title=f"{metric_type} by Segment", text_auto='.2f')
        st.plotly_chart(fig_seg, use_container_width=True)
        
    with col_seg2:
        st.info("ℹ️ **Consistency Check:** Analyzing if aggregate results align with segment-level performance (Simpson's Paradox).")
        
        # Check logic for Simpson's warning in UI
        desktop = seg_stats[seg_stats['device'] == 'Desktop']
        mobile = seg_stats[seg_stats['device'] == 'Mobile']
        
        if not desktop.empty and not mobile.empty:
            d_val_a = desktop[desktop['group'] == 'A']['metric_val'].values[0]
            d_val_b = desktop[desktop['group'] == 'B']['metric_val'].values[0]
            d_win = 'B' if d_val_b > d_val_a else 'A'
            
            m_val_a = mobile[mobile['group'] == 'A']['metric_val'].values[0]
            m_val_b = mobile[mobile['group'] == 'B']['metric_val'].values[0]
            m_win = 'B' if m_val_b > m_val_a else 'A'
            
            # Global winner
            global_mean_a = freq_res['stats_a']['mean']
            global_mean_b = freq_res['stats_b']['mean']
            global_win = 'B' if global_mean_b > global_mean_a else 'A'
            
            if d_win == m_win and d_win != global_win:
                st.error("🚨 **SIMPSON'S PARADOX DETECTED**")
                st.markdown(f"Segment Trend: **{d_win}** is superior.")
                st.markdown(f"Aggregate Trend: **{global_win}** appears superior.")
                st.markdown("⚠️ **Conclusion:** Aggregate results are misleading due to traffic mix bias. Rely on segment data.")
            else:
                 st.markdown(f"Segment Winner: **{d_win}** (D), **{m_win}** (M)")
                 st.markdown(f"Global Winner: **{global_win}**")

else:
    st.info("👈 Configure parameters and click 'Run Simulation' to begin.")

