import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize

# Page Config
st.set_page_config(page_title="Portfolio Optimizer", layout="wide")
st.title("Portfolio Optimizer")

# ----------------------
# 1. Data Fetching (Cached)
# ----------------------
@st.cache_data
def get_data(tickers_str, period):
    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
    if not tickers:
        return None, None, None, None, None
    
    try:
        # Download user tickers + SPY for benchmarking
        all_tickers = list(set(tickers + ["SPY"]))
        data = yf.download(all_tickers, period=period, auto_adjust=True)['Close']
        
        if data.empty:
            return None, None, None, None, None
            
        # Separate Benchmark
        spy_price = data['SPY'].to_frame()
        spy_returns = spy_price.pct_change().dropna()
        
        # Process User Portfolio
        price = data[tickers]
        if isinstance(price, pd.Series):
            price = price.to_frame()
            
        returns = price.pct_change().dropna()
        mean_daily = returns.mean()
        cov_daily = returns.cov()
        
        return price, returns, mean_daily, cov_daily, spy_returns
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None, None, None, None, None

# ----------------------
# 2. Beta Calculation Function
# ----------------------
def calculate_beta(returns, weights, spy_returns):
    # Calculate daily portfolio returns
    port_daily_returns = (returns * weights).sum(axis=1)
    
    # Ensure indices align
    combined = pd.concat([port_daily_returns, spy_returns], axis=1).dropna()
    combined.columns = ['portfolio', 'spy']
    
    # Beta = Cov(Rp, Rm) / Var(Rm) using NumPy
    covariance_matrix = np.cov(combined['portfolio'], combined['spy'])
    covariance = covariance_matrix[0, 1]
    variance = covariance_matrix[1, 1]
    
    return covariance / variance

# ----------------------
# 3. Efficient Frontier Math
# ----------------------
def compute_ef(mean_daily, cov_daily, rf=0.0, n_points=100, horizon="Annual"):
    trading_days = {"Annual": 252, "Monthly": 21}
    scale = trading_days.get(horizon, 252)
    
    mu = mean_daily * scale
    S = cov_daily * scale
    n = len(mu)
    
    # CONSTRAINT: Min 5% | Max 40% per asset
    bounds = tuple((0.05, 0.4) for _ in range(n))
    if n * 0.05 > 1.0:
        return None

    init = np.repeat(1.0 / n, n)
    cons_sum = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}

    def portfolio_return(w): return float(np.dot(w, mu))
    def portfolio_var(w): return float(np.dot(w, S @ w))
    def neg_sharpe(w): 
        return - (portfolio_return(w) - rf) / (np.sqrt(portfolio_var(w)) + 1e-12)

    # 1. Max Sharpe
    res_sh = minimize(neg_sharpe, init, method='SLSQP', bounds=bounds, constraints=cons_sum)
    w_sh = res_sh.x
    ret_sh, vol_sh = portfolio_return(w_sh), np.sqrt(portfolio_var(w_sh))
    
    # 2. Min Variance
    res_min = minimize(portfolio_var, init, method='SLSQP', bounds=bounds, constraints=cons_sum)
    w_min = res_min.x
    ret_min, vol_min = portfolio_return(w_min), np.sqrt(portfolio_var(w_min))

    # 3. Frontier Curve
    max_individual_ret = mu.max()
    target_returns = np.linspace(ret_min, max_individual_ret, n_points)
    
    ef_vols, ef_rets = [], []
    for target in target_returns:
        cons = (cons_sum, {'type': 'eq', 'fun': lambda w: portfolio_return(w) - target})
        res = minimize(portfolio_var, init, method='SLSQP', bounds=bounds, constraints=cons)
        if res.success:
            ef_vols.append(np.sqrt(res.fun))
            ef_rets.append(target)

    # GAP FIX: Explicitly add and sort Max Sharpe into the curve
    ef_vols.append(vol_sh)
    ef_rets.append(ret_sh)
    sorted_curve = sorted(zip(ef_vols, ef_rets))
    final_vols, final_rets = zip(*sorted_curve)

    return {
        "mu": mu, "w_sh": w_sh, "ret_sh": ret_sh, "vol_sh": vol_sh,
        "w_min": w_min, "ret_min": ret_min, "vol_min": vol_min,
        "ef_vols": np.array(final_vols), "ef_returns": np.array(final_rets)
    }

# ----------------------
# 4. Sidebar
# ----------------------
st.sidebar.header("Portfolio Configuration")
ticker_input = st.sidebar.text_input("Tickers", "AAPL, MSFT, GOOG, TSLA, NVDA")
time_selector = st.sidebar.selectbox("Period", ["1y", "2y", "5y"], index=0)
rf_input = st.sidebar.number_input("Risk-Free Rate", value=0.04)

# ----------------------
# 5. App Execution
# ----------------------
price, returns, mean_daily, cov_daily, spy_rets = get_data(ticker_input, time_selector)

if price is not None:
    tab1, tab2, tab3 = st.tabs(["Performance vs Benchmark", "Optimization", "Risk Simulation"])

    # --- TAB 1: BENCHMARKING ---
    with tab1:
        st.subheader("Portfolio Growth vs. S&P 500 (Normalized)")
        port_cum = (1 + returns.mean(axis=1)).cumprod()
        spy_cum = (1 + spy_rets['SPY']).cumprod()
        
        fig_bench = go.Figure()
        fig_bench.add_trace(go.Scatter(x=port_cum.index, y=port_cum, name="Your Portfolio (Equal Weight)"))
        fig_bench.add_trace(go.Scatter(x=spy_cum.index, y=spy_cum, name="S&P 500 (SPY)", line=dict(dash='dot')))
        st.plotly_chart(fig_bench, use_container_width=True)

    # --- TAB 2: OPTIMIZATION ---
    with tab2:
        horizon_sel = st.selectbox("Scale", ["Annual", "Monthly"])
        stats = compute_ef(mean_daily, cov_daily, rf=rf_input, horizon=horizon_sel)
        
        if stats:
            scale = 252 if horizon_sel == "Annual" else 21
            spy_ann_ret = spy_rets['SPY'].mean() * scale
            spy_ann_vol = spy_rets['SPY'].std() * np.sqrt(scale)

            fig_ef = go.Figure()
            fig_ef.add_trace(go.Scatter(x=stats["ef_vols"]*100, y=stats["ef_returns"]*100, mode='lines', name='Efficient Frontier', line=dict(color='black', width=2, dash='dash')))
            fig_ef.add_trace(go.Scatter(x=[stats["vol_sh"]*100], y=[stats["ret_sh"]*100], mode='markers', name='Max Sharpe', marker=dict(size=15, color='red', symbol='star')))
            fig_ef.add_trace(go.Scatter(x=[stats["vol_min"]*100], y=[stats["ret_min"]*100], mode='markers', name='Min Variance', marker=dict(size=12, color='green', symbol='circle')))
            fig_ef.add_trace(go.Scatter(x=[spy_ann_vol*100], y=[spy_ann_ret*100], mode='markers', name='S&P 500 Benchmark', marker=dict(size=12, color='blue', symbol='x')))
            
            fig_ef.update_layout(title=f"Portfolio Efficiency Comparison ({horizon_sel})", xaxis_title="Volatility (Risk %)", yaxis_title="Expected Return (%)", template="plotly_white", height=600)
            st.plotly_chart(fig_ef, use_container_width=True)

            # Calculate Betas
            beta_sh = calculate_beta(returns, stats["w_sh"], spy_rets)
            beta_min = calculate_beta(returns, stats["w_min"], spy_rets)

            st.subheader("Performance Summary")
            comparison_data = {
                "Max Sharpe": [stats["ret_sh"]*100, stats["vol_sh"]*100, (stats["ret_sh"]-rf_input)/stats["vol_sh"], beta_sh],
                "Min Variance": [stats["ret_min"]*100, stats["vol_min"]*100, (stats["ret_min"]-rf_input)/stats["vol_min"], beta_min],
                "S&P 500": [spy_ann_ret*100, spy_ann_vol*100, (spy_ann_ret-rf_input)/spy_ann_vol, 1.0]
            }
            
            comparison_df = pd.DataFrame(comparison_data, index=["Expected Return (%)", "Volatility (%)", "Sharpe Ratio", "Portfolio Beta (β)"]).round(2)
            st.table(comparison_df)

            st.subheader("Asset Allocation (%)")
            st.write("Constraints Applied: Min 5% | Max 40% per asset.")
            weights_distribution = pd.DataFrame({
                "Max Sharpe Portfolio (%)": stats["w_sh"] * 100,
                "Min Variance Portfolio (%)": stats["w_min"] * 100
            }, index=returns.columns)
            st.dataframe(weights_distribution.round(0).astype(int), use_container_width=True)

    # --- TAB 3: MONTE CARLO ---
    with tab3:
        col1, col2, col3 = st.columns(3)
        port_val = col1.number_input("Initial Investment ($)", value=10000)
        mc_sims = col2.slider("Number of Simulations", 100, 1000, 300)
        mc_days = col3.slider("Simulation Days", 30, 756, 252)
        use_ef_check = st.checkbox("Apply Max-Sharpe Weights to Simulation", value=True)

        n = len(returns.columns)
        if use_ef_check:
            weights = stats["w_sh"]
        else:
            weights = np.repeat(1.0/n, n)

        L = np.linalg.cholesky(cov_daily.values + 1e-9 * np.eye(n))
        sims_matrix = np.zeros((mc_days, mc_sims))
        
        for m in range(mc_sims):
            Z = np.random.normal(size=(mc_days, n))
            daily_rets = mean_daily.values + Z @ L.T
            port_rets = daily_rets @ weights
            sims_matrix[:, m] = np.cumprod(1.0 + port_rets) * port_val

        fig_mc = go.Figure()
        for i in range(min(mc_sims, 100)):
            fig_mc.add_trace(go.Scatter(y=sims_matrix[:, i], mode='lines', line=dict(width=1), opacity=0.15, showlegend=False))
        
        median_path = np.median(sims_matrix, axis=1)
        fig_mc.add_trace(go.Scatter(y=median_path, mode='lines', line=dict(color='red', width=3), name='Median Growth'))
        fig_mc.update_layout(title="Portfolio Value Projection", xaxis_title="Days", yaxis_title="Portfolio Value ($)", template="plotly_white")
        st.plotly_chart(fig_mc, use_container_width=True)

        final_vals = sims_matrix[-1, :]
        p5, p50, p95 = np.percentile(final_vals, [5, 50, 95])
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Downside (5th %)", f"${p5:,.0f}")
        m2.metric("Median (50th %)", f"${p50:,.0f}")
        m3.metric("Upside (95th %)", f"${p95:,.0f}")

else:
    st.info("Awaiting input. Enter tickers in the sidebar to generate analysis.")