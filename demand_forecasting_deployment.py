import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.stats import norm

st.set_page_config(page_title="Demand & Inventory Forecasting", layout="wide")

st.title("📈 Demand Forecasting & 📦 Inventory Optimization (SARIMA)")

# ---- Sidebar ----
with st.sidebar:
    st.header("⚙️ Configuration")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    st.markdown("---")
    st.caption("Your CSV should have columns like Date, Demand, Inventory.")

# ---- Data Loading ----
@st.cache_data
def load_df(file):
    df = pd.read_csv(file)
    return df

if uploaded is not None:
    data = load_df(uploaded)
else:
    st.info("Upload a CSV to begin.")
    st.stop()

st.subheader("1) Data Preview")
st.dataframe(data.head())

# ---- Column Selection ----
st.subheader("2) Select Columns")
date_col = st.selectbox("Date column", options=data.columns, index=list(data.columns).index("Date") if "Date" in data.columns else 0)
demand_col = st.selectbox("Demand (target) column", options=data.columns, index=list(data.columns).index("Demand") if "Demand" in data.columns else 0)
inv_col = st.selectbox("Inventory column (optional)", options=["<none>"] + list(data.columns), index=(["<none>"] + list(data.columns)).index("Inventory") if "Inventory" in data.columns else 0)

# ---- Clean Data ----
df = data.copy()
for col in list(df.columns):
    if col.lower().startswith("unnamed"):
        df = df.drop(columns=[col])

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col])
df = df.sort_values(date_col)

# ---- Visuals ----
st.subheader("3) Exploratory Charts")
fig_demand = px.line(df, x=date_col, y=demand_col, title="Demand Over Time")
st.plotly_chart(fig_demand, use_container_width=True)

if inv_col != "<none>":
    fig_inventory = px.line(df, x=date_col, y=inv_col, title="Inventory Over Time")
    st.plotly_chart(fig_inventory, use_container_width=True)

# ---- Time Series ----
ts = df.set_index(date_col)[demand_col].asfreq("D")
ts = ts.interpolate(method="linear")

st.markdown("#### Stationarity Aids: ACF & PACF")
diff_ts = ts.diff().dropna()

fig1 = plt.figure(figsize=(7, 4))
plot_acf(diff_ts, ax=plt.gca())
st.pyplot(fig1)

fig2 = plt.figure(figsize=(7, 4))
plot_pacf(diff_ts, ax=plt.gca(), method="ywm")
st.pyplot(fig2)

# ---- SARIMA Settings ----
st.subheader("4) Train/Test Split & SARIMA Settings")
c1, c2, c3 = st.columns(3)
with c1:
    test_size = st.slider("Test size (days)", 7, min(180, len(ts)//3), min(30, max(7, len(ts)//4)))
with c2:
    horizon = st.number_input("Forecast horizon (days)", min_value=1, max_value=365, value=14, step=1)
with c3:
    st.caption("Tip: For weekly patterns try m=7, monthly ≈30.")

st.markdown("##### SARIMA Orders")
c4, c5 = st.columns(2)
with c4:
    p = st.number_input("p", 0, 5, 1)
    d = st.number_input("d", 0, 2, 1)
    q = st.number_input("q", 0, 5, 1)
with c5:
    P = st.number_input("P (seasonal)", 0, 5, 1)
    D = st.number_input("D (seasonal)", 0, 2, 1)
    Q = st.number_input("Q (seasonal)", 0, 5, 1)
    m = st.number_input("m (seasonal period)", 1, 365, 7)

train, test = ts.iloc[:-test_size], ts.iloc[-test_size:]

# ---- Train SARIMA ----
st.subheader("5) Fit Model & Forecast")
if st.button("Train SARIMA & Forecast"):
    with st.spinner("Training model..."):
        try:
            model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, m),
                            enforce_stationarity=False, enforce_invertibility=False)
            results = model.fit(disp=False)
            st.success("Model fit complete.")

            fc = results.get_forecast(steps=len(test)+horizon)
            fc_mean = fc.predicted_mean

            st.session_state["fc_mean"] = fc_mean
            st.session_state["ts"] = ts

            pred_test = fc_mean.iloc[:len(test)]
            mae = np.mean(np.abs(test - pred_test))
            rmse = np.sqrt(np.mean((test - pred_test)**2))
            mape = np.mean(np.abs((test - pred_test) / np.maximum(1e-9, test))) * 100

            st.markdown(f"**MAE:** {mae:,.2f} | **RMSE:** {rmse:,.2f} | **MAPE:** {mape:,.2f}%")

            hist = px.line(ts.reset_index(), x=ts.index.name, y=ts.name, title="History & Forecast")
            future_line = fc_mean.reset_index()
            future_line.columns = [ts.index.name, "Forecast"]
            hist = hist.add_scatter(x=future_line[ts.index.name], y=future_line["Forecast"], mode="lines", name="Forecast")
            st.plotly_chart(hist, use_container_width=True)

            st.markdown("##### Forecast Table")
            fc_df = pd.DataFrame({"date": fc_mean.index, "forecast": fc_mean.values})
            st.dataframe(fc_df.tail(horizon))

        except Exception as e:
            st.error(f"Model failed: {e}")

# ---- Inventory Optimization ----
st.subheader("6) Inventory Optimization")
if "fc_mean" not in st.session_state:
    st.info("⚠️ Train the SARIMA model first to generate forecasts.")
else:
    if st.button("Run Inventory Optimization"):
        ts = st.session_state["ts"]
        fc_mean = st.session_state["fc_mean"]

        colA, colB, colC = st.columns(3)
        with colA:
            initial_inventory = st.number_input("Initial Inventory", min_value=0, value=5500, step=1)
            lead_time = st.number_input("Lead Time (days)", min_value=1, value=1, step=1)
            ordering_cost = st.number_input("Ordering Cost per order", min_value=1.0, value=50.0, step=1.0)
        with colB:
            service_level = st.slider("Service Level", min_value=0.5, max_value=0.999, value=0.95)
            holding_cost = st.number_input("Holding Cost per unit", min_value=0.01, value=0.1, step=0.01)
        with colC:
            stockout_cost = st.number_input("Stockout Cost per unit", min_value=0.0, value=10.0, step=0.5)

        # Demand statistics
        mu_d = ts.mean()
        sigma_d = ts.std(ddof=1)

        # Safety Stock
        z = float(norm.ppf(service_level))
        safety_stock = z * sigma_d * np.sqrt(lead_time)

        # Reorder Point
        reorder_point = mu_d * lead_time + safety_stock

        # EOQ = Minimum Order Quantity
        annual_demand = mu_d * 365
        eoq = np.sqrt((2 * annual_demand * ordering_cost) / (holding_cost if holding_cost > 0 else 1))
        min_order_qty = int(np.ceil(eoq)) if eoq > 0 else 1

        # Suggested Order Quantity
        expected_demand_next_LT = mu_d * lead_time
        raw_order_qty = np.ceil(reorder_point + expected_demand_next_LT - initial_inventory)
        suggested_order_qty = int(max(min_order_qty, raw_order_qty))

        # Cost Estimation
        avg_inventory = max(0, (initial_inventory + suggested_order_qty) / 2)
        total_holding_cost = holding_cost * avg_inventory
        expected_shortage = max(0.0, expected_demand_next_LT - (initial_inventory + suggested_order_qty))
        total_stockout_cost = stockout_cost * expected_shortage
        total_cost = total_holding_cost + total_stockout_cost

        st.markdown(f"""
        **Reorder Point (ROP):** {reorder_point:,.2f}  
        **Safety Stock:** {safety_stock:,.2f}  
        **Economic Order Quantity (MOQ):** {min_order_qty:,}  
        **Suggested Order Quantity:** {suggested_order_qty:,}  
        **Estimated Total Cost:** {total_cost:,.2f}
        """)
        st.caption("Formula: Safety Stock = z × σ_d × √L, ROP = μ_d × L + SS, MOQ from EOQ formula")
