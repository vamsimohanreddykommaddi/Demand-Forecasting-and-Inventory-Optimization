# 📈 Demand Forecasting & 📦 Inventory Optimization (SARIMA + Streamlit)

This project is a **Streamlit-based web application** that performs **demand forecasting** using the **SARIMA (Seasonal ARIMA)** model and applies **inventory optimization techniques** to recommend safety stock, reorder point, minimum order quantity, and suggested order quantity.

---

## 🚀 Features

- **CSV Upload**  
  Upload your dataset with columns like `Date`, `Demand`, and optionally `Inventory`.

- **Interactive Data Preview**  
  View raw data and time series plots.

- **Exploratory Visualizations**  
  - Demand over time  
  - Inventory over time  
  - ACF & PACF plots to assist SARIMA parameter selection  

- **Forecasting (SARIMA)**  
  - Train SARIMA model with configurable parameters `(p,d,q)(P,D,Q,m)`  
  - Forecast demand for a user-defined horizon  
  - Evaluate model performance (MAE, RMSE, MAPE)  
  - Visualize history vs. forecast  

- **Inventory Optimization**  
  Based on forecasted demand:  
  - Safety Stock (SS)  
  - Reorder Point (ROP)  
  - Economic Order Quantity (EOQ)  
  - Minimum Order Quantity (MOQ)  
  - Suggested Order Quantity (SOQ)  
  - Estimated Total Inventory Cost (holding + stockout cost)  

---


---

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/vamsimohanreddykommaddi/demand-inventory-forecasting.git
  
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the streamlit app
   ```bash
   streamlit run demand_forecasting_deployment.py




