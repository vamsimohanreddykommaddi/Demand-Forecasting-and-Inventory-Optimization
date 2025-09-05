# importing necessary libraries

import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

import plotly.io as pio
pio.renderers.default = 'browser'

# load the data into pandas dataframe

data = pd.read_csv("D:\Demand_forecasting\demand_inventory.csv")

print(data.head())

# Remove the unnamed column from data

data = data.drop(columns=['Unnamed: 0'])


# Visualizing demand over time

fig_demand = px.line(data, x='Date',
                     y='Demand',
                     title='Demand Over Time')
fig_demand.show()


# Visualizing inventory over time

fig_inventory = px.line(data, x='Date',
                        y='Inventory',
                        title='Inventory Over Time')
fig_inventory.show()

'''
Demand Forecasting :
    There are seasonal patterns in the demand. 
    We can forecast the demand using SARIMA. 
    Let’s first calculate the value of p and q using ACF and PACF plots
'''
data['Date'] = pd.to_datetime(data['Date'],
                                     format='%Y-%m-%d')
time_series = data.set_index('Date')['Demand']

differenced_series = time_series.diff().dropna()

# Plot ACF and PACF of differenced time series
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(differenced_series, ax=axes[0])
plot_pacf(differenced_series, ax=axes[1])
plt.show()

'''
The values of p,d and q are 1,1 and 1
'''

# Training the model and forecasting demand for next 10 days

order = (1, 1, 1)
seasonal_order = (1, 1, 1, 2) #2 because the data contains a time period of 2 months only
model = SARIMAX(time_series, order=order, seasonal_order=seasonal_order)
model_fit = model.fit(disp=False)

future_steps = 10
predictions = model_fit.predict(len(time_series), len(time_series) + future_steps - 1)
predictions = predictions.astype(int)
print(predictions)


'''
Inventory Optimization :
    optimizing inventory according to the forecasted demand for the 
    next ten days
'''

# Create date indices for the future predictions
future_dates = pd.date_range(start=time_series.index[-1] + pd.DateOffset(days=1), periods=future_steps, freq='D')

# Create a pandas Series with the predicted values and date indices
forecasted_demand = pd.Series(predictions, index=future_dates)

# Initial inventory level
initial_inventory = 5500

# Lead time (number of days it takes to replenish inventory) 
lead_time = 1 # it's different for every business, 1 is an example

# Service level (probability of not stocking out)
service_level = 0.95 # it's different for every business, 0.95 is an example

# Calculate the optimal order quantity using the Newsvendor formula
z = np.abs(np.percentile(forecasted_demand, 100 * (1 - service_level)))
order_quantity = np.ceil(forecasted_demand.mean() + z).astype(int)

# Calculate the reorder point
reorder_point = forecasted_demand.mean() * lead_time + z

# Calculate the optimal safety stock
safety_stock = reorder_point - forecasted_demand.mean() * lead_time

# Calculate the total cost (holding cost + stockout cost)
holding_cost = 0.1  # it's different for every business, 0.1 is an example
stockout_cost = 10  # # it's different for every business, 10 is an example
total_holding_cost = holding_cost * (initial_inventory + 0.5 * order_quantity)
total_stockout_cost = stockout_cost * np.maximum(0, forecasted_demand.mean() * lead_time - initial_inventory)

# Calculate the total cost
total_cost = total_holding_cost + total_stockout_cost

print("Optimal Order Quantity:", order_quantity)
print("Reorder Point:", reorder_point)
print("Safety Stock:", safety_stock)
print("Total Cost:", total_cost)

'''
Optimal Order Quantity: 236 – The optimal order quantity refers to the 
quantity of a product that should be ordered from suppliers when the 
inventory level reaches a certain point. In this case, an optimal order 
quantity of 236 units has been calculated.

Reorder Point: 235.25 – The reorder point is the inventory level at which 
a new order should be placed to replenish stock before it runs out. 
In this case, a reorder point of 235.25 units has been calculated, 
which means that when the inventory reaches or falls below this level, 
an order should be placed to replenish stock.

Safety Stock: 114.45 – Safety stock is the additional inventory kept on 
hand to account for uncertainties in demand and supply. 
It acts as a buffer against unexpected variations in demand or lead time. 
In this case, a safety stock of 114.45 units has been calculated, 
which helps ensure that there’s enough inventory to cover potential 
fluctuations in demand or lead time.

Total Cost: 561.80 – The total cost represents the combined costs 
associated with inventory management. In this case, the total cost has 
been calculated as approximately 561.80 units based on the order quantity,
reorder point, safety stock, and associated costs.


'''