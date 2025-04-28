#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Part 1: Loading and Cleaning the Data
try:
    # Load the data
    file_path = 'NASDAQ_100_cleaned.csv'
    df = pd.read_csv(file_path)






    print("Data loaded successfully!")
    print(df.head())

    # Convert 'date' column to datetime and set it as the index
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.set_index('date', inplace=True)
    print("Date column converted and set as index successfully!")
    print(df.head())

except Exception as e:
    print(f"Error in Part 1 (Loading and Cleaning Data): {e}")


# In[17]:


# Part 2: Calculating Daily Returns
try:
    # Calculate daily returns as percentage change
    df['Return'] = df['close'].pct_change()
    print("Daily returns calculated successfully!")
    print(df[['close', 'Return']].head(10))

except Exception as e:
    print(f"Error in Part 2 (Calculating Daily Returns): {e}")


# In[19]:


# Part 3: Classifying Movements
try:
    # Classify movements based on the return value
    def classify_movement(value):
        if value > 0.001:
            return 'Up'
        elif value < -0.001:
            return 'Down'
        else:
            return 'No Change'

    df['Movement'] = df['Return'].apply(classify_movement)
    print("Movements classified successfully!")
    print(df[['Return', 'Movement']].head(10))

except Exception as e:
    print(f"Error in Part 3 (Classifying Movements): {e}")


# In[21]:


# Part 4: Creating the Transition Matrix
try:
    states = ['Up', 'Down', 'No Change']
    transition_matrix = pd.DataFrame(0, index=states, columns=states)

    # Calculate transitions
    for i in range(1, len(df)):
        prev_state = df['Movement'].iloc[i - 1]
        curr_state = df['Movement'].iloc[i]
        transition_matrix.loc[prev_state, curr_state] += 1

    # Normalize to get probabilities
    transition_matrix = transition_matrix.div(transition_matrix.sum(axis=1), axis=0)
    print("Transition matrix created successfully!")
    print(transition_matrix)

except Exception as e:
    print(f"Error in Part 4 (Creating Transition Matrix): {e}")


# In[23]:


# Part 5: Making Predictions
try:
    def predict_next_state(current_state, transition_matrix):
        return np.random.choice(states, p=transition_matrix.loc[current_state].values)

    def predict_return(state, historical_returns):
        if state == 'Up':
            return np.random.choice(historical_returns[historical_returns > 0])
        elif state == 'Down':
            return np.random.choice(historical_returns[historical_returns < 0])
        else:
            return 0

    print("Prediction functions defined successfully!")

except Exception as e:
    print(f"Error in Part 5 (Making Predictions): {e}")


# In[29]:


# Part 6: Historical Prediction to Validate Accuracy

def predict_return(state, historical_returns):
    if state == 'Up':
        positive_returns = historical_returns[historical_returns > 0]
        if not positive_returns.empty:
            return np.random.choice(positive_returns)
        else:
            return historical_returns.mean()  # Fallback to mean
    elif state == 'Down':
        negative_returns = historical_returns[historical_returns < 0]
        if not negative_returns.empty:
            return np.random.choice(negative_returns)
        else:
            return historical_returns.mean()  # Fallback to mean
    else:
        return 0  # No Change

predictions = []

for i in range(1, len(df)):
    # Predict next state and return
    current_state = df['Movement'].iloc[i - 1]
    next_state = predict_next_state(current_state, transition_matrix)
    predicted_return = predict_return(next_state, df['Return'].iloc[:i])  # Use past data for prediction
    predicted_price = df['close'].iloc[i - 1] * (1 + predicted_return)

    # Store the prediction and actual data
    predictions.append({
        'Date': df.index[i],
        'Actual Price': df['close'].iloc[i],
        'Predicted Price': predicted_price,
        'Error': abs(predicted_price - df['close'].iloc[i]),
        'Actual Return': df['Return'].iloc[i],
        'Predicted Return': predicted_return
    })

# Convert predictions to a DataFrame
predictions_df = pd.DataFrame(predictions)

# Calculate accuracy metrics
mae = np.mean(predictions_df['Error'])
mape = np.mean(np.abs((predictions_df['Actual Price'] - predictions_df['Predicted Price']) / predictions_df['Actual Price'])) * 100

print('\nPrediction Accuracy:')
print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')

# Display predicted vs actual prices
print('\nPredicted vs Actual Prices:')
print(predictions_df[['Date', 'Actual Price', 'Predicted Price', 'Error']])

# Plotting the predicted vs actual prices
plt.figure(figsize=(12, 6))
plt.plot(predictions_df['Date'], predictions_df['Actual Price'], label='Actual Price', color='blue', linestyle='-', marker='o')
plt.plot(predictions_df['Date'], predictions_df['Predicted Price'], label='Predicted Price', color='red', linestyle='--', marker='x',alpha=0.3)
plt.title('Predicted vs Actual NASDAQ 100 Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()



# In[33]:


# Plotting the predicted vs actual prices
plt.figure(figsize=(12, 6))
plt.plot(predictions_df['Date'], predictions_df['Actual Price'], label='Actual Price', color='blue', linestyle='-', marker='o', alpha=0.8)
plt.plot(predictions_df['Date'], predictions_df['Predicted Price'], label='Predicted Price', color='red', linestyle='--', marker='x', alpha=0.2)
plt.title('Predicted vs Actual NASDAQ 100 Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# Step 7: Predict Next 30 Days Based on the Last Known State
next_30_days = []
current_state = df['Movement'].iloc[-1]
current_price = df['close'].iloc[-1]

for day in range(1, 31):
    next_state = predict_next_state(current_state, transition_matrix)
    predicted_return = predict_return(next_state, df['Return'].dropna())
    next_price = current_price * (1 + predicted_return)

    # Append the predicted values
    next_30_days.append({
        'Day': day,
        'Predicted Movement': next_state,
        'Predicted Return': predicted_return,
        'Predicted Closing Price': next_price,
        'Predicted Percent Change': predicted_return * 100
    })

    # Update for next iteration
    current_state = next_state
    current_price = next_price

# Convert to DataFrame and display
predicted_30_df = pd.DataFrame(next_30_days)
print('\nPredicted Values for the Next 30 Days:')
print(predicted_30_df)

# Save predictions to a CSV file
predicted_30_df.to_csv('predicted_30_days.csv', index=False)


# In[37]:


# Plotting the predicted vs actual prices
plt.figure(figsize=(12, 6))
plt.plot(predictions_df['Date'], predictions_df['Actual Price'], label='Actual Price', color='blue', linestyle='-', marker='o', alpha=0.8)
plt.plot(predictions_df['Date'], predictions_df['Predicted Price'], label='Predicted Price', color='red', linestyle='--', marker='x', alpha=0.5)
plt.title('Predicted vs Actual NASDAQ 100 Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# Step 7: Predict Next 30 Days Based on the Last Known State
next_30_days = []
current_state = df['Movement'].iloc[-31]  # Start from the last known historical state
current_price = df['close'].iloc[-31]     # Start from the last known historical price

for i in range(30):
    next_state = predict_next_state(current_state, transition_matrix)
    predicted_return = predict_return(next_state, df['Return'].dropna())
    next_price = current_price * (1 + predicted_return)

    # Append the predicted values
    next_30_days.append({
        'Date': df.index[-30 + i],  # Use historical dates for alignment
        'Predicted Movement': next_state,
        'Predicted Return': predicted_return,
        'Predicted Closing Price': next_price,
        'Predicted Percent Change': predicted_return * 100
    })

    # Update for next iteration
    current_state = next_state
    current_price = next_price

# Convert to DataFrame and display
predicted_30_df = pd.DataFrame(next_30_days)

# Retrieve actual data for the same 30 days
actual_next_30_df = df.iloc[-30:][['close', 'Return']]
actual_next_30_df = actual_next_30_df.reset_index()
actual_next_30_df.rename(columns={'close': 'Actual Closing Price', 'Return': 'Actual Return'}, inplace=True)

# Merge predicted and actual data on dates
comparison_df = pd.merge(predicted_30_df, actual_next_30_df, left_on='Date', right_on='date', how='inner')

# Calculate prediction errors
comparison_df['Error'] = abs(comparison_df['Predicted Closing Price'] - comparison_df['Actual Closing Price'])
comparison_df['Percent Error'] = abs((comparison_df['Predicted Closing Price'] - comparison_df['Actual Closing Price']) / comparison_df['Actual Closing Price']) * 100

# Calculate accuracy metrics
mae_next_30 = np.mean(comparison_df['Error'])
mape_next_30 = np.mean(comparison_df['Percent Error'])

print('\nPredicted Values for the Next 30 Days with Comparison:')
print(comparison_df)

print(f'\nPrediction Accuracy for Next 30 Days:')
print(f'Mean Absolute Error (MAE): {mae_next_30:.2f}')
print(f'Mean Absolute Percentage Error (MAPE): {mape_next_30:.2f}%')

# Plotting the predicted vs actual prices for the next 30 days
plt.figure(figsize=(12, 6))
plt.plot(comparison_df['Date'], comparison_df['Actual Closing Price'], label='Actual Closing Price', color='blue', linestyle='-', marker='o', alpha=0.8)
plt.plot(comparison_df['Date'], comparison_df['Predicted Closing Price'], label='Predicted Closing Price', color='red', linestyle='--', marker='x', alpha=0.5)
plt.title('Predicted vs Actual NASDAQ 100 Prices for Next 30 Days')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# Save the comparison to a CSV file
comparison_df.to_csv('comparison_next_30_days.csv', index=False)


# In[ ]:




