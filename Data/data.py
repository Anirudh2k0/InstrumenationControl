import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

def generate_decreasing_data_step(num_values, initial_value, decrease_start=random.randint(250,400)):
  data = []
  for i in range(num_values):
    if i < decrease_start:
      data.append(initial_value)
    else:
      # Calculate linear decrease based on remaining values
      slope = -(initial_value) / (num_values - decrease_start)
      value = initial_value + slope * (i - decrease_start)
      data.append(max(value, 0))  # Ensure value doesn't go below zero
  return data

# def generate_decreasing_data_step(num_values, initial_value, decrease_start=random.randint(250,500), min_value=5, max_value=20):
#     data = []
#     for i in range(num_values):
#         if i < decrease_start:
#             data.append(initial_value)
#         else:
#             # Calculate linear decrease based on remaining values
#             slope = -(initial_value - min_value) / (num_values - decrease_start)
#             value = initial_value + slope * (i - decrease_start)
#             data.append(min(min(max_value, value), min_value))  # Ensure value is within the specified range
#     return data



n_col = 1000
n_exp = 10
data = []

for i in range(1,n_exp+1):
  temperature = list(range(575, 850, 25))[:10]
  voltage = [x / 10.0 for x in range(1, 11)]
  pressure = sorted(random.sample(range(10, 20), n_exp))
  timeStamp = list(range(2, 2001, 2))
  start_gr = sorted(random.sample(range(50,150),n_exp))
  
  for j in range(n_col):
    growth_rate = generate_decreasing_data_step(n_col,start_gr[i-1])
    # max_gr = max(growth_rate)
    # min_gr = min(growth_rate)
    # growth_rate = [(sample_mat - min_gr / (max_gr - min_gr) * (20 - 5) + 5) for sample_mat in growth_rate]
    data.append({'Experiment': i,
              'Temperature': temperature[i - 1],
              'GrowthRate': growth_rate[j],
              'Voltage': voltage[i - 1],
              'Pressure': pressure[i-1],
              'Time': timeStamp[j]})

df = pd.DataFrame(data)
df.to_csv('Data/Synthetic Data.csv',index=False, encoding='utf-8')


for exp_id in range(1, n_exp + 1):
    exp_df = df[df['Experiment'] == exp_id]
    plt.figure(figsize=(12, 5))
    plt.plot(exp_df['Time'], exp_df['GrowthRate'], marker='o')
    plt.xlabel('Time')
    plt.ylabel('Growth Rate')
    plt.text(0.1,0.5,f"Temperature: {exp_df['Temperature'].iloc[0]}")
    plt.title(f'Experiment {exp_id}: Growth Rate vs. Time')
    plt.show()
