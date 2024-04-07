import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load

# Load the trained logistic regression model
model = load("logregressionmodel.joblib")


def extract_features_efficient(data):
    # Define aggregation mappings for features
    agg_map = {
        'Linear Acceleration x (m/s^2)': ['mean', 'std', 'var', 'max', 'min'],
        'Linear Acceleration y (m/s^2)': ['mean', 'std', 'var', 'max', 'min'],
        'Linear Acceleration z (m/s^2)': ['mean', 'std', 'var', 'max', 'min'],
        'Absolute acceleration (m/s^2)': ['mean', 'std', 'var', 'max', 'min']
    }

    # Aggregate data by 'ID' and calculate features
    features = data.groupby('ID').agg(agg_map)
    features.columns = ['_'.join(col).strip().replace(' ', '_').replace('(', '').replace(')', '') for col in
                        features.columns.values]

    # Calculate ranges for each metric
    for axis in ['Linear_Acceleration_x', 'Linear_Acceleration_y', 'Linear_Acceleration_z', 'Absolute_acceleration']:
        max_col = f'{axis}_m/s^2_max'
        min_col = f'{axis}_m/s^2_min'
        features[f'{axis}_range'] = features[max_col] - features[min_col]

    return features


def process_data(filepath):
    # Load the CSV file into a DataFrame
    raw_data = pd.read_csv(filepath)
    data = raw_data[((raw_data["Time (s)"]) >= 2.5) & ((raw_data["Time (s)"]) <= 97.5)]

    # Assigning 'ID' to segments based on time intervals
    start_time = 2.5
    end_time = 7.5
    interval = 5
    current_id = 1
    data['ID'] = None

    for index, row in data.iterrows():
        if row['Time (s)'] < end_time:
            data.at[index, 'ID'] = current_id
        else:
            current_id += 1
            start_time += interval
            end_time += interval
            data.at[index, 'ID'] = current_id

    # Smooth the data using a rolling window
    data = data.groupby('ID').apply(lambda x: x.rolling(window=5, min_periods=1).mean())
    data = data.reset_index(drop=True)

    # Extract features
    features_df = extract_features_efficient(data)

    return features_df


def plot_data(predictions):
    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(predictions)), predictions, c=predictions, cmap='viridis', alpha=0.5)
    plt.colorbar(ticks=[0, 1], label='Activity (0-Walking, 1-Jumping)')
    plt.title('Activity Predictions')
    plt.xlabel('Sample Index')
    plt.ylabel('Predicted Activity')
    plt.show()


def open_file():
    filepath = filedialog.askopenfilename(title="Open CSV file",
                                          filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
    if filepath:
        try:
            processed_data = process_data(filepath)
            predictions = model.predict(processed_data)
            plot_data(predictions)
        except Exception as e:
            messagebox.showerror("Error", "Failed to process the file\n" + str(e))
    else:
        messagebox.showinfo("Canceled", "File selection canceled")


root = tk.Tk()
root.title("Activity Classifier Application")
root.geometry('300x150')

open_button = tk.Button(root, text="Open CSV and Classify", command=open_file)
open_button.pack(expand=True)

root.mainloop()