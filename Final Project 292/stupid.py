from tkinter import *
from tkinter import filedialog, messagebox

import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing

from joblib import load

model = load("logregressionmodel.plk")

def noiseFiltering(inputFile):

    # Removes the first and last column of the data frame
    data = inputFile.iloc[:, 1:]

    window_size = 5
    sma5 = data.rolling(window_size).mean()

    sma5 = sma5.dropna()

    return sma5

def features(df):
    segment_length = 500 # Process every 500 data points every 5 Seconds

    # Empty list to hold features
    features = []

    for start_index in range(0, len(df), segment_length):
        segment_features = {}

        for i, col_name in enumerate(['x', 'y', 'z', 'absolute']):
            # Calculate features for each column
            segment = df.iloc[start_index:start_index + segment_length, i]

            segment_features[f'mean {col_name}'] = segment.mean()
            segment_features[f'max {col_name}'] = segment.max()
            segment_features[f'min {col_name}'] = segment.min()
            segment_features[f'median {col_name}'] = segment.median()
            segment_features[f'std {col_name}'] = segment.std()
            segment_features[f'skew {col_name}'] = segment.skew()
            segment_features[f'kurtosis {col_name}'] = segment.kurt()
            segment_features[f'variance {col_name}'] = segment.var()
            segment_features[f'sum {col_name}'] = segment.sum()

            # Append the features of this segment to the features list
            features.append(segment_features)


    #Make the feature list into a dataframe
    features_df = pd.DataFrame(features)
    features_df = features_df.dropna()
    features_df.to_csv('./data to be tested/inputted data.csv')

    return features_df

def process_data(df):
    # process provided file
    df = noiseFiltering(df)

    df = features(df)

    # Use the model to predict the labels
    df = model.predict(df)

    predictions_df = pd.DataFrame(df, columns=['Predictions'])
    predictions_df.to_csv("./data to be tested/labelled inputted data.csv", index=False)

    return df


def importFile():
    filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if filepath:
        # Now you can use the selected CSV file path
        print(f"Selected file: {filepath}")
    else:
        print("No file was selected")
        df = pd.read_csv(filepath)
        df = process_data(df)


window = Tk()

# Window name
window.title("Welcome to the Walking or Jumping Predictor")

# Window size
window.geometry('500x500')

# Main label when app opens
mainLabel = Label(window, text="Ready to predict whether you are Walking or Jumping?", font=("Times New Roman", 15, "bold"))
mainLabel.pack(pady=20)

# Button to select file
selectFile = Button(window, text="Select Input File", command=importFile)
selectFile.pack(pady=20)

# Run the Tkinter event loop
window.mainloop()