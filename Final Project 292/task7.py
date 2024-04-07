from tkinter import Tk, Button
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import pandas as pd
from joblib import load
from sklearn import preprocessing

model = load("logregressionmodel.joblib")


def reduce_noise_and_normalize(indf):
    window_size = 5
    data = indf
    timestamps = data.iloc[:, 0]
    data = data.iloc[:, 1:]
    filtered_data = data.rolling(window=window_size, min_periods=1).mean()
    sc = preprocessing.StandardScaler()
    normalized_data = sc.fit_transform(filtered_data)
    normalized_df = pd.DataFrame(normalized_data, columns=data.columns)

    final_df = pd.concat([timestamps, normalized_df], axis=1)
    final_df = final_df.dropna()
    return final_df


def extract_features_and_normalize(indf):
    window_size = 5
    data = indf
    # labels = data.iloc[:, -2]
    features = pd.DataFrame()

    for i in range(1, data.shape[1]):
        column_features = pd.DataFrame()
        column_features[f'mean.{i}'] = data.iloc[:, i].rolling(window=window_size, min_periods=1).mean()
        column_features[f'std.{i}'] = data.iloc[:, i].rolling(window=window_size, min_periods=1).std()
        column_features[f'max.{i}'] = data.iloc[:, i].rolling(window=window_size, min_periods=1).max()
        column_features[f'min.{i}'] = data.iloc[:, i].rolling(window=window_size, min_periods=1).min()
        column_features[f'kurtosis.{i}'] = data.iloc[:, i].rolling(window=window_size, min_periods=1).kurt()
        column_features[f'skew.{i}'] = data.iloc[:, i].rolling(window=window_size, min_periods=1).skew()
        column_features[f'range.{i}'] = column_features[f'max.{i}'] - column_features[f'min.{i}']
        column_features[f'variance.{i}'] = data.iloc[:, i].rolling(window=window_size, min_periods=1).var()
        column_features[f'median.{i}'] = data.iloc[:, i].rolling(window=window_size, min_periods=1).median()
        column_features[f'Sum.{i}'] = data.iloc[:, i].rolling(window=window_size, min_periods=1).sum()
        column_features[f'Standard Error.{i}'] = data.iloc[:, i].rolling(window=window_size, min_periods=1).sem()
        column_features[f'Exponential Moving Average.{i}'] = data.iloc[:, i].ewm(span=window_size).mean()
        features = pd.concat([features, column_features], axis=1)

    # Normalize the features using z-score normalization
    normalized_features = (features - features.mean()) / features.std()
    # final_data = pd.concat([normalized_features, labels.reset_index(drop=True)], axis=1)
    final_data = normalized_features.dropna()
    final_data.to_csv('./data to be tested/inputted data.csv')

    return final_data


def process_data(df):
    # process provided file
    df = reduce_noise_and_normalize(df)
    df = extract_features_and_normalize(df)

    # Use the model to predict the labels
    df = model.predict(df)
    predictions_df = pd.DataFrame(df, columns=['Predictions'])
    predictions_df.to_csv("./data to be tested/labelled inputted data.csv", index=False)

    return df


def open_csv_file():
    filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if filepath:
        # Now you can use the selected CSV file path
        print(f"Selected file: {filepath}")
    else:
        print("No file was selected")
    df = pd.read_csv(filepath)
    df = process_data(df)

    # Save the DataFrame to a new CSV file
    # df.to_csv('C:/Users/Simon/Downloads/Year_2/sem2/ELEC_292/Assignment/labeledInputData.csv', index=False)


box = Tk()
box.geometry("800x600")  # Set window size

# Create a button that calls 'open_csv_file' when clicked
b = Button(box, text="Open CSV File", command=open_csv_file)
b.place(x=375, y=250)

box.mainloop()