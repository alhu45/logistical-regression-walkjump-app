from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing

from joblib import load

def noiseFiltering(inputFile):

    # Removes the first and last column of the data frame
    data = inputFile.iloc[:, 1:]

    window_size = 5
    sma5 = data.rolling(window_size).mean()

    sma5 = sma5.dropna()

    return sma5


def features_normalize(df):
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

    # Does preprocessing using standard scaler
    # i changed this from StandardScaler to RobustScler
    sc = preprocessing.RobustScaler()
    # Transform into numPy array
    df_transform = sc.fit_transform(features_df)
    # Changes back panda dataframe
    df_new = pd.DataFrame(df_transform, columns=features_df.columns)

    # features_df.to_csv('./data to be tested/inputted data.csv')

    return df_new

def process_data(df):
    # process provided file
    df = noiseFiltering(df)

    df = features_normalize(df)

    print(df)

    # Use the model to predict the labels
    df = model.predict(df)

    predictions_df = pd.DataFrame(df, columns=['label'])
    predictions_df.to_csv("./data to be tested/labelled inputted data.csv", index=False)

    return df


def importFile():
    filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if filepath:
        # If a file is selected, process it
        print(f"Selected file: {filepath}")
        df = pd.read_csv(filepath)
        df = process_data(df)
    else:
        # If no file is selected, print a message
        print("No file was selected")

model = load("logregressionmodel.plk")

window = Tk()
window.title("Welcome to the Walking or Jumping Predictor")
window.configure(bg='#424242')
window.geometry('1000x1000')

# Load and display the image
#image_path = "/Users/Ryan/Desktop/Screenshot 2024-04-07 at 3.55.11 PM.png"
#img = Image.open(image_path)
#img = img.resize((500, 500), Image.Resampling.LANCZOS)
#photoImg = ImageTk.PhotoImage(img)
#imgLabel = Label(window, image=photoImg, bg='black')
#imgLabel.pack(pady=20)

mainLabel = Label(window, text="Ready to predict whether you are Walking or Jumping?",
                  font=("Times New Roman", 15, "bold"), bg='lightgrey', fg='black')
mainLabel.pack(pady=20)

selectFile = Button(window, text="Select Input File", command=importFile)
selectFile.pack(pady=20)

window.mainloop()

