from tkinter import *
from tkinter import filedialog

import pandas as pd
from sklearn import preprocessing

from task4and5 import noiseFiltering, features, normalize
from joblib import load

model = load("logregressionmodel.joblib")
def importFile():
    file_path = filedialog.askopenfilename(title="Select a file", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
    return file_path

def normalize(df):

    # Does preprocessing using standard scaler
    sc = preprocessing.StandardScaler()

    # Transform into numPy array
    df_transform = sc.fit_transform(df)

    # Changes back panda dataframe
    df_new = pd.DataFrame(df_transform)

    return normalize

def output():
    filepath = importFile()

    filteredData = noiseFiltering(filepath)

    filteredFeatures = features(filteredData)

    normalize(filteredFeatures)



window = Tk()

# Window name
window.title("Welcome to the Walking or Jumping Predictor")

# Window size
window.geometry('1000x1000')

# Main label when app opens
mainLabel = Label(window, text="Ready to predict whether you are Walking or Jumping?", font=("Times New Roman", 15, "bold"))
mainLabel.pack(pady=20)

# Button to select file
selectFile = Button(window, text="Select Input File", command=importFile)
selectFile.pack(pady=20)





# Run the Tkinter event loop
window.mainloop()
