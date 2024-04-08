import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split

# Task 4
def noiseFiltering(inputFile):
    points = pd.read_csv(inputFile)

    # Removes the first and last column of the data frame
    data = points.iloc[:, 1:-1]

    window_size = 5
    sma5 = data.rolling(window_size).mean()

    sma5 = sma5.dropna()

    return sma5


filterWalking = noiseFiltering('./everyone walking/all walking.csv')

filterJumping = noiseFiltering('./everyone jumping/all jumping.csv')

# Task 5
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

    return features_df



walkingFeatures = features(filterWalking)
jumpingFeatures = features(filterJumping)


# Normalize the data after filtering it
def normalize(df, outputFile, labelValues):

    # Does preprocessing using standard scaler
    sc = preprocessing.StandardScaler()

    # Transform into numPy array
    df_transform = sc.fit_transform(df)

    # Changes back panda dataframe
    df_new = pd.DataFrame(df_transform)

    df_new['label'] = labelValues

    df_new.to_csv(outputFile, index=False)



normalize(jumpingFeatures,
          './featured normalized data/featured normalized jumping',
          1)

normalize(walkingFeatures,
          './featured normalized data/featured normalized walking',
          0)

def splitDataset(dataset_path, train_csv_path, test_csv_path):
    # Load the dataset into a DataFrame from the provided CSV file path
    dataset = pd.read_csv(dataset_path)

    # Split the dataset into training and testing subsets from library
    train_subset, test_subset = train_test_split(dataset, test_size=0.1, random_state=42)

    # Save the training subset to a CSV file
    train_subset.to_csv(train_csv_path, index=False)
    # Save the testing subset to a CSV file
    test_subset.to_csv(test_csv_path, index=False)


splitDataset('./featured normalized data/featured normalized jumping',
             './good training data/good training jumping.csv',
             './good testing data/good testing jumping.csv')

splitDataset('./featured normalized data/featured normalized walking',
             './good training data/good training walking.csv',
             './good testing data/good testing walking.csv')