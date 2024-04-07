import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import h5py

# input file for the unlabeled data, output file for the labeled data
def labelData(inputFile, outputFile, labelValues):
    # Reads the csv file using pandas, creating a dataframe called labeled_data
    labeled_Data = pd.read_csv(inputFile)

    # Adding columns to the data frame called label, assigning the value to 0 if it is walking data,
    # 1 if it is jumping data
    labeled_Data['label'] = labelValues

    # Saves the DataFrame with labels back to a new CSV file
    # The index = False
    labeled_Data.to_csv(outputFile, index=False)


# Ryan walking
labelData('./ryan walking/ryan walking.csv',
          './ryan walking/labelled ryan walking.csv', 0.0)

# Ryan jumping
labelData('./ryan jumping/ryan jumping.csv',
          './ryan jumping/labelled ryan jumping.csv', 1.0)

# Alan walking
labelData('./alan walking/alan walking.csv',
          './alan walking/labelled alan walking.csv', 0.0)

# Alan jumping
labelData('./alan jumping/alan jumping.csv',
          './alan jumping/labelled alan jumping.csv', 1.0)

# Kai walking
labelData('./kai walking/kai walking.csv',
          './kai walking/labelled kai walking.csv', 0.0)

# Kai jumping
labelData('./kai jumping/kai jumping.csv',
          './kai jumping/labelled kai jumping.csv', 1.0)


def shuffleData(inputFile, outputFile):
    # Reads the labeled data, stores it inside the shuffled_Data
    shuffled_Data = pd.read_csv(inputFile)

    # Creates a new column called window, divides the Time column by 5 and flooring it. This groups the time by into
    # 5 second windows (0.00s to 4.99s, 5.00s to 9.99s, etc)
    shuffled_Data['window'] = shuffled_Data['Time (s)'] // 5

    # Creates a list full of the 5 second windows.
    uniqueWindows = shuffled_Data['window'].unique()

    # Creating a copy of the uniqueWinodws, then shuffle it
    shuffledWindows = uniqueWindows.copy()
    np.random.shuffle(shuffledWindows)

    # Maps the two lists together
    mapping = dict(zip(uniqueWindows, shuffledWindows))
    shuffled_Data['window'] = shuffled_Data['window'].map(mapping)

    #Sorts the windows corresponding to its row
    shuffled_Data.sort_values(['window', 'Time (s)'], axis=0, ascending=[True, True], inplace=True)

    # Removes the column
    #shuffled_Data = shuffled_Data.iloc[:, :-1]

    # Saves the DataFrame with labels back to a new CSV file
    #shuffled_Data.to_csv(outputFile, index=False)


# Ryan walking
shuffleData('./ryan walking/labelled ryan walking.csv',
            './ryan walking/shuffled ryan walking.csv')

# Ryan jumping
shuffleData('./ryan jumping/labelled ryan jumping.csv',
            './ryan jumping/shuffled ryan jumping.csv')

# Alan walking
shuffleData('./alan walking/labelled alan walking.csv',
            './alan walking/shuffled alan walking.csv')

# Alan jumping
shuffleData('./alan jumping/labelled alan jumping.csv',
            './alan jumping/shuffled alan jumping.csv')

# Kai walking
shuffleData('./kai walking/labelled kai walking.csv',
            './kai walking/shuffled kai walking.csv')

# Kai jumping
shuffleData('./kai jumping/labelled kai jumping.csv',
            './kai jumping/shuffled kai jumping.csv')


def combineData(data1, data2, data3, outputFile):
    person1 = pd.read_csv(data1)
    person2 = pd.read_csv(data2)
    person3 = pd.read_csv(data3)

    # Concatenats all the data together into one column
    df_concatenated = pd.concat([person1, person2, person3], ignore_index=True)

    # Saves the DataFrame with labels back to a new CSV file
    df_concatenated.to_csv(outputFile, index=False)
    # print(df_concatenated)


combineData('./ryan walking/shuffled ryan walking.csv',
            './alan walking/shuffled alan walking.csv',
            './kai walking/shuffled kai walking.csv',
            './everyone walking/all walking.csv')

combineData('./ryan jumping/shuffled ryan jumping.csv',
            './alan jumping/shuffled alan jumping.csv',
            './kai jumping/shuffled kai jumping.csv',
            './everyone jumping/all jumping.csv')



def splitDataset(dataset_path, train_csv_path, test_csv_path):
    # Load the dataset into a DataFrame from the provided CSV file path
    dataset = pd.read_csv(dataset_path)

    # Split the dataset into training and testing subsets from library
    train_subset, test_subset = train_test_split(dataset, test_size=0.1, random_state=42)

    # Save the training subset to a CSV file
    train_subset.to_csv(train_csv_path, index=False)
    # Save the testing subset to a CSV file
    test_subset.to_csv(test_csv_path, index=False)


splitDataset('./everyone walking/all walking.csv',
             './training data/training walking.csv',
             './testing data/testing walking.csv')

splitDataset('./everyone jumping/all jumping.csv',
             './training data/training jumping.csv',
             './testing data/testing jumping.csv')

# Paths for input CSV files and output HDF5 file
walk_csv_files = {
    'ryan': './ryan walking/labelled ryan walking.csv',
    'alan': './alan walking/labelled alan walking.csv',
    'kai': './kai walking/labelled kai walking.csv'
}

jump_csv_files = {
    'ryan': './ryan jumping/labelled ryan jumping.csv',
    'alan': './alan jumping/labelled alan jumping.csv',
    'kai': './kai jumping/labelled kai jumping.csv'
}

training_csv_file = './training data/training walking.csv'
jumping_train_csv_file = './training data/training jumping.csv'

testing_csv_file = './testing data/testing walking.csv'
jumping_test_csv_file = './testing data/testing jumping.csv'

hdf5_output_file = './testing data/walkOrJump.csv'

# Understand what this does

# Read the CSV files into DataFrames
#walk and jump are two dictionaries being created
#each dictionary maps each group member to their corresponding dataframe (data from teh csv)
walk_dataframes = {name: pd.read_csv(filepath) for name, filepath in walk_csv_files.items()}
jump_dataframes = {name: pd.read_csv(filepath) for name, filepath in jump_csv_files.items()}

#seperates dataframes into training and testing for running and jumping
training_df = pd.read_csv(training_csv_file)
jumping_train_df = pd.read_csv(jumping_train_csv_file)

testing_df = pd.read_csv(testing_csv_file)
jumping_test_df = pd.read_csv(jumping_test_csv_file)

# Create an HDF5 file and write the datasets
with h5py.File('./hdf5_output_file', 'w') as hdf: #opens a new hdf5 file for writing
    # Creating groups for each participant
    for participant, walk_df in walk_dataframes.items(): #for each group member, a new group is created called data/{participant} for example data/ryan
        participant_group = hdf.create_group(f'data/{participant}') #within each part members group it creates new datasets for walking and jumping
        participant_group.create_dataset('walk', data=walk_df.to_numpy()) #this just converts acceleromater data into a numpy array
        # Assuming jump data for the same participant is present
        jump_df = jump_dataframes[participant]
        participant_group.create_dataset('jump', data=jump_df.to_numpy())


    # Creating groups for training and test data
    train_group = hdf.create_group('data/train')
    train_group.create_dataset('walkTrain', data=training_df.to_numpy())
    train_group.create_dataset('jumpTrain', data=jumping_train_df.to_numpy())

    test_group = hdf.create_group('data/test')
    test_group.create_dataset('walkTest', data=testing_df.to_numpy())
    test_group.create_dataset('jumpTest', data=jumping_test_df.to_numpy())

# Print data types of training DataFrames for verification
print(training_df.head())
print(jumping_train_df.head())

