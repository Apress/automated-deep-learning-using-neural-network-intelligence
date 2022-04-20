import os
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def telescope_dataset():
    """
    Telescope Dataset
    https://archive.ics.uci.edu/ml/datasets/magic+gamma+telescope

    Binary classification problem
    """
    cd = os.path.dirname(os.path.abspath(__file__))
    telescope_df = pd.read_csv(f'{cd}/data/magic04.data')
    telescope_df.dropna(inplace = True)
    telescope_df.columns = [
        'fLength', 'fWidth', 'fSize', 'fConc', 'fConcl',
        'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'class']

    # Shuffling Dataset
    telescope_df = telescope_df.iloc[np.random.permutation(len(telescope_df))]
    telescope_df.reset_index(drop = True, inplace = True)

    # Class Labeling
    telescope_df['class'] = telescope_df['class'].map({'g': 0, 'h': 1})
    y = telescope_df['class'].values

    # Train / Test Split
    train_ind, test_ind = train_test_split(
        telescope_df.index,
        stratify = y,
        train_size = 0.8,
        test_size = 0.2
    )

    X_train = telescope_df.drop('class', axis = 1).loc[train_ind].values
    X_test = telescope_df.drop('class', axis = 1).loc[test_ind].values

    y_train = telescope_df.loc[train_ind, 'class'].values
    y_test = telescope_df.loc[test_ind, 'class'].values

    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    telescope_dataset()
