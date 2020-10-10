# Imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_fscore_support  # f1 by default
sns.set_style("darkgrid")
LATIN_ALPHABET = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
                  12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
                  23: 'X', 24: 'Y', 25: 'Z'}
GREEK_ALPHABET = {0: '\u03C0', 1: '\u03B1', 2: '\u03B2', 3: '\u03C3', 4: '\u03B3', 5: '\u03B4', 6: '\u03BB',
                  7: '\u03C9', 8: '\u03BC', 9: '\u03BE'}

def plot_data(data):
    """ Description

    Parameters
    ----------
    data: tuple of pd.DataFrame
        Tuple containing a tuple of 2 datasets as pd.Dataframe

    Returns
    -------
    """
    df1, df2 = data
    plt.figure(figsize=(12, 8))
    ax = sns.countplot(df1[df1.columns[-1]].map(LATIN_ALPHABET), order=LATIN_ALPHABET.values(), palette='colorblind')
    ax.set(xlabel='Letters', ylabel='Count', title='Count Of Letters In Dataset 1')
    plt.show()
    ax = sns.countplot(df2[df2.columns[-1]].map(GREEK_ALPHABET), order=GREEK_ALPHABET.values(), palette='colorblind')
    ax.set(xlabel='Letters', ylabel='Count', title='Count Of Letters In Dataset 2')
    plt.show()

    return


def output_to_csv(data):
    """ Description

    Parameters
    ----------
    data: tuple
        Tuple of (y-instance, prediction)

    Returns
    -------
    """
    return


def GNB(train, val):
    from sklearn.naive_bayes import GaussianNB
    """ Description
    Parameters
    ----------
    train: tuple of pd.DataFrame
    val: tuple of pd.DataFrame
    Returns
    -------
    """

    # Unpack datasets
    df1_train, df2_train = train
    df1_val, df2_val = val

    # Define Model
    gnb = GaussianNB()

    # Apply model to dataset1
    X = df1_train[df1_train.columns[:-1]]
    Y = df1_train[df1_train.columns[-1]]
    gnb.fit(X, Y)

    # Get predictions and true labels of dataset1
    Y_pred = gnb.predict(np.array(df1_val)[:, :1024])
    Y_true = df1_val[df1_val.columns[-1]]

    # Get metrics of dataset1
    # TODO Clarify which are pertinent
    precision_per_class, recall_per_class, f1_per_class = precision_recall_fscore_support(Y_true, Y_pred)
    precision_weighted, recall_weighted, f1_weighted = precision_recall_fscore_support(Y_true, Y_pred, average='weighted')
    precision_macro, recall_macro, f1_macro = precision_recall_fscore_support(Y_true, Y_pred,average='macro')

    # Output result of dataset1
    # TODO Confusion matrix is included in output csv? Formatting of metrics in output file?
    output_pred = zip(np.arange(Y_pred.shape[0]), Y_pred)  # In format (y_instance,prediction)
    output_to_csv(output_pred)

    # Apply model to dataset2
    X = df2_train[df2_train.columns[:-1]]
    Y = df2_train[df2_train.columns[-1]]
    gnb.fit(X, Y)

    # Get predictions and true labels of dataset2
    Y_pred = gnb.predict(np.array(df2_val)[:, :1024])
    Y_true = df2_val[df2_val.columns[-1]]

    # Get metrics of dataset2
    # TODO Clarify which are pertinent
    precision_per_class, recall_per_class, f1_per_class = precision_recall_fscore_support(Y_true, Y_pred)
    precision_weighted, recall_weighted, f1_weighted = precision_recall_fscore_support(Y_true, Y_pred, average='weighted')
    precision_macro, recall_macro, f1_macro = precision_recall_fscore_support(Y_true, Y_pred,average='macro')

    # Output result of dataset2
    # TODO Confusion matrix is included in output csv? Formatting of metrics in output file?
    output_pred = zip(np.arange(Y_pred.shape[0]), Y_pred)  # In format (y_instance,prediction)
    output_to_csv(output_pred)

    return


def Base_DT(train, val):
    import sklearn.tree
    """ Description

    Parameters
    ----------
    train: tuple of pd.DataFrame
    val: tuple of pd.DataFrame
    Returns
    -------
    """

    # Unpack datasets
    df1_train, df2_train = train
    df1_val, df2_val = val

    # Define Model

    # Apply model to dataset1

    # Get predictions and true labels of dataset1

    # Get metrics of dataset1

    # Output result of dataset1

    # Apply model to dataset2

    # Get predictions and true labels of dataset2

    # Get metrics of dataset2

    # Output result of dataset2


def Best_DT(train, val):
    import sklearn.tree
    """ Description

    Parameters
    ----------
    train: tuple of pd.DataFrame
    val: tuple of pd.DataFrame    Tuple containing a tuple of 2 datasets as pd.Dataframe
    Returns
    -------
    """

    # Unpack datasets
    df1_train, df2_train = train
    df1_val, df2_val = val

    # Define Model

    # Apply model to dataset1

    # Get predictions and true labels of dataset1

    # Get metrics of dataset1

    # Output result of dataset1

    # Apply model to dataset2

    # Get predictions and true labels of dataset2

    # Get metrics of dataset2

    # Output result of dataset2

    return


def PER(train, val):
    import sklearn.linear_model.perceptron
    """ Description

    Parameters
    ----------
    train: tuple of pd.DataFrame
    val: tuple of pd.DataFrame
    Returns
    -------
    """

    # Unpack datasets
    df1_train, df2_train = train
    df1_val, df2_val = val

    # Define Model

    # Apply model to dataset1

    # Get predictions and true labels of dataset1

    # Get metrics of dataset1

    # Output result of dataset1

    # Apply model to dataset2

    # Get predictions and true labels of dataset2

    # Get metrics of dataset2

    # Output result of dataset2

    return


def Base_MLP(train, val):
    import sklearn.neural_network.multilayer_perceptron
    """ Description

    Parameters
    ----------
    train: tuple of pd.DataFrame
    val: tuple of pd.DataFrame
    Returns
    -------
    """

    # Unpack datasets
    df1_train, df2_train = train
    df1_val, df2_val = val

    # Define Model

    # Apply model to dataset1

    # Get predictions and true labels of dataset1

    # Get metrics of dataset1

    # Output result of dataset1

    # Apply model to dataset2

    # Get predictions and true labels of dataset2

    # Get metrics of dataset2

    # Output result of dataset2

    return


def Best_MLP(train, val):
    import sklearn.neural_network.multilayer_perceptron
    """ Description

    Parameters
    ----------
    train: tuple of pd.DataFrame
    val: tuple of pd.DataFrame
    Returns
    -------
    """

    # Unpack datasets
    df1_train, df2_train = train
    df1_val, df2_val = val

    # Define Model

    # Apply model to dataset1

    # Get predictions and true labels of dataset1

    # Get metrics of dataset1

    # Output result of dataset1

    # Apply model to dataset2

    # Get predictions and true labels of dataset2

    # Get metrics of dataset2

    # Output result of dataset2

    return


def main():
    # Load datasets splits as tuples of each type
    df_train = (pd.read_csv('data/train_1.csv'), pd.read_csv('data/train_2.csv'))
    df_tests_no_label = (pd.read_csv('data/test_no_label_1.csv'), pd.read_csv('data/test_no_label_2.csv'))
    df_tests_with_label = (pd.read_csv('data/test_with_label_1.csv'), pd.read_csv('data/test_with_label_2.csv'))
    df_val = (pd.read_csv('data/val_1.csv'), pd.read_csv('data/val_2.csv'))

    # Plot instance distribution
    plot_data(df_train)

    # Run models

    return


if __name__ == '__main__':
    main()
