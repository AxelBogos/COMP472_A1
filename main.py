# General Imports
import pandas as pd
import seaborn as sns
sns.set_style("darkgrid")
import matplotlib.pyplot as plt
import numpy as np

# Sklearn imports. Uncomment as you need, it gets long to run otherwise
from sklearn.metrics import precision_recall_fscore_support  # f1 by default
from sklearn.naive_bayes import GaussianNB
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import Perceptron
# from sklearn.neural_network import MLPClassifier

# Constant alphabet maps
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
    ax.axhline(df1[df1.columns[-1]].shape[0] / 26, color='red', label='Uniform Distribution')
    plt.legend()
    plt.show()
    ax = sns.countplot(df2[df2.columns[-1]].map(GREEK_ALPHABET), order=GREEK_ALPHABET.values(), palette='colorblind')
    ax.set(xlabel='Letters', ylabel='Count', title='Count Of Letters In Dataset 2')
    ax.axhline(df2[df2.columns[-1]].shape[0] / 10, color='red', label='Uniform Distribution')
    plt.legend()
    plt.show()

    return


def output_metrics_and_csv(y_pred,y_true,model_name,dataset_id):
    """ Description
    Compute metrics and output predictions of dataset to csv file
    Parameters
    ----------
    y_pred: nd.array
    y_true: nd.array
    model_name: string
    dataset_id: int
        y_pred, y_true: arrays of predicted classes and true classes for a dataset
        model_name: name of model result's being outputted
        dataset_id: id of dataset being outputted (1 or 2)

    Returns
    -------
    """

    # Get metrics TODO Clarify which are pertinent
    precision_per_class, recall_per_class, f1_per_class = precision_recall_fscore_support(y_true, y_pred)
    precision_weighted, recall_weighted, f1_weighted = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    precision_macro, recall_macro, f1_macro = precision_recall_fscore_support(y_true, y_pred,average='macro')

    # Prepare Result
    # TODO Confusion matrix is included in output csv? Formatting of metrics in output file?
    output_pred = zip(np.arange(y_pred.shape[0]), y_pred)  # In format (y_instance,prediction)

    #Output to CSV
    pd.DataFrame(output_pred).to_csv('%s-DS%d.csv' % (model_name, dataset_id), index=False, header=['y_instance', 'class'])


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
    y_pred = gnb.predict(np.array(df1_val)[:, :1024])
    y_true = df1_val[df1_val.columns[-1]]

    # Output predictions and metrics of dataset1 to CSV
    output_metrics_and_csv(y_pred, y_true,'GNB',1)


    # Apply model to dataset2
    X = df2_train[df2_train.columns[:-1]]
    Y = df2_train[df2_train.columns[-1]]
    gnb.fit(X, Y)

    # Get predictions and true labels of dataset2
    Y_pred = gnb.predict(np.array(df2_val)[:, :1024])
    Y_true = df2_val[df2_val.columns[-1]]

    # Output predictions and metrics of dataset2 to CSV
    output_metrics_and_csv(y_pred, y_true,'GNB',2)

    return


def Base_DT(train, val):
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

    # Output predictions and metrics of dataset1 to CSV

    # Apply model to dataset2

    # Get predictions and true labels of dataset2

    # Output predictions and metrics of dataset2 to CSV


def Best_DT(train, val):
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

    # Output predictions and metrics of dataset1 to CSV

    # Apply model to dataset2

    # Get predictions and true labels of dataset2

    # Output predictions and metrics of dataset2 to CSV

    return


def PER(train, val):
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

    # Output predictions and metrics of dataset1 to CSV

    # Apply model to dataset2

    # Get predictions and true labels of dataset2

    # Output predictions and metrics of dataset2 to CSV

    return


def Base_MLP(train, val):
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

    # Output predictions and metrics of dataset1 to CSV

    # Apply model to dataset2

    # Get predictions and true labels of dataset2

    # Output predictions and metrics of dataset2 to CSV

    return


def Best_MLP(train, val):
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

    # Output predictions and metrics of dataset2 to CSV

    # Apply model to dataset2

    # Get predictions and true labels of dataset2

    # Output predictions and metrics of dataset2 to CSV

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
