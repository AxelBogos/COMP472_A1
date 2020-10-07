# Imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
sns.set_style("darkgrid")
LATIN_ALPHABET = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
                12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
                23: 'X', 24: 'Y', 25: 'Z'}
GREEK_ALPHABET = {0: '\u03C0', 1: '\u03B1', 2: '\u03B2', 3: '\u03C3', 4: '\u03B3', 5: '\u03B4', 6: '\u03BB', 7: '\u03C9', 8: '\u03BC', 9: '\u03BE'}

def plot_data(data):
    """ Description

    Parameters
    ----------
    data: tuple of pd.DataFrame
        Tuple containing a tuple of 2 datasets as pd.Dataframe

    Returns
    -------
    """
    df1,df2=data
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
    data: tuple of pd.DataFrame
        Tuple containing a tuple of 2 datasets as pd.Dataframe

    Returns
    -------
    """
    return

def GNB(data):
    import sklearn.naive_bayes
    """ Description

    Parameters
    ----------
    data: tuple of pd.DataFrame
        Tuple containing a tuple of 2 datasets as pd.Dataframe

    Returns
    -------
    """
    return

def Base_DT(data):
    import sklearn.tree
    """ Description

    Parameters
    ----------
    data: tuple of pd.DataFrame
        Tuple containing a tuple of 2 datasets as pd.Dataframe

    Returns
    -------
    """
    return

def Best_DT(data):
    import sklearn.tree
    """ Description

    Parameters
    ----------
    data: tuple of pd.DataFrame
        Tuple containing a tuple of 2 datasets as pd.Dataframe

    Returns
    -------
    """
    return

def PER(data):
    import sklearn.linear_model.perceptron
    """ Description

    Parameters
    ----------
    data: tuple of pd.DataFrame
        Tuple containing a tuple of 2 datasets as pd.Dataframe

    Returns
    -------
    """
    return

def Base_MLP(data):
    import sklearn.neural_network.multilayer_perceptron
    """ Description

    Parameters
    ----------
    data: tuple of pd.DataFrame
        Tuple containing a tuple of 2 datasets as pd.Dataframe

    Returns
    -------
    """
    return

def Best_MLP(data):
    import sklearn.neural_network.multilayer_perceptron
    """ Description

    Parameters
    ----------
    data: tuple of pd.DataFrame
        Tuple containing a tuple of 2 datasets as pd.Dataframe

    Returns
    -------
    """
    return

def main():
    #Load datasets splits as tuples of each type
    df_train = (pd.read_csv('data/train_1.csv'), pd.read_csv('data/train_2.csv'))
    df_tests_no_label = (pd.read_csv('data/test_no_label_1.csv'), pd.read_csv('data/test_no_label_2.csv'))
    df_tests_with_label = (pd.read_csv('data/test_with_label_1.csv'), pd.read_csv('data/test_with_label_2.csv'))
    df_val = (pd.read_csv('data/val_1.csv'), pd.read_csv('data/val_2.csv'))

    #Plot instance distribution
    plot_data(df_train)
    
    #Run models

    #output results

    return

if __name__ == '__main__':
    main()
