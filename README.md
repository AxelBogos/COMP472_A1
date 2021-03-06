# :beers: COMP 472 – Assignment 1 :tiger:

[Repo URL](https://github.com/AxelBogos/COMP472_A1) <br>
[Google Colab Notebook](https://colab.research.google.com/drive/1z5IqdQMRNb3Uyo8YO0zG7aFrire5tizM?usp=sharing)
---

Axel Bogos - 40077502 <br>
Luan Meiyi - 40047658 <br>
Xavier Morin - 40077865

---

## Preliminary Information

#### Libraries Used:
* pandas
* seaborn
* matplotlib
* numpy
* sklearn 
---

## How To Run 

---

Execute the main() function of ```main.py``` (also accessible with [this](https://colab.research.google.com/drive/1z5IqdQMRNb3Uyo8YO0zG7aFrire5tizM?usp=sharing) Colab notebook). The main() function will execute the following functions and model calls, in that order: 
```python
    plot_data(df_train)
    GNB(df_train,df_val)
    Base_DT(df_train,df_val)
    Best_DT(df_train,df_val)
    PER(df_train, df_val)
    Base_MLP(df_train, df_val)
    Best_MLP(df_train, df_val)
```
The following are generated in the *result* directory: 

1. 6 plots of the train, val and test data distribution for both datasets named *{train,val,test}dataset#_plot.png* 

2. For every model/dataset pair, a directory named *model-name_dataset#* is created and the following are generated within it: 

   2.1 A *model-name_dataset#*.csv formatted in that order
      * 1 ```Instance,Predicted``` line per instance of test dataset
      * 1 ```Precision,Recall, F-1``` line per class of test dataset
      * 1 ```Accuracy, Macro F-1, Weighted F-1``` line.
    
   2.2 A *model-name-dataset#_Confusion_Matrix*.png file of the confusion matrix plot. 

These files are organized in a directory structure as follows: 
```
.
│ main.py
│ __init__.py    
│ README.md
│
└───results
│   │
│   │ train1-plot.png
│   │ train2-plot.png
│   │ val1-plot.png
│   │ val2-plot.png
│   │ test1-plot.png
│   │ test2-plot.png
│   │
│   └───GNB-DS1
|   |   | GNB-DS1.csv
|   |   | GNB-DS1-Confusion_Matrix.png
|   |
│   └───GNB-DS2.csv
|   |   | GNB-DS2.csv
|   |   | GNB-DS2-Confusion_Matrix.png
|   |   
│   └───Base-DT-DS1
|   |   | Base-DT-DS1.csv
|   |   | Base-DT-DS1-Confusion_Matrix.png
|   |   
│   └───Base-DT-DS2
|   |   | Base-DT-DS2.csv
|   |   | Base-DT-DS2-Confusion_Matrix.png
|   |   
│   └───Best-DT-DS1
|   |   | Best-DT-DS1.csv
|   |   | Best-DT-DS1-Confusion_Matrix.png
|   |   
│   └───Best-DT-DS2
|   |   | Best-DT-DS2.csv
|   |   | Best-DT-DS2-Confusion_Matrix.png
|   |   
│   └───PER-DS1
|   |   | PER-DS1.csv
|   |   | PER-DS1-Confusion_Matrix.png
|   |   
│   └───PER-DS2
|   |   | PER-DS1.csv
|   |   | PER-DS1-Confusion_Matrix.png
|   |   
│   └───Base-MLP-DS1
|   |   | Base-MLP-DS1.csv
|   |   | Base-MLP-DS1-Confusion_Matrix.png
|   |   
│   └───Base-MLP-DS2
|   |   | Base-MLP-DS2.csv
|   |   | Base-MLP-DS2-Confusion_Matrix.png
|   |   
│   └───Best-MLP-DS1
|   |   | Best-MLP-DS1.csv
|   |   | Best-MLP-DS1-Confusion_Matrix.png
|   |   
│   └───Best-MLP-DS2
|       | Best-MLP-DS2.csv
|       | Best-MLP-DS2-Confusion_Matrix.png
|   
└───data
    │   ...
```
---
