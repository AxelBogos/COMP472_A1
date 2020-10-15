# :beers: COMP 472 – Assignment 1 :tiger:

---

Axel Bogos - 40077502 <br>
Luan Meiyi - 40047658 <br>
Xavier Morin - 40077865

---

## Preliminary Information

---
#### Libraries Used:
* pandas
* seaborn
* matplotlib
* numpy
* sklearn 
---

## How To Run 

---

Execute the main() function of ```main.py```. This will execute the following functions and model calls, in that order: 
```python
    plot_data(df_train)
    GNB(df_train,df_val)
    Base_DT(df_train,df_val)
    Best_DT(df_train,df_val)
    PER(df_train, df_val)
    Base_MLP(df_train, df_val)
    Best_MLP(df_train, df_val)
```
where each the plot function and each model call generate the output files in directory 'results' with the following structure
```
.
│   main.py
│   __init__.py    
│
└───results
│   │   GNB-DS1.csv
│   │   GNB-DS2.csv
│   │   Base-DT-DS1.csv
│   │   Base-DT-DS2.csv
│   │   Best-DT-DS1.csv
│   │   Best-DT-DS2.csv
│   │   PER-DS1.csv
│   │   PER-DS2.csv
│   │   Base-MLP-DS1.csv
│   │   Base-MLP-DS2.csv
│   │   Best-MLP-DS1.csv
│   │   Best-MLP-DS2.csv
│   
└───data
    │   ...
```
---
