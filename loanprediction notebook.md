
# Gettting system ready and loading the data.
Getting system ready and loading data.
In this section i will load the neccesary packages in pytho for data analysis.I will also import my data and get it ready for scruitiny.

## Importing Packages


```python
import pandas as pd
import numpy as np
#import searborn as np
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
```

## Reading Data



```python
train = pd.read_csv('/Users/admin/Downloads/train.csv')
test = pd.read_csv('/Users/admin/Downloads/test.csv')
```

It is important to have a copy of the original data so that any changes made in the dataset will not affect the orriginall file.


```python
train_original = train.copy()
test_original = test.copy()
```

# Understanding the Data
In this section i will attempt to understand the shape of my data sets.The names of different features ,the dimension and any other thing of interest.


```python
train.columns
```




    Index([u'Loan_ID', u'Gender', u'Married', u'Dependents', u'Education',
           u'Self_Employed', u'ApplicantIncome', u'CoapplicantIncome',
           u'LoanAmount', u'Loan_Amount_Term', u'Credit_History', u'Property_Area',
           u'Loan_Status'],
          dtype='object')



The train dataset has 12 explanatory variables and one taerget variable ,i.e Loan Status .


```python
test.columns
```




    Index([u'Loan_ID', u'Gender', u'Married', u'Dependents', u'Education',
           u'Self_Employed', u'ApplicantIncome', u'CoapplicantIncome',
           u'LoanAmount', u'Loan_Amount_Term', u'Credit_History',
           u'Property_Area'],
          dtype='object')



The train data set has all the exlplanatory variable except one predicted variable.


```python
#Printing data types for each variable.
train.dtypes
```




    Loan_ID               object
    Gender                object
    Married               object
    Dependents            object
    Education             object
    Self_Employed         object
    ApplicantIncome        int64
    CoapplicantIncome    float64
    LoanAmount           float64
    Loan_Amount_Term     float64
    Credit_History       float64
    Property_Area         object
    Loan_Status           object
    dtype: object



Investigating the shape of the data set


```python
train.shape,test.shape
```




    ((614, 13), (367, 12))



The train data set has 614 rows and 13 columns while the test data set has 367 rows and 12 columns.

# Univariate Analysis
This section involves analysis of every varuable individually.

## Target Variable
Friequency table is best for this variable as it gives count values per category.


```python
train['Loan_Status'].value_counts()
```




    Y    422
    N    192
    Name: Loan_Status, dtype: int64




```python
# Setting normalize to True to print propotions instead.
train['Loan_Status'].value_counts(normalize = True)*100
counts = train['Loan_Status'].value_counts(normalize = True)*100
counts.plot.bar(title = "Loan Status")
```




    Y    68.729642
    N    31.270358
    Name: Loan_Status, dtype: float64






    <matplotlib.axes._subplots.AxesSubplot at 0x118f88b50>




![png](output_19_2.png)


## Independent variables(Categorical)


```python
plt.figure(1)
plt.subplot(221)
train['Gender'].value_counts(normalize = True).plot.bar(figsize = (20,10),title = "Gender")

plt.subplot(222)
train['Married'].value_counts(normalize = True).plot.bar(title = "Married")

plt.subplot(223)
train['Self_Employed'].value_counts(normalize = True).plot.bar(title = "Self_Employed")

plt.subplot(224)
train['Credit_History'].value_counts(normalize = True).plot.bar(title = "Credit_History")

plt.show()
```




    <Figure size 432x288 with 0 Axes>






    <matplotlib.axes._subplots.AxesSubplot at 0x119d1c910>






    <matplotlib.axes._subplots.AxesSubplot at 0x119d1c910>






    <matplotlib.axes._subplots.AxesSubplot at 0x119f74fd0>






    <matplotlib.axes._subplots.AxesSubplot at 0x119f74fd0>






    <matplotlib.axes._subplots.AxesSubplot at 0x119fcc910>






    <matplotlib.axes._subplots.AxesSubplot at 0x119fcc910>






    <matplotlib.axes._subplots.AxesSubplot at 0x11a02fad0>






    <matplotlib.axes._subplots.AxesSubplot at 0x11a02fad0>




![png](output_21_9.png)


* We can see that 80% of the applicants in the test data set are males.
* About 65% of the applicants are married.
* 15% applicants are self employed.
* 85% have repaid their debts.

## Independent Variable(Ordinal)


```python
plt.figure(1)
plt.subplot(131)
train['Dependents'].value_counts(normalize = True).plot.bar(title = 'Dependents')

plt.subplot(132)
train['Education'].value_counts(normalize = True).plot.bar(title = 'Education')

plt.subplot(133)
train['Property_Area'].value_counts(normalize = True).plot.bar(title = 'Property_Area')

```




    <Figure size 432x288 with 0 Axes>






    <matplotlib.axes._subplots.AxesSubplot at 0x123117910>






    <matplotlib.axes._subplots.AxesSubplot at 0x123117910>






    <matplotlib.axes._subplots.AxesSubplot at 0x123391e90>






    <matplotlib.axes._subplots.AxesSubplot at 0x123391e90>






    <matplotlib.axes._subplots.AxesSubplot at 0x12342c690>






    <matplotlib.axes._subplots.AxesSubplot at 0x12342c690>




![png](output_24_7.png)


## Independent Variable(Numerical)


```python
plt.figure(1)
plt.subplot(1,2,1)
#sns.distplot(train['Applicant Income'])

plt.subplot(1,2,2)
train['ApplicantIncome'].plot.box()
```




    <Figure size 432x288 with 0 Axes>






    <matplotlib.axes._subplots.AxesSubplot at 0x123622210>






    <matplotlib.axes._subplots.AxesSubplot at 0x122ede210>






    <matplotlib.axes._subplots.AxesSubplot at 0x122ede210>




![png](output_26_4.png)


The box plot suggests prescence of outliers in our data and the density plot suggest a left tailed screnario,


```python
#Segregate tye income eaners by edycation.
train.boxplot(column = 'ApplicantIncome',by ='Education')

plt.suptitle('')
Text(.5,.98,'')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x124262910>






    Text(0.5,0.98,'')




    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-90-dc1a1d0736bd> in <module>()
          3 
          4 plt.suptitle('')
    ----> 5 Text(.5,.98,'')
    

    NameError: name 'Text' is not defined



![png](output_28_3.png)

