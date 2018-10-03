
# Problem Statement
This is a loab approvement problem.I am supposd to predict whether a loan will be approved or not.I am going to answer this question by performing thorough feature ingeneering to see the relationships within the explantory variables.

# Gettting system ready and loading the data.
Getting system ready and loading data.
In this section i will load the neccesary packages in pytho for data analysis.I will also import my data and get it ready for scruitiny.

## Importing Packages


```python
import pandas as pd
import numpy as np
import seaborn as sns
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
```1§§§§§§§§§§§§§§§§§§1
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
This section involves analysis of every variable individually.

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






    <matplotlib.axes._subplots.AxesSubplot at 0x1a161a0f90>




![png](output_20_2.png)


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






    <matplotlib.axes._subplots.AxesSubplot at 0x1a21886890>






    <matplotlib.axes._subplots.AxesSubplot at 0x1a21886890>






    <matplotlib.axes._subplots.AxesSubplot at 0x1a162f35d0>






    <matplotlib.axes._subplots.AxesSubplot at 0x1a162f35d0>






    <matplotlib.axes._subplots.AxesSubplot at 0x1a2198ead0>






    <matplotlib.axes._subplots.AxesSubplot at 0x1a2198ead0>






    <matplotlib.axes._subplots.AxesSubplot at 0x1a219f4fd0>






    <matplotlib.axes._subplots.AxesSubplot at 0x1a219f4fd0>




![png](output_22_9.png)


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






    <matplotlib.axes._subplots.AxesSubplot at 0x1a21a58b90>






    <matplotlib.axes._subplots.AxesSubplot at 0x1a21a58b90>






    <matplotlib.axes._subplots.AxesSubplot at 0x1a21eea850>






    <matplotlib.axes._subplots.AxesSubplot at 0x1a21eea850>






    <matplotlib.axes._subplots.AxesSubplot at 0x1a221bdf90>






    <matplotlib.axes._subplots.AxesSubplot at 0x1a221bdf90>




![png](output_25_7.png)


## Independent Variable(Numerical)


```python
plt.figure(1)
plt.subplot(1,2,1)
sns.distplot(train['ApplicantIncome'])

plt.subplot(1,2,2)
train['ApplicantIncome'].plot.box()
```




    <Figure size 432x288 with 0 Axes>






    <matplotlib.axes._subplots.AxesSubplot at 0x1a221b1590>






    <matplotlib.axes._subplots.AxesSubplot at 0x1a221b1590>






    <matplotlib.axes._subplots.AxesSubplot at 0x1a222c1bd0>






    <matplotlib.axes._subplots.AxesSubplot at 0x1a222c1bd0>




![png](output_27_5.png)


The box plot suggests prescence of outliers in our data and the density plot suggest a right tailed screnario.W can futher unfer that this data is not normaly distributed.


```python
#Segregate type income eaners by education.
train.boxplot(column = 'ApplicantIncome',by ='Education')
plt.suptitle('')
#Text(.5,.98,'')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a223868d0>






    Text(0.5,0.98,'')




![png](output_29_2.png)


More graduates gave a higher inceoe as compared to nin graduates.


```python
plt.figure(1)
plt.subplot(121)
sns.distplot(train['CoapplicantIncome'])

plt.subplot(122)
train['CoapplicantIncome'].plot.box(figsize = (16,5))

plt.show()
```




    <Figure size 432x288 with 0 Axes>






    <matplotlib.axes._subplots.AxesSubplot at 0x1a2257a990>






    <matplotlib.axes._subplots.AxesSubplot at 0x1a2257a990>






    <matplotlib.axes._subplots.AxesSubplot at 0x1a2261ff10>






    <matplotlib.axes._subplots.AxesSubplot at 0x1a2261ff10>




![png](output_31_5.png)


Majority of the coapplicants income ranges from 0 to 5000.
There is a prescence of outliers in the applicant income .

A look at the LoanAmount variable gives:


```python
plt.figure(1)
plt.subplot(121)
df = train.dropna()
sns.distplot(df['LoanAmount']);

plt.subplot(122)
train['LoanAmount'].plot.box(figsize = (16,5))

plt.show()
```




    <Figure size 432x288 with 0 Axes>






    <matplotlib.axes._subplots.AxesSubplot at 0x1a226455d0>






    <matplotlib.axes._subplots.AxesSubplot at 0x1a226455d0>






    <matplotlib.axes._subplots.AxesSubplot at 0x1a22691950>






    <matplotlib.axes._subplots.AxesSubplot at 0x1a22691950>




![png](output_33_5.png)



```python
train['LoanAmount'].plot.box()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a22b419d0>




![png](output_34_1.png)


# Bivariate Analysis
In this type of analysis we will look into the relationshio between dependent variable and independent variables one by one.

A look int the previous hypotheses that we defined ealier.
* Applicants with high income should have more loan chances of approval.
* Applicants who have paid their previous debts should have higher chances of approval.
* Loan approval should depend on the loan amount .
* Lesser the amount to be paid monthly to repay the loan ,higher the chances of loa approval.

## Categorical Independent Variable vs Target Variable
I will first look at the relationship betwen the categorical independent variables.Let us look at the stacked bar plot now which give us the propotion of approved and unaproved loans.


```python
Gender = pd.crosstab(train['Gender'],train['Loan_Status'])
Gender.div(Gender.sum(1).astype(float),axis = 0).plot(kind = "bar",stacked = True,figsize = (4,4))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a22cf1d10>




![png](output_37_1.png)


The propotion of males and females approved and not approved seems to be equal.

* Propotion for married is higher for the the approved loans.
* Distribution of applicants with 1 or 3+ dependents is simillar acress both the categories of Loan Status


```python
Married = pd.crosstab(train['Married'],train['Loan_Status'])
#Married
Dependents = pd.crosstab(train['Dependents'],train['Loan_Status'])
#Dependents
Education = pd.crosstab(train['Education'],train['Loan_Status'])
Education
Self_Employed = pd.crosstab(train['Self_Employed'],train['Loan_Status'])
#Dependents
Married.div(Married.sum(1).astype(float),axis = 0).plot(kind = "bar",stacked = False,figsize = (4,4))
plt.show()
Dependents.div(Dependents.sum(1).astype(float),axis = 0).plot(kind = "bar",stacked = False,figsize = (4,4))
plt.show()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Loan_Status</th>
      <th>N</th>
      <th>Y</th>
    </tr>
    <tr>
      <th>Education</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Graduate</th>
      <td>140</td>
      <td>340</td>
    </tr>
    <tr>
      <th>Not Graduate</th>
      <td>52</td>
      <td>82</td>
    </tr>
  </tbody>
</table>
</div>






    <matplotlib.axes._subplots.AxesSubplot at 0x1a22e4de90>




![png](output_40_2.png)





    <matplotlib.axes._subplots.AxesSubplot at 0x1a22ed7cd0>




![png](output_40_4.png)



```python
Credit_History = pd.crosstab(train['Credit_History'],train['Loan_Status'])
Property_Area = pd.crosstab(train['Property_Area'],train['Loan_Status'])
Credit_History.div(Credit_History.sum(1).astype(float),axis = 0).plot(kind = 'bar',stacked = True,figsize = (4,4))
plt.show()
Property_Area.div(Property_Area.sum(1).astype(float),axis = 0).plot(kind = 'bar',stacked = True,figsize = (4,4))
plt.show()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a22e42b50>




![png](output_41_1.png)





    <matplotlib.axes._subplots.AxesSubplot at 0x1a230a80d0>




![png](output_41_3.png)


* Number of people gting their loans approved is higher in semi urban areas as compared to other places
* People with credit history = 1 are more likely to be approved for loans.

## Numerical Independent Variable Vs Target Variable.
I will find the mean income for people whose loans have been approved against those whose loanse have not been approved .



```python
train.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a22dbb290>




![png](output_44_1.png)


There is no big inference we can get from here,so i will group the incomes into diferent categories or bins.


```python
bins = [0,2500,4000,6000,81000]
group = ['Low','Average','High','Very High']
#bins2 = list(np.arange(0,max(train["ApplicantIncome"]),2000))
train['Income_bin'] = pd.cut(df['ApplicantIncome'],bins,labels=group)

Income_bin = pd.crosstab(train['Income_bin'],train['Loan_Status'])
Income_bin.div(Income_bin.sum(1).astype(float),axis = 0).plot(kind = "bar",stacked = True)

plt.xlabel('Applicants Income')
P = plt.ylabel('Percentage')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a232516d0>






    Text(0.5,0,'Applicants Income')




![png](output_46_2.png)


From the output above ,it seems income doest influence a loans approval or not.This is contradicting our initial hypothesis.

Analysis of coapllicants income will also be done in the same way.


```python
bins=[0,1000,3000,42000]
group=['Low','Average','High']
train['Coapplicant_Income_bin']=pd.cut(df['CoapplicantIncome'],bins,labels=group)
Coapplicant_Income_bin=pd.crosstab(train['Coapplicant_Income_bin'],train['Loan_Status'])
Coapplicant_Income_bin.div(Coapplicant_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('Coapplicant\'s Income')
P = plt.ylabel('Percentage')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a22f4f1d0>






    Text(0.5,0,"Coapplicant's Income")




![png](output_48_2.png)


From the output,we can easily infer that the lower the income of the coaplicant,the hire the chance of a loan being approved.Mmmh!This is unrealistic!This can be a caused by the fact that most applicants do not have coapplicants so this value remains 0.This makes the loan approval not dependent on coaplicants income.

To tackle this problem i will create a *new variable* by combiningthe two variables and visualize the total effect in Loan status.


```python
train['Total_Income'] = train['ApplicantIncome']+train['CoapplicantIncome']


bins=[0,2500,4000,6000,81000]
group=['Low','Average','High', 'Very high']
train['Total_Income_bin']=pd.cut(train['Total_Income'],bins,labels=group)
Total_Income_bin=pd.crosstab(train['Total_Income_bin'],train['Loan_Status'])
Total_Income_bin.div(Total_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('Total_Income')
P = plt.ylabel('Percentage')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a23378110>






    Text(0.5,0,'Total_Income')




![png](output_50_2.png)


Now there is some sanity!

I will now visualize the loan amount variable.


```python

bins=[0,100,200,700]
group=['Low','Average','High']
train['LoanAmount_bin']=pd.cut(df['LoanAmount'],bins,labels=group)
LoanAmount_bin=pd.crosstab(train['LoanAmount_bin'],train['Loan_Status'])
LoanAmount_bin.div(LoanAmount_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('LoanAmount')
P = plt.ylabel('Percentage')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a234719d0>






    Text(0.5,0,'LoanAmount')




![png](output_52_2.png)


The propotion for approved loans is higher for low low and average and lower for high loan applications.

I will now drop the bins i created and change the 3+ to 3 to make it a numerical variable.I will also convert the target variable(LoanStatus) into 0 and 1


```python
train=train.drop(['Income_bin', 'Coapplicant_Income_bin', 'LoanAmount_bin', 'Total_Income_bin', 'Total_Income'], axis=1)
```


```python
train['Dependents'].replace('3+',3,inplace = True)
test['Dependents'].replace('3+',3,inplace = True)
train['Loan_Status'].replace('N',0,inplace = True)
train['Loan_Status'].replace('Y',1,inplace = True)

```

From here i will investigate the correlation of all numerical variables.


```python
matrix = train.corr()
f,ax = plt.subplots(figsize = (9,6))
sns.heatmap(matrix,vmax = 0.8,square = True,cmap = "BuPu")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a2352ebd0>




![png](output_57_1.png)


Most correlated variables are.
* ApplicantIncome and LoanAmount
* Credit_History and Loan_Status

# Missing Value and Outlier Treatment.
In this section I am going to explore the missing values in my dataset.

## Missing Value Imputation
I will start by counting the missing values from each variable.



```python
train.isnull().sum()
```




    Loan_ID               0
    Gender               13
    Married               3
    Dependents           15
    Education             0
    Self_Employed        32
    ApplicantIncome       0
    CoapplicantIncome     0
    LoanAmount           22
    Loan_Amount_Term     14
    Credit_History       50
    Property_Area         0
    Loan_Status           0
    dtype: int64



Mmmh looks like  I have some missing values in Gender,Married,Dependents,Self_Employed,Loan Amount,Loan amount term and credit history features.

I will impute all nnumerical values by mean or median and for categorical variablies i wil impute the missing values by mode.


```python
train['Gender'].fillna(train["Gender"].mode()[0],inplace = True)
train['Married'].fillna(train["Married"].mode()[0],inplace = True)
train['Dependents'].fillna(train["Dependents"].mode()[0],inplace = True)
train['Self_Employed'].fillna(train["Self_Employed"].mode()[0],inplace = True)
train['Credit_History'].fillna(train["Credit_History"].mode()[0],inplace = True)

```

For Loan_Amount:
First I will check the value counts for the loan amount variable.


```python
train['Loan_Amount_Term'].value_counts()
#train['Loan_Amount_Term'].mean()
#train['Loan_Amount_Term'].mode()
```




    360.0    512
    180.0     44
    480.0     15
    300.0     13
    84.0       4
    240.0      4
    120.0      3
    36.0       2
    60.0       2
    12.0       1
    Name: Loan_Amount_Term, dtype: int64



The loan amount value 360 is the highest occuring so i am going to replace the missing values by 360



```python
train['Loan_Amount_Term'].fillna(train["Loan_Amount_Term"].mode()[0],inplace = True)

```

Loan Amount


```python
train['LoanAmount'].fillna(train["LoanAmount"].median(),inplace = True)
```

Just to check if i have replaced all the missing values in my data


```python
train.isnull().sum()
```




    Loan_ID              0
    Gender               0
    Married              0
    Dependents           0
    Education            0
    Self_Employed        0
    ApplicantIncome      0
    CoapplicantIncome    0
    LoanAmount           0
    Loan_Amount_Term     0
    Credit_History       0
    Property_Area        0
    Loan_Status          0
    dtype: int64



I will use the same technique to fill the missng data in the test data frame as well.
test


```python
test['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
test['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
test['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
test['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
test['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)
```

## Outlier Treatment
Prevoiusly we saw that loan amount had outliers.
For a remider i will plot a histogram of the loan amounts.


```python
train['LoanAmount'].hist(bins = 20)
plt.title("Current LoanAmount")
plt.xlabel("LoanAmount")
plt.ylabel("Frequency")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a23698cd0>






    Text(0.5,1,'Current LoanAmount')






    Text(0.5,0,'LoanAmount')






    Text(0,0.5,'Frequency')




![png](output_74_4.png)



```python
train['LoanAmount_log'] = np.log(train['LoanAmount'])
train['LoanAmount_log'].hist(bins = 30)
plt.xlabel("LoanAmount")
plt.ylabel("Frequency")
plt.title("LoanAmount after log transformations")
#Doing the same to test data
test['LoanAmount_log'] = np.log(train['LoanAmount'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a23698b50>






    Text(0.5,0,'LoanAmount')






    Text(0,0.5,'Frequency')






    Text(0.5,1,'LoanAmount after log transformations')




![png](output_75_4.png)


After transforming the LoanaAmount values,I now have an almost normaly distributed data.

# Evaluation Metrics for Classification Variables.

Inorder for a models values to be trustworthy,its important to evalueate them.
I will wcaluate my model using any of the following:
1. **CONFUSION MATRIX**

**Accuracy**:This can be deine by the use of a confusion matrix.
    A confusion matrix has the following main parts:
* True Negative - Targets which are actually true$(Y)$ and we have predicted them true$(Y)$.
* True Negative - Targets which are actually false $(N)$ and we have predicted them false $(N)$
* False Positive - Targets which are actually false $(N)$ but we have predicted them true$(Y)$
* False Negative - Targets which are actually true $(Y)$ but we have predicted them false $(N)$

Accuracy is therefore given by:
$$accuracy = \frac{True Positive+True Nehative}{True Positive+True Nehative+False Positive+False Negatives}$$

**Precision**It is ameasure of correctness achieved in true prediction that is observations marked as true.
$$Precision = \frac{True Positive}{True Positive+True Negative}$$

**Sensitivity/Recal** - Measure of actual predictions which are predicted correctly 
$$Sensitivity = \frac{True Positive}{True Negative+False Positive}$$

**Specificity** - How many observations of false class are labbeled correctly.
$$Specificity = \frac{True Negative}{True Negative + False positive}$$

2 **ROC Curve** - *Receiver Operating Characteristic*

Summarizes the model performance by evaluating the trade offs between true positive rate (sensitivity) and false positive rate (Specificity.)

The area under the curve ,refered as index of accuracy or accordance index ,is perfect perfremance metric for ROC curve.A higher area indicates a better prediction power of the model.


# MODEL BUILDING (1)
This first model to predict loan status will be logistic regression.

I will first drop the LOAN_ID variable since it lacks any effect in the target variable.


```python
train = train.drop('Loan_ID',axis = 1)
test = test.drop('Loan_ID',axis = 1)

```


```python
x = train.drop('Loan_Status',1)
y = train.Loan_Status
```


```python
x = pd.get_dummies(x)
train = pd.get_dummies(train)
test = pd.get_dummies(test)
```


```python
from sklearn.model_selection import train_test_split
```


```python
x_train,x_cv,y_train,y_cv = train_test_split(x,y,test_size = .3)
```


```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model = LogisticRegression()
model.fit(x_train,y_train)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)




```python
#Predicting the loan status for validation set 
pred_cv = model.predict(x_cv)
#Accuracy of our predictionns by calculating the accuracy
accuracy_score(y_cv,pred_cv)
```




    0.827027027027027




```python
pred_test = model.predict(test)
submission = pd.read_csv("/Users/admin/Downloads/submissions.csv")
```


```python
submission['Loan_Status']=pred_test
submission['Loan_ID'] = test_original['Loan_ID']
```

Since I need to report my predictions in Y and N,i will convert 1's and 0's to Y and N.


```python
submission['Loan_Status'].replace(0,'N',inplace = True)
submission['Loan_Status'].replace(1,'Y',inplace = True)

```


```python
pd.DataFrame(submission,columns = ['Loan_ID','Loan_Status']).to_csv('logistic.csv')
submission
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Loan_ID</th>
      <th>Loan_ID.1</th>
      <th>Loan_Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LP001015</td>
      <td>NaN</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LP001022</td>
      <td>NaN</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LP001031</td>
      <td>NaN</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LP001035</td>
      <td>NaN</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LP001051</td>
      <td>NaN</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>5</th>
      <td>LP001054</td>
      <td>NaN</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>6</th>
      <td>LP001055</td>
      <td>NaN</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>7</th>
      <td>LP001056</td>
      <td>NaN</td>
      <td>N</td>
    </tr>
    <tr>
      <th>8</th>
      <td>LP001059</td>
      <td>NaN</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>9</th>
      <td>LP001067</td>
      <td>NaN</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>10</th>
      <td>LP001078</td>
      <td>NaN</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>11</th>
      <td>LP001082</td>
      <td>NaN</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>12</th>
      <td>LP001083</td>
      <td>NaN</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>13</th>
      <td>LP001094</td>
      <td>NaN</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>14</th>
      <td>LP001096</td>
      <td>NaN</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>15</th>
      <td>LP001099</td>
      <td>NaN</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>16</th>
      <td>LP001105</td>
      <td>NaN</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>17</th>
      <td>LP001107</td>
      <td>NaN</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>18</th>
      <td>LP001108</td>
      <td>NaN</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>19</th>
      <td>LP001115</td>
      <td>NaN</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>20</th>
      <td>LP001121</td>
      <td>NaN</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>21</th>
      <td>LP001124</td>
      <td>NaN</td>
      <td>N</td>
    </tr>
    <tr>
      <th>22</th>
      <td>LP001128</td>
      <td>NaN</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>23</th>
      <td>LP001135</td>
      <td>NaN</td>
      <td>N</td>
    </tr>
    <tr>
      <th>24</th>
      <td>LP001149</td>
      <td>NaN</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>25</th>
      <td>LP001153</td>
      <td>NaN</td>
      <td>N</td>
    </tr>
    <tr>
      <th>26</th>
      <td>LP001163</td>
      <td>NaN</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>27</th>
      <td>LP001169</td>
      <td>NaN</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>28</th>
      <td>LP001174</td>
      <td>NaN</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>29</th>
      <td>LP001176</td>
      <td>NaN</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>337</th>
      <td>LP002856</td>
      <td>NaN</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>338</th>
      <td>LP002857</td>
      <td>NaN</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>339</th>
      <td>LP002858</td>
      <td>NaN</td>
      <td>N</td>
    </tr>
    <tr>
      <th>340</th>
      <td>LP002860</td>
      <td>NaN</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>341</th>
      <td>LP002867</td>
      <td>NaN</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>342</th>
      <td>LP002869</td>
      <td>NaN</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>343</th>
      <td>LP002870</td>
      <td>NaN</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>344</th>
      <td>LP002876</td>
      <td>NaN</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>345</th>
      <td>LP002878</td>
      <td>NaN</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>346</th>
      <td>LP002879</td>
      <td>NaN</td>
      <td>N</td>
    </tr>
    <tr>
      <th>347</th>
      <td>LP002885</td>
      <td>NaN</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>348</th>
      <td>LP002890</td>
      <td>NaN</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>349</th>
      <td>LP002891</td>
      <td>NaN</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>350</th>
      <td>LP002899</td>
      <td>NaN</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>351</th>
      <td>LP002901</td>
      <td>NaN</td>
      <td>N</td>
    </tr>
    <tr>
      <th>352</th>
      <td>LP002907</td>
      <td>NaN</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>353</th>
      <td>LP002920</td>
      <td>NaN</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>354</th>
      <td>LP002921</td>
      <td>NaN</td>
      <td>N</td>
    </tr>
    <tr>
      <th>355</th>
      <td>LP002932</td>
      <td>NaN</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>356</th>
      <td>LP002935</td>
      <td>NaN</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>357</th>
      <td>LP002952</td>
      <td>NaN</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>358</th>
      <td>LP002954</td>
      <td>NaN</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>359</th>
      <td>LP002962</td>
      <td>NaN</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>360</th>
      <td>LP002965</td>
      <td>NaN</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>361</th>
      <td>LP002969</td>
      <td>NaN</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>362</th>
      <td>LP002971</td>
      <td>NaN</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>363</th>
      <td>LP002975</td>
      <td>NaN</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>364</th>
      <td>LP002980</td>
      <td>NaN</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>365</th>
      <td>LP002986</td>
      <td>NaN</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>366</th>
      <td>LP002989</td>
      <td>NaN</td>
      <td>Y</td>
    </tr>
  </tbody>
</table>
<p>367 rows × 3 columns</p>
</div>



# Logistic Regression using stratified k-folds cross validation.
I will now check how robust my model is to unseen data and for this i will use what is known as k fold cross validation.

## Stratified K fold validation.

* Stratification is the process of rearranging the data so as to ensure that each fold is aa good representative of the whole.

 


```python
from sklearn.model_selection import StratifiedKFold
```


```python
i=1
kf = StratifiedKFold(n_splits=50,random_state=1,shuffle=True)
for train_index,test_index in kf.split(x,y):
     print('\n{} of kfold {}'.format(i,kf.n_splits))
     xtr,xvl = x.loc[train_index],x.loc[test_index]
     ytr,yvl = y[train_index],y[test_index]
    
     model = LogisticRegression(random_state=1)
     model.fit(xtr, ytr)
     pred_test = model.predict(xvl)
     score = accuracy_score(yvl,pred_test)
     print('accuracy_score',score)
     i+=1
pred_test = model.predict(test)
pred=model.predict_proba(xvl)[:,1]
```

    
    1 of kfold 50





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.7692307692307693)
    
    2 of kfold 50





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.8461538461538461)
    
    3 of kfold 50





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.8461538461538461)
    
    4 of kfold 50





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.6923076923076923)
    
    5 of kfold 50





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.7692307692307693)
    
    6 of kfold 50





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.8461538461538461)
    
    7 of kfold 50





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.8461538461538461)
    
    8 of kfold 50





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.7692307692307693)
    
    9 of kfold 50





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.7692307692307693)
    
    10 of kfold 50





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.8461538461538461)
    
    11 of kfold 50





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.9230769230769231)
    
    12 of kfold 50





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.9230769230769231)
    
    13 of kfold 50





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.7692307692307693)
    
    14 of kfold 50





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.8461538461538461)
    
    15 of kfold 50





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.7692307692307693)
    
    16 of kfold 50





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.9230769230769231)
    
    17 of kfold 50





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.8461538461538461)
    
    18 of kfold 50





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.7692307692307693)
    
    19 of kfold 50





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.7692307692307693)
    
    20 of kfold 50





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.7692307692307693)
    
    21 of kfold 50





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.7692307692307693)
    
    22 of kfold 50





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.8461538461538461)
    
    23 of kfold 50





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.6666666666666666)
    
    24 of kfold 50





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.9166666666666666)
    
    25 of kfold 50





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.75)
    
    26 of kfold 50





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.75)
    
    27 of kfold 50





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.8333333333333334)
    
    28 of kfold 50





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.8333333333333334)
    
    29 of kfold 50





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.9166666666666666)
    
    30 of kfold 50





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.75)
    
    31 of kfold 50





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.75)
    
    32 of kfold 50





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.8333333333333334)
    
    33 of kfold 50





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.9166666666666666)
    
    34 of kfold 50





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.6666666666666666)
    
    35 of kfold 50





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.8333333333333334)
    
    36 of kfold 50





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.75)
    
    37 of kfold 50





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.8333333333333334)
    
    38 of kfold 50





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.6666666666666666)
    
    39 of kfold 50





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.8333333333333334)
    
    40 of kfold 50





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.75)
    
    41 of kfold 50





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 1.0)
    
    42 of kfold 50





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.75)
    
    43 of kfold 50





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.8181818181818182)
    
    44 of kfold 50





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.7272727272727273)
    
    45 of kfold 50





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.7272727272727273)
    
    46 of kfold 50





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.9090909090909091)
    
    47 of kfold 50





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.7272727272727273)
    
    48 of kfold 50





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.9090909090909091)
    
    49 of kfold 50





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.9090909090909091)
    
    50 of kfold 50





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 1.0)


The mean of this cross validation is 0.81.When visualized in a ROC Curve I have something like this.



```python
from sklearn import metrics
fpr, tpr, _ = metrics.roc_curve(yvl,  pred)
auc = metrics.roc_auc_score(yvl, pred)
plt.figure(figsize=(12,8))
plt.plot(fpr,tpr,label="validation, auc="+str(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)
plt.show()
```




    <Figure size 864x576 with 0 Axes>






    [<matplotlib.lines.Line2D at 0x1a23f91390>]






    Text(0.5,0,'False Positive Rate')






    Text(0,0.5,'True Positive Rate')






    <matplotlib.legend.Legend at 0x1a23f3d690>




![png](output_95_5.png)


Here the AUC is 1.0.I will make a predicted submissions in my sub file.


```python
submission["Loan_Status"] = pred_test
submission['Loan_ID']=test_original['Loan_ID']
```


```python
submission['Loan_Status'].replace(0,'N',inplace=True)
submission['Loan_Status'].replace(1,'Y',inplace=True)
pd.DataFrame(submission,columns=['Loan_ID','Loan_Status']).to_csv('submision3.csv')

```

# Feature Engeneering
I will create new features based on the ones that  I am having currently.

* **Total Income** this is as seen earlier during bivariate anlaysis
* **EMI** - monthly amount to be payed by the individual to repay the loan.To get this i will devide the loan amount by the loarn term.
* **Balance income** - Income left after EMI has been paid.


```python
train['Total_Income']=train['ApplicantIncome']+train['CoapplicantIncome']
test['Total_Income']=test['ApplicantIncome']+test['CoapplicantIncome']
```


```python
sns.distplot(train['Total_Income']);
```


![png](output_101_0.png)


Here there is a shift to the left.I will correct this by introducing logarithms.


```python
train['Total_Income_log'] = np.log(train['Total_Income'])
sns.distplot(train['Total_Income_log']);
test['Total_Income_log'] = np.log(test['Total_Income'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a24250fd0>




![png](output_103_1.png)



```python
train['EMI']=train['LoanAmount']/train['Loan_Amount_Term']
test['EMI']=test['LoanAmount']/test['Loan_Amount_Term']
```


```python
sns.distplot(train['EMI'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a2428e090>




![png](output_105_1.png)



```python
train['Balance Income'] = train['Total_Income']-(train['EMI']*1000)
test['Balance Income'] = test['Total_Income']-(train['EMI']*1000)
```


```python
sns.distplot(train['Balance Income'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a22ddc250>




![png](output_107_1.png)



```python
train = train.drop(['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term'],axis = 1)
test = test.drop(['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term'],axis = 1)

```

# Model Building 2
I am going to build the following models in this section.
* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost


```python
x = train.drop('Loan_Status',1)
y = train.Loan_Status
```

## Logistic Regression


```python
i=1
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for train_index,test_index in kf.split(x,y):
     print('\n{} of kfold {}'.format(i,kf.n_splits))
     xtr,xvl = x.loc[train_index],x.loc[test_index]
     ytr,yvl = y[train_index],y[test_index]
    
     model = LogisticRegression(random_state=1)
     model.fit(xtr, ytr)
     pred_test = model.predict(xvl)
     score = accuracy_score(yvl,pred_test)
     print('accuracy_score',score)
     i+=1
pred_test = model.predict(test)
pred=model.predict_proba(xvl)[:,1]
```

    
    1 of kfold 5





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.8064516129032258)
    
    2 of kfold 5





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.8225806451612904)
    
    3 of kfold 5





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.7786885245901639)
    
    4 of kfold 5





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.7868852459016393)
    
    5 of kfold 5





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



    ('accuracy_score', 0.819672131147541)



```python
submission['Loan_Status']=pred_test
submission['Loan_ID']=test_original['Loan_ID']
```


```python
submission['Loan_Status'].replace(0, 'N',inplace=True)
submission['Loan_Status'].replace(1, 'Y',inplace=True)
# Converting submission file to .csv format
pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('Log2.csv')
```

## Decision Tree


```python
from sklearn import tree
i=1
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for train_index,test_index in kf.split(x,y):
     print('\n{} of kfold {}'.format(i,kf.n_splits))
     xtr,xvl = x.loc[train_index],x.loc[test_index]
     ytr,yvl = y[train_index],y[test_index]
    
     model = tree.DecisionTreeClassifier(random_state=1)
     model.fit(xtr, ytr)
     pred_test = model.predict(xvl)
     score = accuracy_score(yvl,pred_test)
     print('accuracy_score',score)
     i+=1
pred_test = model.predict(test)
```

    
    1 of kfold 5





    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=1,
                splitter='best')



    ('accuracy_score', 0.7258064516129032)
    
    2 of kfold 5





    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=1,
                splitter='best')



    ('accuracy_score', 0.7419354838709677)
    
    3 of kfold 5





    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=1,
                splitter='best')



    ('accuracy_score', 0.7049180327868853)
    
    4 of kfold 5





    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=1,
                splitter='best')



    ('accuracy_score', 0.680327868852459)
    
    5 of kfold 5





    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=1,
                splitter='best')



    ('accuracy_score', 0.7049180327868853)


The mean validation accuracy for this model is 0.69


```python
submission['Loan_Status']=pred_test
submission['Loan_ID']=test_original['Loan_ID']
```


```python
submission['Loan_Status'].replace(0, 'N',inplace=True)
submission['Loan_Status'].replace(1, 'Y',inplace=True)
pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('Decision Tree.csv')
```

## Random Forest


```python
from sklearn.ensemble import RandomForestClassifier
```


```python
i=1
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for train_index,test_index in kf.split(x,y):
     print('\n{} of kfold {}'.format(i,kf.n_splits))
     xtr,xvl = x.loc[train_index],x.loc[test_index]
     ytr,yvl = y[train_index],y[test_index]
    
     model = RandomForestClassifier(random_state=1, max_depth=10)
     model.fit(xtr, ytr)
     pred_test = model.predict(xvl)
     score = accuracy_score(yvl,pred_test)
     print('accuracy_score',score)
     i+=1
pred_test = model.predict(test)
```

    
    1 of kfold 5





    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=10, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                oob_score=False, random_state=1, verbose=0, warm_start=False)



    ('accuracy_score', 0.8225806451612904)
    
    2 of kfold 5





    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=10, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                oob_score=False, random_state=1, verbose=0, warm_start=False)



    ('accuracy_score', 0.8145161290322581)
    
    3 of kfold 5





    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=10, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                oob_score=False, random_state=1, verbose=0, warm_start=False)



    ('accuracy_score', 0.7377049180327869)
    
    4 of kfold 5





    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=10, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                oob_score=False, random_state=1, verbose=0, warm_start=False)



    ('accuracy_score', 0.7295081967213115)
    
    5 of kfold 5





    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=10, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                oob_score=False, random_state=1, verbose=0, warm_start=False)



    ('accuracy_score', 0.8114754098360656)



```python

```
