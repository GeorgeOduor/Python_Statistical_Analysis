
# _T-tests_ in Python
In this kernel i am  discuss how to perform _T-tests_ in python.T-test is a parametric statistical test that is used to test if there exists a statisticaly significant difference between groups or a group and a hypothesised mean.For these tests to be implemented the data must meet the parametric assumptions which i am not going to go through in this short kernel.

There are three types of _T-tests_:

* One sample _T-test_.
* Two sample _T-test_.
* Paired sample _T-test_.

## Part one: One sample _T-test_.
### What it is.
One sample _T-test_ is a parametric statistical technique that is used to test if there exists a statisticaly significant difference between a groups mean and a hypothesised mean.

**Example**

The average income of people working in a town can be assumed /hypothesised to be Ksh 25,500.A reseacher who feels that this mean is not true samples 50 people from this town ,calculates the mean and tests that mean against the hypothesised mean stated earlier.
### Sample problem.

#### Dataset

In this example i have the iris dataset in seaborn library in python.
#### Question

It has been hypothesised that the mean sepal length of floweres in the iris dataset is 4cm.We are going to test if this statement is plausible.

**Step 1:State the hypothesis.**

$H_0:\mu_0=\bar x$ Hypothesized mean $\mu_0$ is equal to the sample mean $\bar x$

$H_a:\mu_0 \neq \bar x$ Hypothesizid mean $\mu_0$ is not equal to the sample mean $\bar x$

In this case $\mu_0$=4



To handle this data should first import the dataset into my notebook.


```python
import seaborn as sns # loading seaborn library
import pandas as pd
iris = sns.load_dataset("iris") # loading the dataset
xbar = round(iris['sepal_length'].mean(),1) # calculating sample mean of sepal_length and rounding to the nearest one dp
xbar

```




    5.8




```python
mu = float(input("What is your hypothesized mean? "))
def one_sanple(dataset,variable,mu):
    mu = float(input("What is your hypothesized mean? "))
    from scipy import stats # importing SciPy library
    testresults = stats.ttest_1samp(dataset[variable],mu)
    xbar = round(iris['sepal_length'].mean(),1) # calculating sample mean of sepal_length and rounding to the nearest one dp
    results=pd.DataFrame({"":["Sample Data"],"Test Statistic":testresults[0],"p-value":testresults[1],"true mean":xbar,"hypothesized mean":mu})
    return(results)

one_sanple(iris,'sepal_length',mu)

```

    What is your hypothesized mean? 4
    What is your hypothesized mean? 4





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
      <th></th>
      <th>Test Statistic</th>
      <th>p-value</th>
      <th>true mean</th>
      <th>hypothesized mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Sample Data</td>
      <td>27.263681</td>
      <td>8.764592e-60</td>
      <td>5.8</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>



###  Observation and inference.

* T score:

T score tells us how far or hypothesised  mean is from the true mean.A bigger t score tels us the hypothesized mean is far away from the true mean and vice versa.

**But how big is this 'big'???**

In order to answer this question we can use our the tscore values from the t test table.We can then eject the null hypothesis of equality of mean if the value we calculated greater than the t value.

* P- Value:

In most research scenarios,the t tables are not available so *p values* are most frequently used.P-value is the probability of getting the observed results or even more extreme results if the null hypothesis was true.We can safely reject the null hypothesis in favour of the alternative if the p value is less than an  acceptable threshold of 0.05.

Looking at the output above,the t value is very large which is supported with a very small p value.This evidently support our alternative hypothesis that the true mean is not equal to 4.

## Part two:Independent Sample _T test._

### What it is.

Independent sample Ttest is used to test if two indpendent groups of data come from the same population,i.e if they have the same mean.

### Example

From the salaries example above,lets say we want to compare the salaries of two different groups of people,say from town A and town B.Data on these peoples salary will be sampled randomly from these two areas and compared.

### Hypothesis

$H_0:\mu_1=\mu_2$

$H_a:\mu_1 \neq \mu_2$

### Dataset

In an attempt to show this i will still use the iris dataset.I am going to test if the petall length average for the setosa and 



```python
sum =iris['petal_length'].groupby(iris['species']).std(ddof=1)
setosa_virginica = iris[(iris.species == "setosa") | (iris.species == "virginica")]
setosa = iris[(iris.species=='setosa')]['sepal_width']
virginica = iris[(iris.species=='virginica')]['sepal_width']
```

The statistic
$$t=\frac{M_x-M_y}{\sqrt {[\frac{(\sum X^2- \frac{(\sum X)^2}{N_x})+\sum Y^2- \frac{(\sum Y)^2}{N_y}
}{N_x+N_y-2}]}.[\frac{1}{N_x}+\frac{1}{N_y}]}$$
where:

$\sum$= sum the following scores

$M_x$= mean for Group A,
$M_y$= mean for Group B,
X = score in Group 1,
Y = score in Group 2,
$N_x$= number of scores in Group 1,
$N_y$= number of scores in Group 2 

The tric is to first calculate the t statistic.


```python
from scipy import stats
twosample = stats.ttest_ind(setosa,virginica)# for calcualating the t statistic and pvalue
#descriptive statistics
setosa_mean = setosa.mean()
pd.DataFrame({"":['Setosa-Virginica'],
             "Test statistic":twosample[0],
             "p-value":twosample[1],
             "Setosa Mean":setosa.mean(),
             "Virginica Mean":virginica.mean(),
             print('Setosa Mean'):setosa.mean()-virginica.mean()})#presenting results


```

    Setosa Mean





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
      <th></th>
      <th>Test statistic</th>
      <th>p-value</th>
      <th>Setosa Mean</th>
      <th>Virginica Mean</th>
      <th>None</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Setosa-Virginica</td>
      <td>6.450349</td>
      <td>4.246355e-09</td>
      <td>3.428</td>
      <td>2.974</td>
      <td>0.454</td>
    </tr>
  </tbody>
</table>
</div>



### Observation and inference

* P-value is less than 0.05.

There is no statisticaly significant proof that the means are the same so it is safe to reject the null hypothesis of equality of means in this case.

## Part three Paired Sample _T-test_.

Paired sample t test is tests for impact or effect of something.It tests if there is a difference between two paides groups,i.e before and after situations.In such a case we have one sample population but measured twice.

### Example

A reseacher may want to know if there is any effect of a particular drug on malaria patients.He then takes the number of plamodioum cells recorded on a patients blood before and after the patient is exposed to a articular drug.The mean number is thus compared.

### Hypothesis

$H_0:\mu_1=\mu_0$ Population means are equal(NO EFFECT)

$H_a:\mu_1 \neq \mu_0$ Population means are not equal.(EFFECT)

### Data Set

In order to illustrate this , we will still use the iris dataset.We are going to assume our flowers  were  treated with a particular chemical which is believed to have an effect on sepal length.In order to achieve this i am going to add random numbers to sepal length and then test if there was realy an effect.


```python
# Generating data
import numpy as np
np.random.seed(123)
randnum = stats.norm.rvs(loc=1,scale = 2,size = len(iris))
iris['sepal_after']=round(iris['sepal_length']+randnum,1)
```

### Test statistic

$$t = \frac{\bar x_{diff}-0}{s_{\bar x}}$$
Where

$\bar x_{diff}$=Sample mean of the differences

$n$=sample size

$s_{diff}$=sample standard deviation

$s_{\bar x}$=estimated standard error of the mean($\frac{s_{diff}}{\sqrt{n}}$)

The calculated t value is then compared to the critical t value with df =n-1 from the distributation table for chosel level of confidence.


```python
pairedresult = stats.ttest_rel(iris['sepal_length'],iris['sepal_after'])
pd.DataFrame({"":['Sepal Before - Sepal After'],
             "Test statistic":pairedresult[0],
             "p-value":pairedresult[1],
             "Mean after":iris['sepal_after'].mean(),
             "Mean before":iris['sepal_length'].mean(),
             'Mean Difference':iris['sepal_after'].mean()-iris['sepal_length'].mean()})#presenting results
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
      <th></th>
      <th>Test statistic</th>
      <th>p-value</th>
      <th>Mean after</th>
      <th>Mean before</th>
      <th>Mean Difference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Sepal Before - Sepal After</td>
      <td>-6.223403</td>
      <td>4.687086e-09</td>
      <td>6.956667</td>
      <td>5.843333</td>
      <td>1.113333</td>
    </tr>
  </tbody>
</table>
</div>



### Observation and Inference

* There was a significant average difference in mean length of sepal lengths after the chemical addition ($t_{150}=-6.223,p<0.05$.

* On average ,Sepal lengths before were 1.113cm lower than after.


```python

```


<style type="text/css">
table.dataframe td, table.dataframe th {
    border-style: solid;
}
# to give my tables some nice borders in jupyter notebook


