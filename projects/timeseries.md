
# Creating a time series forecast with python. 
## Problem statement
A company wants to make an investment in a new form of transportation - JetRail. JetRail uses Jet propulsion technology to run rails and move people at a high speed! The investment would only make sense, if they can get more than 1 Million monthly users with in next 18 months. In order to help this company in their decision, I need to forecast the traffic on JetRail for the next 7 months. I am provided with traffic data of JetRail since inception in the test file.

# Understanding the Data
Before taking any data analysis project it is important to understand the data thoroughly.At this stage i am going to imprt the data and necessary packages.I am going to look at possible hypothesis that can be generated for this data ie the possible factors that can affect the outcome.

##  1. Hypothesis Generation.

* There will be an increase in the number of traffic as the years pass by.This is becuse population has tendency of increasing with time.
* Traffic will be high from May to October.This is because in the city the data was collected in is a tourist center
* Traffic on week days will be more as compared to weekend and holydays because most people go to work iduring week days.
* Traffic during peak hours will be high.Here peak hours mean the time people travel to and from work/colleges.

I am going to try to validate my hypotheses based on the data.

## 2. Getting System Ready and loading the data.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime    # To access datetime
from pandas import Series
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
```

The data


```python
train = pd.read_csv("/Users/admin/Downloads/timeseriesdata/train.csv")
test = pd.read_csv("/Users/admin/Downloads/timeseriesdata/test.csv")
```

Because so many things happen in the process of analysis,i will make a copy of my original data so that i do not loose it.


```python
train_original = train.copy()
test_original = test.copy()
```

## 3.Data Stucture and content
What makes up my data?


```python
train.columns,test.columns
```




    (Index(['ID', 'Datetime', 'Count'], dtype='object'),
     Index(['ID', 'Datetime'], dtype='object'))




```python

train.head(5),test.head(5)
```




    (   ID          Datetime  Count
     0   0  25-08-2012 00:00      8
     1   1  25-08-2012 01:00      2
     2   2  25-08-2012 02:00      6
     3   3  25-08-2012 03:00      2
     4   4  25-08-2012 04:00      2,       ID          Datetime
     0  18288  26-09-2014 00:00
     1  18289  26-09-2014 01:00
     2  18290  26-09-2014 02:00
     3  18291  26-09-2014 03:00
     4  18292  26-09-2014 04:00)



The output tells me that the train data is made up of three columns ie the **ID,Datetime,Count**.The test is made up of only the **ID and the Datetime**.

* ID in this case is the unique number given to each observation point.
* Datetime is  the time of each observation .
* Count is the passenger count to each day.

Then i will look at the data types used to store my values.


```python
train.dtypes,test.dtypes
```




    (ID           int64
     Datetime    object
     Count        int64
     dtype: object, ID           int64
     Datetime    object
     dtype: object)



Then the shape


```python
train.shape,test.shape
```




    ((18288, 3), (5112, 2))



So train is made up of 18288 observations and 3 variables while the test data set is made up 5112 records and two variables.The extra variable in train is the count which i will be forecasting.

## 4. Feature Extraction.
First i will convert the datetime column into a format that python can understand and then extract the date and time from it.


```python
train['Datetime'] = pd.to_datetime(train.Datetime,format='%d-%m-%Y %H:%M')
test['Datetime'] = pd.to_datetime(train.Datetime,format='%d-%m-%Y %H:%M')
train_original['Datetime'] = pd.to_datetime(train.Datetime,format='%d-%m-%Y %H:%M')
test_original['Datetime'] = pd.to_datetime(train.Datetime,format='%d-%m-%Y %H:%M')
train.head(4)
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
      <th>ID</th>
      <th>Datetime</th>
      <th>Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2012-08-25 00:00:00</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2012-08-25 01:00:00</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2012-08-25 02:00:00</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>2012-08-25 03:00:00</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



One of my hypotheses touched the effect of the hour,day and month and year on the pass count.I will thus extract these parameters from the Datetime variable to try to validate my hypothesis.

This step will create 4 more columns in my dataframe.
* Year
* Month
* Hour
* Day


```python
for i in (train,test,train_original,test_original):
    i['year'] = i.Datetime.dt.year
    i['month'] = i.Datetime.dt.month
    i['day'] = i.Datetime.dt.day
    i['Hour'] = i.Datetime.dt.hour
```

I will also make a new variable for week and weekend to visualize the impact of trafic during these times.

* First extract the day of the week from date time and then based on the values assign whether the day is a weekend or not
* Values 5 and 6 show that the day is a weekend.


```python
train['day_of_week']=train['Datetime'].dt.dayofweek
train.head(2)
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
      <th>ID</th>
      <th>Datetime</th>
      <th>Count</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>Hour</th>
      <th>day_of_week</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2012-08-25 00:00:00</td>
      <td>8</td>
      <td>2012</td>
      <td>8</td>
      <td>25</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2012-08-25 01:00:00</td>
      <td>2</td>
      <td>2012</td>
      <td>8</td>
      <td>25</td>
      <td>1</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



So i will asign 1 if the day is weekend and 0 if the day is a week day


```python
def applyer(row):
    if row.dayofweek == 5 or row.dayofweek == 6:
        return 1
    else:
        return 0
    
temp2 = train['Datetime'].apply(applyer)
train['weekend'] = temp2
```

Looking at time series


```python
train.index = train['Datetime']
df = train.drop('ID',1)
ts = df['Count']
plt.figure(figsize = (16,8))
plt.plot(ts,label = 'Passenger Count')
plt.title('Title Series')
plt.xlabel('Time')
plt.ylabel('Paaenger Count')
plt.legend(loc = "best")
```




    <matplotlib.legend.Legend at 0x11979c860>




![png](/timeseriesFiles/output_24_1.png)


* There is an increasing trend in the series .
* At some points there is a sudden increase in the number of counts  ,could be because of some event in town.

## Exploratory Analysis
So here i will go by my hypotheses.

  1. Increase in traffic as the years go by.


```python
train.groupby('year')['Count'].mean().plot.bar()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x119760358>




![png](/timeseriesFiles/output_26_1.png)


There is an exponential growth in the traffic with respect to year.

2. Traffic will be high on may to october.
 
I will visualize this in an attempt to check the hypothesis.


```python
train.groupby('month')['Count'].mean().plot.bar()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x119551390>




![png](/timeseriesFiles/output_28_1.png)


This plot doesnt give us a satisfactory result as some years doent ave complete number of months.


```python
temp=train.groupby(['year', 'month'])['Count'].mean()
temp.plot(figsize=(15,5), title= 'Passenger Count(Monthwise)', fontsize=14)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1194d8278>




![png](/timeseriesFiles/output_30_1.png)


Above plot show an increasing trend in monthly passenger count and growth is aprox exponential.
3. Mean of passenger count daily


```python
train.groupby('day')['Count'].mean().plot.bar()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x119438d30>




![png](/timeseriesFiles/output_32_1.png)


Theis daywise plot isn giving me alot of information.I will thus check thr mean hourly passenger count.


```python
train.groupby('Hour')['Count'].mean().plot.bar()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1189c0ba8>




![png](/timeseriesFiles/output_34_1.png)


The peak traffic is at 7 pm and then we see a decreasing trend till 5 am.
The number rises steadily again up to 12noon.

*  **Traffic is more on weekdays.**

I will validate this.


```python
train.groupby('weekend')['Count'].mean().plot.bar()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1190ac0b8>




![png](/timeseriesFiles/output_36_1.png)


Okay so my hpothesis is validated.

Looking at the day wise passenger count:-


```python
train.groupby('day_of_week')['Count'].mean().plot.bar()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x119e8db38>




![png](/timeseriesFiles/output_38_1.png)


Its evident again that weekends has the least passenger count compared to other days.

Again i will drop the ID variable since it has nothing to do with the passenger count.


```python
train = train.drop('ID',1)
```


```python
train.Timestamp = pd.to_datetime(train.Datetime,format = '%d-%m-%Y %H:%M')
train.index = train.Timestamp

hourly = train.resample('H').mean()

daily = train.resample('D').mean()

weekly = train.resample('W').mean()

monthly = train.resample('M').mean()
```

Hourly ,daily and monthly counts


```python
fig,axs = plt.subplots(4,1)
hourly.Count.plot(figsize = (16,8),title = "Hourly count",ax= axs[0])
daily.Count.plot(figsize = (16,8),title = "Hourly count",ax= axs[1])
weekly.Count.plot(figsize = (16,8),title = "Hourly count",ax= axs[2])
monthly.Count.plot(figsize = (16,8),title = "Hourly count",ax= axs[3])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11a521208>




![png](/timeseriesFiles/output_43_1.png)


So here the time series is becoming stable when i agregate it dayly weekly and monthly.

For daily time series:


```python
test.Timestamp = pd.to_datetime(test.Datetime,format = '%d-%m-%Y %H:%M')
test.index = test.Timestamp

test = test.resample('D').mean()
train.Timestamp

train = train.resample('D').mean()
```

# ForeCating Using Multiple modeling techniques.
 
## 1.Splitting data into training and validation


```python
Train = train.ix['2012-08-25':'2014-06-24']
valid = train.ix['2014-06-25':'2014-09-25']
```

I have selected the last three months for validation data and the rest in the train data.

So down here i will visualize how train and validation data play out then predict pass count for the validation data.


```python
Train.Count.plot(label = 'train')
valid.Count.plot(label = 'valid')
plt.ylabel("Passenger count")
plt.legend(loc = "best")
```




    <matplotlib.legend.Legend at 0x119f5e208>




![png](/timeseriesFiles/output_49_1.png)


## 2. Modelling techniques
### i. Naive approach
In this approach,the next expected point is equal to the last observed point.


```python
p = np.asanyarray(Train.Count)
y_hat = valid.copy()
y_hat['naive'] = p[len(p)-1]
plt.figure(figsize=(12,8))
plt.plot(Train.index,Train['Count'],label = 'Train')
plt.plot(valid.index,valid['Count'],label = 'Valid')
plt.plot(y_hat.index,y_hat['naive'],label = 'Naive forecast')
plt.legend(loc = 2)
plt.title('Naive forecasting method')
plt.show()

```


![png](/timeseriesFiles/output_51_0.png)


How accurate is this model?I will answer this question by doing a calculation of the RMSE


```python
from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(valid.Count,y_hat.naive))
print(rms)
```

    111.79050467496724


This method is not suitable for data with high variability as we see above.
## 2. Moving average
This technique takes the average of the passenger counts for the lass feww times only ie the predictions are made on the basis of the average of the last few points in stead of taking all the previosly known values.

I will try the rolling mean for the last 10,20,50 days and visualize the results.


```python
yhatavg = valid.copy()
yhatavg['movingavgforecast'] = Train['Count'].rolling(10).mean().iloc[-1] # average of last 10 observations.
plt.figure(figsize=(15,5)) 
plt.plot(Train['Count'], label='Train')
plt.plot(valid['Count'], label='Valid')
plt.plot(yhatavg['movingavgforecast'], label='Moving Average Forecast using 10 observations')
plt.legend(loc='best')
plt.show()

yhatavg = valid.copy()
yhatavg['movingavgforecast'] = Train['Count'].rolling(20).mean().iloc[-1] # average of last 10 observations.
plt.figure(figsize=(15,5)) 
plt.plot(Train['Count'], label='Train')
plt.plot(valid['Count'], label='Valid')
plt.plot(yhatavg['movingavgforecast'], label='Moving Average Forecast using 20 observations')
plt.legend(loc='best')
plt.show()

yhatavg = valid.copy()
yhatavg['movingavgforecast'] = Train['Count'].rolling(50).mean().iloc[-1] # average of last 10 observations.
plt.figure(figsize=(15,5)) 
plt.plot(Train['Count'], label='Train')
plt.plot(valid['Count'], label='Valid')
plt.plot(yhatavg['movingavgforecast'], label='Moving Average Forecast using 50 observations')
plt.legend(loc='best')
plt.show()

```


![png](/timeseriesFiles/output_55_0.png)



![png](/timeseriesFiles/output_55_1.png)



![png](/timeseriesFiles/output_55_2.png)


The predictions are getting weaker as i ad more observations.


```python
rms = sqrt(mean_squared_error(valid.Count,yhatavg.movingavgforecast))
print(rms)
```

    144.19175679986802


### iii. Simple Exponential Smoothing
* In this technique ,we asign larger weights to more recent observations than to observations from the distance past.
* The weiights decrease exponentially as observations come from futher in the past ,the smallest weights are associated with the oldest observations.


```python
from statsmodels.tsa.api import ExponentialSmoothing,SimpleExpSmoothing,Holt
yhatavg =valid.copy()
fit2 = SimpleExpSmoothing(np.asarray(Train['Count'])).fit(smoothing_level = 0.6,optimized = False)
yhatavg['SES'] = fit2.forecast(len(valid))
plt.figure(figsize=(16,8))
plt.plot(Train['Count'],label = 'Train')
plt.plot(yhatavg['SES'],label = 'SES')
plt.plot(valid['Count'],label = 'Valid')
plt.legend(loc = 'best')
plt.show()
```


![png](/timeseriesFiles/output_59_0.png)



```python
rms = sqrt(mean_squared_error(valid.Count,yhatavg.SES))
print(rms)
```

    113.43708111884514


### iv. Holt's Linear Trend Model
* It is an extensin of simple exponention smoothing to follow forecasting of data with a trend.
*  This method takes into account the trend of the dataset.The forecast function in this method is a function of level and trend.

So i will visualize the trend ,seosonality and error in the series.

Time series can be grouped into four parts.:
* Observed
* Trend
* Seosonal
* Residual


```python
import statsmodels.api as sm
sm.tsa.seasonal_decompose(Train.Count).plot()
result = sm.tsa.stattools.adfuller(train.Count)
plt.show()
```


![png](/timeseriesFiles/output_62_0.png)



```python
y_hat_avg = valid.copy()

fit1 = Holt(np.asarray(Train['Count'])).fit(smoothing_level = 0.3,smoothing_slope = 0.1)
y_hat_avg['Holt_linear'] = fit1.forecast(len(valid))

plt.figure(figsize=(16,8))
plt.plot(Train['Count'], label='Train')
plt.plot(valid['Count'], label='Valid')
plt.plot(y_hat_avg['Holt_linear'], label='Holt_linear')
plt.legend(loc='best')
plt.show()
```


![png](/timeseriesFiles/output_63_0.png)


This shows an increasing trend.

And now the rmse


```python
rms = sqrt(mean_squared_error(valid.Count,y_hat_avg.Holt_linear))
print(rms)
```

    112.94278345314041


This has reduced the rms.
The next steps i will be predicting the passenger count for the test dataset.

## 3. Holt linear Trend Model on daily time series.


```python
#loading the submissions file.
sub = pd.read_csv("/Users/admin/Downloads/timeseriesdata/Submission.csv")
#predicting for the test data set
predict = fit1.forecast(len(test))
#saving predictions in a new column
test['prediction'] = predict
```

Because these are are the daily predictions ,i will convert these predictions to hourly basis as follows.
* _Calculate the ratio of of pass count for each hour per day._
* _Calculate the average ratio of passenger count every hour._
* _Calculate hourly predictitions will multiply the daily prediction within hourly ratio._


```python
#hourly ratio count
train_original['ratio'] = train_original['Count']/train_original['Count'].sum()

#houly ratio
temp = train_original.groupby(['Hour'])['ratio'].sum()

# groupby to csv
pd.DataFrame(temp,columns = ['Hour','ratio']).to_csv('Groupby.csv')

temp2 = pd.read_csv("Groupby.csv")
temp2 = temp2.drop('Hour.1',1)

merge = pd.merge(test,test_original,on = ('day','month','year'),how = 'left')
merge['Hour'] = merge['Hour_y']
merge=merge.drop(['year', 'month', 'Datetime','Hour_x','Hour_y'], axis=1)

prediction = pd.merge(merge,temp2,on = 'Hour',how = 'left')

prediction['Count'] = prediction['prediction']*prediction['ratio']*24
prediction['ID'] = prediction['ID_y']
```


```python
submission = prediction.drop(['ID_x','day','ID_y','prediction','Hour','ratio'],axis = 1)
pd.DataFrame(submission,columns=['ID','Count']).to_csv("Holt.csv")
```

To be continued...
