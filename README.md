
# Stock Market Analysis and Prediction using LSTM
Stock Market Analysis and Prediction Using LSTM involves leveraging Long Short-Term Memory (LSTM), a type of recurrent neural network (RNN), for the analysis and prediction of stock prices. LSTM is particularly advantageous for processing sequential data such as time-series stock market information. The process entails using historical stock prices, trading volumes, market indicators, and other financial data as input for training the LSTM model. By learning complex patterns and dependencies within the historical data, the LSTM model becomes capable of forecasting future stock prices, thereby assisting investors, traders, and financial analysts in making informed decisions. It is important to note, however, that while LSTM models can provide valuable insights, their predictive accuracy may be impacted by market volatility, unforeseen events, and other external factors.

Key considerations in this domain include data preprocessing to ensure optimal LSTM performance, feature engineering for selecting relevant indicators, model evaluation using metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE), regularization techniques to prevent overfitting, and continuous model monitoring and retraining to adapt to changing market conditions and improve prediction accuracy over time.

In conclusion, stock market analysis and prediction using LSTM represents an impactful application of machine learning in the financial domain. While LSTM models can offer valuable insights into potential stock price trends, they should be used in conjunction with other fundamental and technical analysis methods for well-informed investment decisions.

<p align="centre">
    <img src="https://github.com/DURGABHAVANI66/Stock-Market-Analysis-and-Prediction-using-LSTM/assets/103325696/1e7ab2d0-3296-496f-b25c-fbdb897f2ad9" />
</p>

## 1. Change in price of the stock overtime


It imports necessary libraries and sets up the necessary configurations for visualizations and data retrieval.
It retrieves historical stock data for the companies listed in the tech_list  from Yahoo Finance using the pandas_datareader.
and  stores the retrieved stock data for each company in separate variables then assigns a corresponding company name to each dataset. The data is then concatenated into a single dataframe, df, for combined analysis.

``` python
pip install pandas
```

``` python
pip install pandas_datareader
```

``` python
pip install numpy
```

``` python
pip install keras
```

``` python
pip install tensorflow
```

``` python
pip install --upgrade pandas numpy keras tensorflow
```

``` text
installing the libraries in jupyter notebook use -----> %pip install
installing in colab use -----> !pip install
installing through the terminal use -----> pip install
```
`Importing the Data form yfinance `
``` python
pip install -q yfinance
```

``` python
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
%matplotlib inline

# For reading stock data from yahoo
from pandas_datareader.data import DataReader
import yfinance as yf
from pandas_datareader import data as pdr

yf.pdr_override()

# For time stamps
from datetime import datetime


# The tech stocks we'll use for this analysis
tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN']

# Set up End and Start times for data grab
tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN']

end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)

for stock in tech_list:
    globals()[stock] = yf.download(stock, start, end)
    

company_list = [AAPL, GOOG, MSFT, AMZN]
company_name = ["APPLE", "GOOGLE", "MICROSOFT", "AMAZON"]

for company, com_name in zip(company_list, company_name):
    company["company_name"] = com_name
    
df = pd.concat(company_list, axis=0)
df.tail(10)
```

###  Descriptive Statistics of Data 

The code .describe() is computing descriptive statistics for the stock data of company Inc. and displaying the summary output, providing insights into the distribution and characteristics of the data. 
``` python  
AAPL.describe()
```
```python
GOOG.describe()
```
``` python
MSFT.describe()
```
``` python
AMZN.describe()
```
### Information about the data
The code AAPL.info() is displaying essential information about the stock data for Apple Inc. (AAPL), including the data type, range of values, and memory usage. This provides insights into the structure and completeness of the dataset, such as the presence of missing values, and allows for a quick assessment of the data's quality and integrity.

``` python
AAPL.info()
```

### Closing Price  
The provided code segment creates a single figure with a 2x2 grid layout using matplotlib, where each subplot visualizes the adjusted close prices of a different technology company (AAPL, GOOG, MSFT, AMZN). The for loop iterates through the company_list and creates a subplot for each company, displaying its adjusted close price data over time. The visualization allows for a comparative analysis of the closing prices of the selected technology stocks. Adjustments are made to ensure the subplots are clearly labeled, and the layout is well-structured to enhance readability and comprehension.


``` python
plt.figure(figsize=(15, 10))
plt.subplots_adjust(top=1.25, bottom=1.2)

for i, company in enumerate(company_list, 1):
    plt.subplot(2, 2, i)
    company['Adj Close'].plot()
    plt.ylabel('Adj Close')
    plt.xlabel(None)
    plt.title(f"Closing Price of {tech_list[i - 1]}")
    
plt.tight_layout()
```
<p align='centre'> 
<img src= "https://github.com/DURGABHAVANI66/Stock-Market-Analysis-and-Prediction-using-LSTM/assets/103325696/0ba78041-f56e-4a25-acff-85ef8ee51213" />
</p>

### Volume of Sales

The provided code creates a 2x2 grid layout using matplotlib, displaying the volume data for each of the specified technology companies It iterates through the company_list and generates a separate subplot for each company, where the volume of shares traded is plotted over time. Each subplot is labeled with the corresponding company's name and provides a visual comparison of sales volumes. Adjustments are made to ensure that the subplots are clearly labeled, and the layout is well-structured for optimal readability and analysis.

``` python
plt.figure(figsize=(15, 10))
plt.subplots_adjust(top=1.25, bottom=1.2)

for i, company in enumerate(company_list, 1):
    plt.subplot(2, 2, i)
    company['Volume'].plot()
    plt.ylabel('Volume')
    plt.xlabel(None)
    plt.title(f"Sales Volume for {tech_list[i - 1]}")
    
plt.tight_layout()
```

## 2. Moving average of the stocks 

The provided code computes the moving averages for the adjusted close prices of the specified tech companies   over varying window periods (10, 20, and 50 days). Subsequently, it creates a 2x2 grid of subplots, each displaying the adjusted close prices alongside the computed moving averages for these different companies. The subplots are arranged to provide a comparative analysis of the adjusted close prices and their respective moving averages over the specified window periods for each tech company. Adjustments are made to ensure clear labeling and a well-structured layout, facilitating an effective visual assessment of the moving averages across the different companies.


``` python
ma_day = [10, 20, 50]

for ma in ma_day:
    for company in company_list:
        column_name = f"MA for {ma} days"
        company[column_name] = company['Adj Close'].rolling(ma).mean()
        

fig, axes = plt.subplots(nrows=2, ncols=2)
fig.set_figheight(10)
fig.set_figwidth(15)

AAPL[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[0,0])
axes[0,0].set_title('APPLE')

GOOG[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[0,1])
axes[0,1].set_title('GOOGLE')

MSFT[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[1,0])
axes[1,0].set_title('MICROSOFT')

AMZN[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[1,1])
axes[1,1].set_title('AMAZON')

fig.tight_layout()
```

<p align='centre'> 
    <img src = "https://github.com/DURGABHAVANI66/Stock-Market-Analysis-and-Prediction-using-LSTM/assets/103325696/8e53bdb4-78bf-4ea7-88fa-9ed3cf032e16)"/>
</p>

## 3. The daily return of the stock on average

 
The provided code calculates the daily return percentage for each tech company's stock prices and then generates visualizations to illustrate the distribution of the daily returns as histograms. It first computes the daily return percentage for each company's adjusted close prices and then creates a 2x2 grid of subplots to display the daily return percentages for each tech company, along with their respective stock symbols. Next, it plots histograms of the daily return percentages for each company in a single 2x2 grid, effectively comparing the distribution of the daily returns across the selected tech companies. Finally, adjustments are made to ensure clear labeling and a well-structured layout, facilitating an effective comparison of the daily return distributions.


``` python

for company in company_list:
    company['Daily Return'] = company['Adj Close'].pct_change()

# Then we'll plot the daily return percentage
fig, axes = plt.subplots(nrows=2, ncols=2)
fig.set_figheight(10)
fig.set_figwidth(15)

AAPL['Daily Return'].plot(ax=axes[0,0], legend=True, linestyle='--', marker='o')
axes[0,0].set_title('APPLE')

GOOG['Daily Return'].plot(ax=axes[0,1], legend=True, linestyle='--', marker='o')
axes[0,1].set_title('GOOGLE')

MSFT['Daily Return'].plot(ax=axes[1,0], legend=True, linestyle='--', marker='o')
axes[1,0].set_title('MICROSOFT')

AMZN['Daily Return'].plot(ax=axes[1,1], legend=True, linestyle='--', marker='o')
axes[1,1  ].set_title('AMAZON')

fig.tight_layout()

plt.figure(figsize=(12, 9))

for i, company in enumerate(company_list, 1):
    plt.subplot(2, 2, i)
    company['Daily Return'].hist(bins=50)
    plt.xlabel('Daily Return')
    plt.ylabel('Counts')
    plt.title(f'{company_name[i - 1]}')
    
plt.tight_layout()
```


## 4.Corelation between different stocks closing prices 

Correlation is a statistic that measures the degree to which two variables move in relation to each other which has a value that must fall between -1.0 and +1.0. Correlation measures association, but doesn’t show if x causes y or vice versa — or if the association is caused by a third factor[1].

Now what if we wanted to analyze the returns of all the stocks in our list? Let's go ahead and build a DataFrame with all the ['Close'] columns for each of the stocks dataframes.

``` python
# Grab all the closing prices for the tech stock list into one DataFrame

closing_df = pdr.get_data_yahoo(tech_list, start=start, end=end)['Adj Close']

# Make a new tech returns DataFrame
tech_rets = closing_df.pct_change()
tech_rets.head()
```

Now we can compare the daily percentage return of two stocks to check how correlated. First let's see a sotck compared to itself.

``` python
# Comparing Google to itself should show a perfectly linear relationship
sns.jointplot(x='GOOG', y='GOOG', data=tech_rets, kind='scatter', color='seagreen')
```

``` python
# We'll use joinplot to compare the daily returns of Google and Microsoft
sns.jointplot(x='GOOG', y='MSFT', data=tech_rets, kind='scatter')
```
So now we can see that if two stocks are perfectly (and positivley) correlated with each other a linear relationship bewteen its daily return values should occur.

Seaborn and pandas make it very easy to repeat this comparison analysis for every possible combination of stocks in our technology stock ticker list. We can use sns.pairplot() to automatically create this plot
``` python
# We can simply call pairplot on our DataFrame for an automatic visual analysis 
# of all the comparisons

sns.pairplot(tech_rets, kind='reg')
```
Above we can see all the relationships on daily returns between all the stocks. A quick glance shows an interesting correlation between Google and Amazon daily returns. It might be interesting to investigate that individual comaprison.

While the simplicity of just calling sns.pairplot() is fantastic we can also use sns.PairGrid() for full control of the figure, including what kind of plots go in the diagonal, the upper triangle, and the lower triangle. Below is an example of utilizing the full power of seaborn to achieve this result

``` python
# Set up our figure by naming it returns_fig, call PairPLot on the DataFrame
return_fig = sns.PairGrid(tech_rets.dropna())

# Using map_upper we can specify what the upper triangle will look like.
return_fig.map_upper(plt.scatter, color='purple')

# We can also define the lower triangle in the figure, inclufing the plot type (kde) 
# or the color map (BluePurple)
return_fig.map_lower(sns.kdeplot, cmap='cool_d')

# Finally we'll define the diagonal as a series of histogram plots of the daily return
return_fig.map_diag(plt.hist, bins=30)
```

``` python
# Set up our figure by naming it returns_fig, call PairPLot on the DataFrame
returns_fig = sns.PairGrid(closing_df)

# Using map_upper we can specify what the upper triangle will look like.
returns_fig.map_upper(plt.scatter,color='purple')

# We can also define the lower triangle in the figure, inclufing the plot type (kde) or the color map (BluePurple)
returns_fig.map_lower(sns.kdeplot,cmap='cool_d')

# Finally we'll define the diagonal as a series of histogram plots of the daily return
returns_fig.map_diag(plt.hist,bins=30)
```
Finally, we could also do a correlation plot, to get actual numerical values for the correlation between the stocks' daily return values. By comparing the closing prices, we see an interesting relationship between Microsoft and Apple.


``` python
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
sns.heatmap(tech_rets.corr(), annot=True, cmap='summer')
plt.title('Correlation of stock return')

plt.subplot(2, 2, 2)
sns.heatmap(closing_df.corr(), annot=True, cmap='summer')
plt.title('Correlation of stock closing price')
```

Just like we suspected in our PairPlot we see here numerically and visually that Microsoft and Amazon had the strongest correlation of daily stock return. It's also interesting to see that all the technology comapnies are positively correlated.


## 5.By investing in a specific stock, we are exposing ourselves to a potential level of risk.

There are many ways we can quantify risk, one of the most basic ways using the information we've gathered on daily percentage returns is by comparing the expected return with the standard deviation of the daily returns.

``` python
rets = tech_rets.dropna()

area = np.pi * 20

plt.figure(figsize=(10, 8))
plt.scatter(rets.mean(), rets.std(), s=area)
plt.xlabel('Expected return')
plt.ylabel('Risk')

for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(label, xy=(x, y), xytext=(50, 50), textcoords='offset points', ha='right', va='bottom', 
                 arrowprops=dict(arrowstyle='-', color='blue', connectionstyle='arc3,rad=-0.3'))
```

<p align ='centre'> 
<img src ="https://github.com/DURGABHAVANI66/Stock-Market-Analysis-and-Prediction-using-LSTM/assets/103325696/1c209b05-3202-4e3b-a325-679d05ea9deb"/>
</p>




## 6.Prediction of the closing price stock price of APPLE inc
The provided code conducts a series of crucial tasks to develop and assess a Long Short-Term Memory (LSTM) model tailored for predicting stock prices, specifically for Apple Inc. (AAPL). Commencing with the retrieval and visualization of historical stock price data sourced from Yahoo Finance, the code prepares the data for modeling. Employing Min-Max scaling, the code preprocesses the stock price information and establishes a training dataset encompassing 95% of the available data. Subsequently, the code defines, compiles, and trains an LSTM model via the Keras library, utilizing the Adam optimizer and mean squared error loss function. It then proceeds to prepare the testing dataset, make predictions employing the trained model, and evaluate performance using the root mean squared error (RMSE). Lastly, the code integrates a visualization component, allowing for the comparative depiction of actual and predicted stock prices. Overall, this comprehensive workflow effectively harnesses LSTM technology for the precise and efficient prediction of AAPL stock prices.

``` python  
# Get the stock quote
df = pdr.get_data_yahoo('AAPL', start='2012-01-01', end=datetime.now())
# Show teh data
df
```


``` python  
plt.figure(figsize=(16,6))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.show()

```
``` python  
# Create a new dataframe with only the 'Close column 
data = df.filter(['Close'])
# Convert the dataframe to a numpy array
dataset = data.values
# Get the number of rows to train the model on
training_data_len = int(np.ceil( len(dataset) * .95 ))

training_data_len
```
``` python  
# Scale the data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

scaled_data
```
``` python  
# Create the training data set 
# Create the scaled training data set
train_data = scaled_data[0:int(training_data_len), :]
# Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i<= 61:
        print(x_train)
        print(y_train)
        print()
        
# Convert the x_train and y_train to numpy arrays 
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# x_train.shape
```
``` python  
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Build the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)
```
``` python  
# Create the testing data set
# Create a new array containing scaled values from index 1543 to 2002 
test_data = scaled_data[training_data_len - 60: , :]
# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    
# Convert the data to a numpy array
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

# Get the models predicted price values 
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
rmse
```
``` python  
# Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
# Visualize the data
plt.figure(figsize=(16,6))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()
```
``` python  
# Show the valid and predicted prices
valid
```



## Summary
 The above project and the code   conducts a comprehensive analysis of stock market data for tech companies, encompassing data retrieval from Yahoo Finance and the creation of visualizations. It involves the retrieval of stock data, combined with the visualization of adjusted closing prices and sales volume for Apple, Google, Microsoft, and Amazon. Moving averages for different periods and daily return percentages are computed and displayed through various visualizations, including line plots, histograms, and correlation heatmaps. Additionally, the code utilizes the LSTM model to predict Apple Inc.'s stock prices, encompassing data preprocessing, model building, training, error evaluation, and visualization of model performance. Overall, the code provides a detailed exploration of stock data analysis and includes advanced techniques for stock price prediction using LSTM technology.

<p align='centre'>

<img src="https://github.com/DURGABHAVANI66/Stock-Market-Analysis-and-Prediction-using-LSTM/assets/103325696/79ad1c34-dec5-4fbb-abed-24538b5b5ffe" />
</p>






