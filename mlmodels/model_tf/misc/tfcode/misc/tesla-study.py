#!/usr/bin/env python
# coding: utf-8

# In[2]:


from datetime import date

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib.dates import date2num
from matplotlib.finance import candlestick_ohlc
from sklearn import linear_model
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, adfuller, pacf

get_ipython().run_line_magic("matplotlib", "inline")
plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (15, 10)


# In[3]:


tesla = pd.read_csv("TSLA.csv")
tesla = tesla[["Date", "Open", "High", "Low", "Close"]]
print(tesla.shape)
tesla.head()


# In[4]:


tesla_2011 = pd.read_csv("TSLA-2011.csv")
tesla_2011 = tesla_2011[["Date", "Open", "High", "Low", "Close"]]
print(tesla_2011.shape)
tesla_2011.head()


# In[5]:


df_cp = tesla.copy()
df_cp.Date = date2num(pd.to_datetime(tesla.Date).dt.to_pydatetime())
ax1 = plt.subplot2grid((1, 1), (0, 0))
candlestick_ohlc(ax1, df_cp.values, width=0.4, colorup="#77d879", colordown="#db3f3f", alpha=2)
x_range = np.arange(df_cp.shape[0])
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
ax1.grid(True)
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()


# In[6]:


ax1 = plt.subplot2grid((1, 1), (0, 0))
ret = candlestick_ohlc(
    ax1, df_cp.iloc[:100, :].values, width=0.4, colorup="#77d879", colordown="#db3f3f", alpha=2
)
x_range = np.arange(df_cp.shape[0])
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
ax1.grid(True)
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()


# In[7]:


tesla.Close.plot()


# In[8]:


tesla_2011.Close.plot()


# In[9]:


tesla.plot(kind="line", y=["Open", "High", "Low", "Close"])


# In[10]:


tesla_2011.plot(kind="line", y=["Open", "High", "Low", "Close"])


# In[11]:


tesla_2011["months"] = pd.DatetimeIndex(tesla_2011["Date"]).month
tesla_2011["year"] = pd.DatetimeIndex(tesla_2011["Date"]).year
tesla_2011.head()


# In[12]:


teslaPivot = pd.pivot_table(tesla_2011, values="Close", columns="year", index="months")


# In[13]:


teslaPivot.head()


# In[14]:


teslaPivot.plot()


# In[15]:


teslaPivot.plot(subplots=True, figsize=(15, 15), layout=(4, 4), sharey=True)


# In[16]:


tesla.Close.plot(kind="hist", bins=30)


# In[17]:


tesla["Closelog"] = np.log(tesla.Close)
tesla.head()


# In[18]:


tesla.Closelog.plot(kind="hist", bins=30)


# In[19]:


tesla.Closelog.plot()


# In[20]:


model_mean_pred = tesla.Closelog.mean()
# reverse log e
tesla["Closemean"] = np.exp(model_mean_pred)
tesla.plot(kind="line", x="Date", y=["Close", "Closemean"])


# In[21]:


x = np.arange(tesla.shape[0]).reshape((-1, 1))
y = tesla.Close.values.reshape((-1, 1))
reg = linear_model.LinearRegression()
pred = reg.fit(x, y).predict(x)


# In[22]:


tesla["linear"] = pred
tesla.plot(kind="line", x="Date", y=["Close", "Closemean", "linear"])


# In[23]:


tesla.Date = pd.DatetimeIndex(tesla.Date)
tesla.index = pd.PeriodIndex(tesla.Date, freq="D")
tesla = tesla.sort_values(by="Date")
tesla.head()


# In[24]:


tesla["timeIndex"] = tesla.Date - tesla.Date.min()
tesla["timeIndex"] = tesla["timeIndex"] / np.timedelta64(1, "D")
tesla.head()


# In[25]:


tesla["timeIndex"] = tesla["timeIndex"].round(0).astype(int)
tesla.tail()


# In[26]:


# In[27]:


model_linear = smf.ols("Closelog ~ timeIndex", data=tesla).fit()
model_linear.summary()


# In[28]:


model_linear.params


# In[29]:


model_linear_pred = model_linear.predict()
model_linear_pred.shape


# In[30]:


tesla["linear_stats"] = model_linear_pred
tesla.head()


# In[31]:


tesla.plot(kind="line", x="timeIndex", y=["Closelog", "linear_stats"])


# In[32]:


model_linear.resid.plot(kind="bar").get_xaxis().set_visible(False)


# In[33]:


model_linear_forecast_auto = model_linear.predict(exog=pd.DataFrame(dict(timeIndex=252), index=[0]))
model_linear_forecast_auto


# In[34]:


tesla["pricelinear"] = np.exp(model_linear_pred)
tesla.head()


# In[35]:


tesla.plot(kind="line", x="timeIndex", y=["Close", "Closemean", "pricelinear"])


# In[36]:


tesla["CloselogShift1"] = tesla.Closelog.shift()
tesla.head()


# In[37]:


tesla.plot(kind="scatter", y="Closelog", x="CloselogShift1", s=50)


# In[38]:


tesla["CloselogDiff"] = tesla.Closelog - tesla.CloselogShift1
tesla.CloselogDiff.plot()


# In[61]:


tesla["CloseRandom"] = np.exp(tesla.CloselogShift1)
tesla.head()


# In[55]:


def adf(ts):
    rolmean = pd.rolling_mean(ts, window=12)
    rolstd = pd.rolling_std(ts, window=12)

    orig = plt.plot(ts.values, color="blue", label="Original")
    mean = plt.plot(rolmean.values, color="red", label="Rolling Mean")
    std = plt.plot(rolstd.values, color="black", label="Rolling Std")
    plt.legend(loc="best")
    plt.title("Rolling Mean & Standard Deviation")
    plt.show(block=False)

    adftest = adfuller(ts, autolag="AIC")
    adfoutput = pd.Series(
        adftest[0:4],
        index=["Test Statistic", "p-value", "# of Lags Used", "Number of Observations Used"],
    )
    for key, value in adftest[4].items():
        adfoutput["Critical Value (%s)" % key] = value
    return adfoutput


# In[40]:


tesla["CloselogMA12"] = pd.rolling_mean(tesla.Closelog, window=12)
tesla.plot(kind="line", y=["CloselogMA12", "Closelog"])


# In[56]:


ts = tesla.Closelog - tesla.CloselogMA12
ts.dropna(inplace=True)
adf(ts)


# if test statistic < critical value (any), we can assume this data is stationary.

# In[57]:


half_life = 12
tesla["CloselogExp12"] = pd.ewma(tesla.Closelog, halflife=half_life)
1 - np.exp(np.log(0.5) / half_life)


# In[58]:


tesla.plot(kind="line", y=["CloselogExp12", "Closelog"])


# In[63]:


tesla["CloseExp12"] = np.exp(tesla.CloselogExp12)
tesla.tail()


# In[65]:


tesla.plot(
    kind="line", x="timeIndex", y=["Close", "Closemean", "pricelinear", "CloseRandom", "CloseExp12"]
)


# In[67]:


ts = tesla.Closelog - tesla.CloselogExp12
ts.dropna(inplace=True)
adf(ts)


# In[68]:


tesla.index = tesla.index.to_datetime()


# In[80]:


decomposition = seasonal_decompose(tesla.Closelog, freq=31)


# In[81]:


decomposition.plot()


# In[82]:


ts = tesla.Closelog
ts_diff = tesla.CloselogDiff
ts_diff.dropna(inplace=True)


# In[83]:


lag_acf = acf(ts_diff, nlags=20)


# In[84]:


ACF = pd.Series(lag_acf)


# In[85]:


ACF.plot(kind="bar")
