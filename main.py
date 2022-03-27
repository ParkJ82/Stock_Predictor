# from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
# import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


class StockMarketPrediction:
    def __init__(self, data_source, ticker = None, api_key_input = None, file_path = None, file_name = None):
        downloaded_data = DownloadData(data_source, ticker, api_key_input, file_path, file_name)
        splitted_data = SplittingData(downloaded_data)
        normalized_data = NormalizingData(splitted_data)
        prediction_data = OneStepAheadPrediction(normalized_data, downloaded_data)


class DownloadData:
    def __init__(self, data_source, ticker = None, api_key_input = None, file_path = None, file_name = None):

        # Initialize data source
        self.data_source = data_source

        # Data source is Alphavantage
        if self.data_source == "alphavantage":
            self.getting_data_alphavantage(ticker, api_key_input)

        # Data source is Kaggle
        else:
            self.getting_data_kaggle(file_path, file_name)

        # Sorting the value by date
        self.sort_data_by_date()

        # Visualize Data
        self.data_visualization()

    def getting_data_alphavantage(self, ticker, api_key_input):

        # JSON file with all the stock market data for AAL from the last 20 years
        self.url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s" \
                     % (ticker, api_key_input)

        # Save data to file
        self.file_to_save = "stock_market_data-%s.csv"%ticker

        # Storing Process
        # Execute if data is not already in path
        if not os.path.exists(self.file_to_save):
            self.create_csv()

        # Execute if data is already in path
        else:
            print("File already exists. Loading data from CSV")

            # Load the previous CSV file
            self.df = pd.read_csv(self.file_to_save)

    def getting_data_kaggle(self, file_path, file_name):

        # Creating the spreadsheet file
        self.df = pd.read_csv(os.path.join(file_path, file_name), delimiter=",", usecols=["Date", "Open", "High",
                                                                                          "Low", "Close"])
        print("Loaded data from the Kaggle repository")

    def create_csv(self):

        # Open the url string
        with urllib.request.urlopen(self.url_string) as url:

            # Extract data from whole url
            self.stock_data = json.loads(url.read().decode())

            # Extract stock market data from whole url
            self.stock_data = self.stock_data["Time Series (Daily)"]

        # Initialize columns
        self.df = pd.DataFrame(columns=["Date", "Low", "High", "Close", "Open"])

        # Store all the data (CSV)
        self.store_data_csv()

        print("Data saved to: %s" % self.file_to_save)
        self.df.to_csv(self.file_to_save)

    def store_data_csv(self):

        # Rotate through rows
        for current_date, current_price in self.stock_data.items():

            # Print date
            date = dt.datetime.strptime(current_date, "%Y-%m-%d")

            # Generate Row
            data_row = [date.date(), float(current_price["3. low"]), float(current_price["2. high"]),
                        float((current_price["4. close"])), float(current_price["1. open"])]

            # Insert one day's stock at a bottommost row
            self.df.loc[-1, :] = data_row

            # Increase index by 1
            self.df.index = self.df.index + 1

        self.current_date = current_date

    def sort_data_by_date(self):

        # Sort the values by date
        self.df = self.df.sort_values("Date")

        # Double check results
        self.df.head()

    def data_visualization(self):

        # Establish height and width
        height = int(input("Height of visualization (in inches): "))
        width = int(input("Width of visualization (in inches): "))
        plt.figure(figsize=(height, width))

        # Establish domain and values
        plt.plot(range(self.df.shape[0]), (self.df["Low"] + self.df["High"])/2.0)

        # Establish x-axis labels
        plt.xticks(range(0, self.df.shape[0], 500), self.df["Date"].loc[::500], rotation=45)

        # Establish x-axis/y-axis big label
        plt.xlabel("Date", fontsize=18)
        plt.ylabel("Mid Price", fontsize=18)

        # Show entire resulting plot
        plt.show()


class SplittingData:
    def __init__(self, downloaded_data):

        # Calculating mid_prices
        self.high_prices = downloaded_data.df.loc[:, "High"].to_numpy()
        self.low_prices = downloaded_data.df.loc[:, "Low"].to_numpy()
        self.mid_prices = (self.high_prices + self.low_prices) / 2.0

        # Dividing train and test data
        self.train_data = self.mid_prices[:3500]
        self.test_data = self.mid_prices[3500:]


class NormalizingData:
    def __init__(self, splitted_data):

        # Reshape matrix to have a single column
        self.train_data = splitted_data.train_data.reshape(-1, 1)
        self.test_data = splitted_data.test_data.reshape(-1, 1)

        # Smooth out all the windows
        self.smooth_windows()

    def smooth_windows(self):
        # Normalize every data in the region of 0 and 1
        scaler = MinMaxScaler()

        # Initialize window size
        smoothing_window_size = 1000

        # Repeat through batches and normalize each window
        self.normalize_each_window(scaler, smoothing_window_size)

        # Smooth every data through exponential moving average smoothing
        self.exponential_moving_average_smoothing()

    def normalize_each_window(self, scaler, smoothing_window_size):
        # Repeat through window batches
        for di in range(0, 3000, smoothing_window_size):

            # Pre-compute min and max
            scaler.fit(self.train_data[di:di+smoothing_window_size, :])

            # Fit selected data into range of min and max
            self.train_data[di:di+smoothing_window_size, :] \
                = scaler.transform(self.train_data[di:di+smoothing_window_size, :])

        # Normalize last bit of remaining data
        scaler.fit(self.train_data[di+smoothing_window_size:, :])
        self.train_data[di+smoothing_window_size:, :] = scaler.transform(self.train_data[di+smoothing_window_size:, :])

        # Reshape train data
        self.train_data = self.train_data.reshape(-1)

        # Reshape and normalize test data
        self.test_data = scaler.transform(self.test_data).reshape(-1)

    def exponential_moving_average_smoothing(self):

        # Initialize exponential moving average and gamma parameter
        EMA = 0.0
        gamma = 0.1

        # Repeat through all the data
        for ti in range(3500):

            # Update exponential moving average based on current point
            EMA = gamma * self.train_data[ti] + (1 - gamma) * EMA

            # Update the current point
            self.train_data[ti] = EMA

        # Concatenate all the data
        self.all_mid_data = np.concatenate([self.train_data, self.test_data], axis=0)

class OneStepAheadPrediction:
    def __init__(self, normalized_data, downloaded_data):

        # Get train data from normalized_data/downloaded_data class
        self.train_data = normalized_data.train_data
        self.downloaded_data = downloaded_data

        # Get mid data from normalized_data class
        self.all_mid_data = normalized_data.all_mid_data

        # Set window size to consider
        self.window_size = 100

        # Initiate empty list of average predictions
        self.avg_predictions = []

        # Initiate empty list of average x
        self.avg_x = []

        # Initiate mean square error
        self.mse_errors = []

        # N represents the size of the train data
        self.N = self.train_data.size

        # Get standard average (NOT USING)
        # # self.standard_average()

        # Get exponential moving average
        self.exponential_moving_average()

        # Visualize plot
        self.data_visualization()

    def standard_average(self):

        # Go through the training sets in window size -> train data size
        for pred_idx in range(self.window_size, self.N):

            # If the current data is over the max training data
            if pred_idx >= self.N:

                # Go to future data and get the next day's date
                date = dt.datetime.strptime(self.downloaded_data.current_date, "%Y-%m-%d").date() + dt.timedelta(days=1)

                # If the current data is not over the max training data
            else:

                # Get the date of the current data
                date = self.downloaded_data.df.loc[pred_idx, "Date"]

            # Get average prediction from current date's prediction - window size to current date's prediction
            self.avg_predictions.append(np.mean(self.train_data[pred_idx - self.window_size : pred_idx]))

            # Get the square error of the prediction
            self.mse_errors.append((self.avg_predictions[-1] - self.train_data[pred_idx]) ** 2)

            # Append the current date
            self.avg_x.append(date)

        # Print the MSE error (divide average mse errors by 0.5)
        print("MSE error for standard averaging: %.5f"%(0.5*np.mean(self.mse_errors)))

    def exponential_moving_average(self):

        # Window size does not exist
        self.window_size = 0

        # Initiate empty list of running average predictions
        self.avg_predictions = []

        # Initiate empty list of running average x
        self.run_avg_x = []

        # Initiate running mean and appeand it to first prediction
        running_mean = 0.0
        self.avg_predictions.append(running_mean)

        # Initiate decay
        decay = 0.99

        # Run through all the training data
        for pred_idx in range(1, self.N):

            # Calculate current running mean through exponential moving average and append it to predictions
            running_mean = running_mean * decay + (1.0 - decay) * self.train_data[pred_idx - 1]
            self.avg_predictions.append(running_mean)

            # Get the square error of the prediction
            self.mse_errors.append((self.avg_predictions[-1] - self.train_data[pred_idx]) ** 2)

            # Append the current date
            date = self.downloaded_data.df.loc[pred_idx, "Date"]
            self.run_avg_x.append(date)

        # Print the MSE error (divide average mse errors by 0.5)
        print("MSE error for standard averaging: %.5f" % (0.5 * np.mean(self.mse_errors)))

    def data_visualization(self):
        # Establish height and width
        height = int(input("Height of visualization (in inches): "))
        width = int(input("Width of visualization (in inches): "))
        plt.figure(figsize=(height, width))

        # Establish domain and values of original data
        plt.plot(range(self.downloaded_data.df.shape[0]), self.all_mid_data, color = "b", label = "True")

        # Establish domain and values of predicted data
        plt.plot(range(self.window_size, self.N), self.avg_predictions, color="orange", label="Prediction")

        # Establish x-axis labels
        plt.xticks(range(0, self.downloaded_data.df.shape[0], 50), self.downloaded_data.df["Date"].loc[::50], rotation=45)

        # Establish x-axis/y-axis big label
        plt.xlabel("Date")
        plt.ylabel("Mid Price")

        # Establish a legend
        plt.legend(fontsize=18)

        # Show entire resulting plot
        plt.show()




StockMarketPrediction("alphavantage", ticker = "AAL", api_key_input="PISC90JUIZAQBLVT")