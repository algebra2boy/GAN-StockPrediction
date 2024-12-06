import pandas as pd
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class Generator():
    def __init__(self):
        pass

    def SMA(self, data, windows):
        res = data.rolling(window=windows).mean()
        return res

    def EMA(self, data, windows):
        res = data.ewm(span=windows).mean()
        return res

    def MACD(self, data, long, short, windows):
        short_ = data.ewm(span=short).mean()
        long_ = data.ewm(span=long).mean()
        macd_ = short_ - long_
        res = macd_.ewm(span=windows).mean()
        return res

    def RSI(self, data, windows):
        delta = data.diff(1)
        up = delta.copy()
        down = delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        avg_up = up.rolling(window=windows).mean()
        avg_down = abs(down.rolling(window=windows).mean())  # Take absolute value
        # Handle division by zero
        rsi = np.where(avg_down != 0,
                      100 - (100 / (1 + (avg_up / avg_down))),
                      100)  # If avg_down is 0, RSI is 100
        return pd.Series(rsi, index=data.index)

    def atr(self, data_high, data_low, windows):
        range_ = data_high - data_low
        res = range_.rolling(window=windows).mean()
        return res

    def bollinger_band(self, data, windows):
        sma = data.rolling(window=windows).mean()
        std = data.rolling(window=windows).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        return upper, lower

    def rsv(self, data, windows):
        min_ = data.rolling(window=windows).min()
        max_ = data.rolling(window=windows).max()
        # Handle division by zero
        denominator = max_ - min_
        rsv = np.where(denominator != 0,
                      (data - min_) / denominator * 100,
                      50)  # Use 50 as default when max equals min
        return pd.Series(rsv, index=data.index)

# Data preprocessing
def preprocess_stock_data(data):
    # Drop any existing NaN values
    data = data.dropna()
    
    # Initialize Generator
    generator = Generator()
    
    # Add Percentage and Logarithmic Changes (with safety checks)
    data['pct_change'] = data['Close'].pct_change().fillna(0)
    # Handle log change safely
    data['log_change'] = np.log(data['Close'] / data['Close'].shift(1)).replace([np.inf, -np.inf], 0).fillna(0)
    
    # Technical Indicators
    data['7ma'] = generator.EMA(data['Close'], 7)
    data['14ma'] = generator.EMA(data['Close'], 14)
    data['21ma'] = generator.EMA(data['Close'], 21)
    data['7macd'] = generator.MACD(data['Close'], 3, 11, 7)
    data['14macd'] = generator.MACD(data['Close'], 7, 21, 14)
    data['7rsi'] = generator.RSI(data['Close'], 7)
    data['14rsi'] = generator.RSI(data['Close'], 14)
    data['21rsi'] = generator.RSI(data['Close'], 21)
    data['7atr'] = generator.atr(data['High'], data['Low'], 7)
    data['14atr'] = generator.atr(data['High'], data['Low'], 14)
    data['21atr'] = generator.atr(data['High'], data['Low'], 21)
    data['7upper'], data['7lower'] = generator.bollinger_band(data['Close'], 7)
    data['14upper'], data['14lower'] = generator.bollinger_band(data['Close'], 14)
    data['21upper'], data['21lower'] = generator.bollinger_band(data['Close'], 21)
    
    # Replace any remaining infinities with NaN and then forward/backward fill
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.fillna(method='ffill').fillna(method='bfill')
    
    # Normalize Selected Columns
    columns_to_normalize = ['Open', 'High', 'Low', 'Close', 'Volume', 'pct_change', 'log_change',
                          '7ma', '14ma', '21ma', '7macd', '14macd', '7rsi', '14rsi', '21rsi',
                          '7atr', '14atr', '21atr', '7upper', '7lower', '14upper', '14lower', 
                          '21upper', '21lower']
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])
    
    # Fourier Transform Features
    close_fft = np.fft.fft(data['Close'].values)
    fft_df = pd.DataFrame({'fft': close_fft})
    fft_df['absolute'] = fft_df['fft'].apply(np.abs)
    fft_df['angle'] = fft_df['fft'].apply(np.angle)
    
    # Retain important components
    for num_components in [3, 6, 9, 27, 81]:
        fft_filtered = np.copy(close_fft)
        fft_filtered[num_components:-num_components] = 0
        data[f'FT_{num_components}components'] = np.real(np.fft.ifft(fft_filtered))
    
    return data

# Usage:
data = pd.read_csv("stock_data.csv")
preprocessed_data = preprocess_stock_data(data)

#print number of columns and rows
print(preprocessed_data.shape)
preprocessed_data.to_csv("preprocessed_stock_data.csv")
# Name the index column "Date"
preprocessed_data.index.name = "Date"