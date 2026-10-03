import pandas as pd
import numpy as np
import os


# =========================================================
# PROJECT PATH
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)


# =========================================================
# FILE PATHS
# =========================================================

INPUT_FILE = os.path.join(
    DATA_DIR,
    "Stock_Prices.csv"
)

OUTPUT_FILE = os.path.join(
    DATA_DIR,
    "Stock_Prices_Features.csv"
)


# =========================================================
# CHECK FILE
# =========================================================

if not os.path.exists(INPUT_FILE):

    print()
    print("ERROR: Stock_Prices.csv not found.")
    print()
    print("Expected location:")
    print(INPUT_FILE)
    print()

    raise SystemExit


# =========================================================
# LOAD DATA
# =========================================================

print()
print("==============================================")
print("   STOCK MARKET FEATURE ENGINEERING")
print("==============================================")
print()

print("Loading stock price data...")

df = pd.read_csv(
    INPUT_FILE
)

print(
    "Records Loaded:",
    len(df)
)


# =========================================================
# 1. DATA QUALITY CHECK
# =========================================================

print()
print("----------------------------------------------")
print("1. DATA QUALITY CHECK")
print("----------------------------------------------")


# ---------------------------------------------------------
# Columns
# ---------------------------------------------------------

print()
print("Columns:")

for column in df.columns:

    print("-", column)


# ---------------------------------------------------------
# Missing Values
# ---------------------------------------------------------

print()
print("Missing Values:")

missing_values = df.isnull().sum()

print(
    missing_values
)


# ---------------------------------------------------------
# Duplicate Rows
# ---------------------------------------------------------

duplicate_count = df.duplicated().sum()

print()
print(
    "Duplicate Rows:",
    duplicate_count
)


# =========================================================
# 2. DATA CLEANING
# =========================================================

print()
print("----------------------------------------------")
print("2. DATA CLEANING")
print("----------------------------------------------")


# ---------------------------------------------------------
# Convert Date
# ---------------------------------------------------------

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)


# ---------------------------------------------------------
# Numeric Columns
# ---------------------------------------------------------

numeric_columns = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Daily_Return"
]


for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ---------------------------------------------------------
# Remove Invalid Dates
# ---------------------------------------------------------

df = df.dropna(
    subset=[
        "Date"
    ]
)


# ---------------------------------------------------------
# Remove Invalid Price Rows
# ---------------------------------------------------------

df = df[
    (df["Open"] > 0) &
    (df["High"] > 0) &
    (df["Low"] > 0) &
    (df["Close"] > 0)
]


# ---------------------------------------------------------
# Remove Duplicate Stock-Date Records
# ---------------------------------------------------------

df = df.drop_duplicates(
    subset=[
        "Ticker",
        "Date"
    ]
)


# ---------------------------------------------------------
# Sort Data
# ---------------------------------------------------------

df = df.sort_values(
    [
        "Ticker",
        "Date"
    ]
).reset_index(
    drop=True
)


print()
print(
    "Records After Cleaning:",
    len(df)
)


# =========================================================
# 3. TECHNICAL FEATURE ENGINEERING
# =========================================================

print()
print("----------------------------------------------")
print("3. TECHNICAL FEATURE ENGINEERING")
print("----------------------------------------------")


# =========================================================
# DAILY RETURN
# =========================================================

df["Daily_Return"] = (
    df.groupby("Ticker")["Close"]
    .pct_change()
    * 100
)


# =========================================================
# PREVIOUS CLOSE
# =========================================================

df["Previous_Close"] = (
    df.groupby("Ticker")["Close"]
    .shift(1)
)


# =========================================================
# TRADING RANGE
# =========================================================

df["Trading_Range"] = (
    df["High"] -
    df["Low"]
)


# =========================================================
# TRADING RANGE %
# =========================================================

df["Trading_Range_Percent"] = (
    (
        df["High"] -
        df["Low"]
    )
    /
    df["Previous_Close"]
    * 100
)


# =========================================================
# 20-DAY MOVING AVERAGE
# =========================================================

df["MA_20"] = (
    df.groupby("Ticker")["Close"]
    .transform(
        lambda x:
        x.rolling(
            window=20,
            min_periods=20
        ).mean()
    )
)


# =========================================================
# 50-DAY MOVING AVERAGE
# =========================================================

df["MA_50"] = (
    df.groupby("Ticker")["Close"]
    .transform(
        lambda x:
        x.rolling(
            window=50,
            min_periods=50
        ).mean()
    )
)


# =========================================================
# 20-DAY VOLATILITY
# =========================================================

df["Volatility_20D"] = (
    df.groupby("Ticker")["Daily_Return"]
    .transform(
        lambda x:
        x.rolling(
            window=20,
            min_periods=20
        ).std()
    )
)


# =========================================================
# VOLUME CHANGE %
# =========================================================

df["Volume_Change_Percent"] = (
    df.groupby("Ticker")["Volume"]
    .pct_change()
    * 100
)


# =========================================================
# CUMULATIVE RETURN
# =========================================================

df["Cumulative_Return_Percent"] = (
    df.groupby("Ticker")["Close"]
    .transform(
        lambda x:
        (
            x / x.iloc[0] - 1
        ) * 100
    )
)


# =========================================================
# RUNNING MAXIMUM
# =========================================================

df["Running_Max"] = (
    df.groupby("Ticker")["Close"]
    .cummax()
)


# =========================================================
# DRAWDOWN %
# =========================================================

df["Drawdown_Percent"] = (
    (
        df["Close"] -
        df["Running_Max"]
    )
    /
    df["Running_Max"]
    * 100
)


# =========================================================
# 4. RSI - RELATIVE STRENGTH INDEX
# =========================================================

print()
print("Calculating RSI...")


def calculate_rsi(
    prices,
    period=14
):

    delta = prices.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    average_gain = (
        gain.rolling(
            window=period,
            min_periods=period
        ).mean()
    )

    average_loss = (
        loss.rolling(
            window=period,
            min_periods=period
        ).mean()
    )

    rs = (
        average_gain /
        average_loss
    )

    rsi = (
        100 -
        (
            100 /
            (1 + rs)
        )
    )

    return rsi


df["RSI_14"] = (
    df.groupby("Ticker")["Close"]
    .transform(
        lambda x:
        calculate_rsi(x)
    )
)


# =========================================================
# 5. MACD
# =========================================================

print(
    "Calculating MACD..."
)


# ---------------------------------------------------------
# EMA 12
# ---------------------------------------------------------

df["EMA_12"] = (
    df.groupby("Ticker")["Close"]
    .transform(
        lambda x:
        x.ewm(
            span=12,
            adjust=False
        ).mean()
    )
)


# ---------------------------------------------------------
# EMA 26
# ---------------------------------------------------------

df["EMA_26"] = (
    df.groupby("Ticker")["Close"]
    .transform(
        lambda x:
        x.ewm(
            span=26,
            adjust=False
        ).mean()
    )
)


# ---------------------------------------------------------
# MACD Line
# ---------------------------------------------------------

df["MACD"] = (
    df["EMA_12"] -
    df["EMA_26"]
)


# ---------------------------------------------------------
# MACD Signal
# ---------------------------------------------------------

df["MACD_Signal"] = (
    df.groupby("Ticker")["MACD"]
    .transform(
        lambda x:
        x.ewm(
            span=9,
            adjust=False
        ).mean()
    )
)


# ---------------------------------------------------------
# MACD Histogram
# ---------------------------------------------------------

df["MACD_Histogram"] = (
    df["MACD"] -
    df["MACD_Signal"]
)


# =========================================================
# 6. MARKET TREND SIGNAL
# =========================================================

df["Trend_Signal"] = np.where(

    df["MA_20"] >
    df["MA_50"],

    "Bullish",

    "Bearish"
)


# =========================================================
# 7. RSI CONDITION
# =========================================================

df["RSI_Condition"] = np.select(

    [
        df["RSI_14"] < 30,

        df["RSI_14"] > 70
    ],

    [
        "Oversold",

        "Overbought"
    ],

    default="Neutral"
)


# =========================================================
# 8. VOLUME ANOMALY
# =========================================================

volume_mean = (
    df.groupby("Ticker")["Volume"]
    .transform(
        lambda x:
        x.rolling(
            window=20,
            min_periods=20
        ).mean()
    )
)


df["Volume_Ratio"] = (
    df["Volume"] /
    volume_mean
)


df["Volume_Anomaly"] = np.where(

    df["Volume_Ratio"] >= 2,

    "High Volume",

    "Normal Volume"
)


# =========================================================
# 9. DAILY PRICE MOVEMENT CATEGORY
# =========================================================

df["Market_Movement"] = np.select(

    [
        df["Daily_Return"] >= 2,

        df["Daily_Return"] <= -2
    ],

    [
        "Strong Positive",

        "Strong Negative"
    ],

    default="Normal"
)


# =========================================================
# 10. ROUND VALUES
# =========================================================

round_columns = [

    "Daily_Return",

    "Trading_Range",

    "Trading_Range_Percent",

    "MA_20",

    "MA_50",

    "Volatility_20D",

    "Volume_Change_Percent",

    "Cumulative_Return_Percent",

    "Drawdown_Percent",

    "RSI_14",

    "EMA_12",

    "EMA_26",

    "MACD",

    "MACD_Signal",

    "MACD_Histogram",

    "Volume_Ratio"
]


for column in round_columns:

    df[column] = df[column].round(
        4
    )


# =========================================================
# 11. FINAL COLUMN ORDER
# =========================================================

final_columns = [

    "Date",

    "Ticker",

    "Company_Name",

    "Open",

    "High",

    "Low",

    "Close",

    "Volume",

    "Previous_Close",

    "Daily_Return",

    "Trading_Range",

    "Trading_Range_Percent",

    "MA_20",

    "MA_50",

    "Volatility_20D",

    "Volume_Change_Percent",

    "Cumulative_Return_Percent",

    "Running_Max",

    "Drawdown_Percent",

    "RSI_14",

    "EMA_12",

    "EMA_26",

    "MACD",

    "MACD_Signal",

    "MACD_Histogram",

    "Trend_Signal",

    "RSI_Condition",

    "Volume_Ratio",

    "Volume_Anomaly",

    "Market_Movement"
]


df = df[
    final_columns
]


# =========================================================
# 12. SAVE ENGINEERED DATA
# =========================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================================================
# 13. FINAL DATA QUALITY REPORT
# =========================================================

print()
print("----------------------------------------------")
print("4. FINAL DATA QUALITY REPORT")
print("----------------------------------------------")

print()

print(
    "Final Records:",
    len(df)
)

print(
    "Companies:",
    df["Ticker"].nunique()
)

print(
    "Date Range:",
    df["Date"].min().date(),
    "to",
    df["Date"].max().date()
)

print()

print(
    "Remaining Missing Values:"
)

print(
    df.isnull().sum()
)


# =========================================================
# 14. PREVIEW
# =========================================================

print()
print("----------------------------------------------")
print("5. SAMPLE ENGINEERED DATA")
print("----------------------------------------------")

print()

print(
    df[
        [
            "Date",
            "Ticker",
            "Close",
            "Daily_Return",
            "MA_20",
            "MA_50",
            "RSI_14",
            "MACD",
            "Trend_Signal",
            "Volume_Anomaly"
        ]
    ].tail(10)
)


# =========================================================
# COMPLETE
# =========================================================

print()
print("==============================================")
print(" FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
print("==============================================")

print()

print(
    "Output File:"
)

print(
    OUTPUT_FILE
)

print()

print(
    "Created Dataset:"
)

print(
    "Stock_Prices_Features.csv"
)

print()