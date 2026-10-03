import pandas as pd
import numpy as np
import os

from sklearn.ensemble import IsolationForest


# =========================================================
# 1. PROJECT PATH
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

INPUT_FILE = os.path.join(
    DATA_DIR,
    "Stock_Prices_Features.csv"
)

OUTPUT_FILE = os.path.join(
    DATA_DIR,
    "AI_Market_Anomalies.csv"
)


# =========================================================
# 2. CHECK INPUT FILE
# =========================================================

print()
print("==============================================")
print("       AI MARKET INTELLIGENCE")
print("==============================================")
print()

if not os.path.exists(INPUT_FILE):

    print("ERROR: Stock_Prices_Features.csv not found.")
    print()
    print("Expected location:")
    print(INPUT_FILE)

    raise SystemExit


print("Input file found successfully.")


# =========================================================
# 3. LOAD DATA
# =========================================================

df = pd.read_csv(
    INPUT_FILE
)

print(
    "Total records loaded:",
    len(df)
)


# =========================================================
# 4. CHECK REQUIRED COLUMNS
# =========================================================

required_columns = [

    "Date",
    "Ticker",
    "Close",
    "Volume",
    "Daily_Return",
    "Trading_Range_Percent",
    "Volatility_20D",
    "Volume_Change_Percent",
    "RSI_14",
    "MACD",
    "Drawdown_Percent"
]


missing_columns = [

    column

    for column in required_columns

    if column not in df.columns
]


if missing_columns:

    print()
    print("ERROR: Required columns are missing:")

    for column in missing_columns:

        print("-", column)

    raise SystemExit


print("Required columns verified.")


# =========================================================
# 5. DATE CONVERSION
# =========================================================

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)


# =========================================================
# 6. SORT DATA
# =========================================================

df = df.sort_values(
    [
        "Ticker",
        "Date"
    ]
).reset_index(
    drop=True
)


# =========================================================
# 7. AI FEATURES
# =========================================================

ai_features = [

    "Daily_Return",

    "Trading_Range_Percent",

    "Volatility_20D",

    "Volume_Change_Percent",

    "RSI_14",

    "MACD",

    "Drawdown_Percent"
]


print()
print("----------------------------------------------")
print("AI FEATURES")
print("----------------------------------------------")

for feature in ai_features:

    print(
        "-",
        feature
    )


# =========================================================
# 8. PREPARE AI DATA
# =========================================================

model_data = df[
    [
        "Date",
        "Ticker",
        "Close",
        "Volume"
    ]
    + ai_features
].copy()


# =========================================================
# 9. CONVERT AI FEATURES TO NUMERIC
# =========================================================

for column in ai_features:

    model_data[column] = pd.to_numeric(
        model_data[column],
        errors="coerce"
    )


# =========================================================
# 10. REMOVE INFINITY VALUES
# =========================================================

model_data[ai_features] = (
    model_data[ai_features]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
)


# =========================================================
# 11. REMOVE INVALID ROWS
# =========================================================

before_cleaning = len(
    model_data
)


model_data = model_data.dropna(
    subset=ai_features
).reset_index(
    drop=True
)


after_cleaning = len(
    model_data
)


print()
print("----------------------------------------------")
print("AI DATA QUALITY CHECK")
print("----------------------------------------------")

print(
    "Records before cleaning:",
    before_cleaning
)

print(
    "Records after cleaning:",
    after_cleaning
)

print(
    "Invalid records removed:",
    before_cleaning - after_cleaning
)


# =========================================================
# 12. FINAL NUMERIC VALIDATION
# =========================================================

feature_values = model_data[
    ai_features
].to_numpy(
    dtype=float
)


if not np.isfinite(
    feature_values
).all():

    print()
    print(
        "ERROR: Invalid NaN or Infinity values remain."
    )

    raise SystemExit


print(
    "AI feature validation: PASSED"
)


# =========================================================
# 13. ISOLATION FOREST
# =========================================================

print()
print("----------------------------------------------")
print("RUNNING ISOLATION FOREST")
print("----------------------------------------------")


results = []


for ticker, group in model_data.groupby(
    "Ticker"
):

    group = group.copy()

    print(
        "Processing:",
        ticker,
        "| Records:",
        len(group)
    )


    # -----------------------------------------------------
    # Minimum records check
    # -----------------------------------------------------

    if len(group) < 30:

        print(
            "Skipped:",
            ticker,
            "- insufficient records"
        )

        group[
            "AI_Anomaly_Flag"
        ] = 0

        group[
            "AI_Anomaly_Score"
        ] = np.nan

        results.append(
            group
        )

        continue


    # -----------------------------------------------------
    # Select AI features
    # -----------------------------------------------------

    X = group[
        ai_features
    ].copy()


    # -----------------------------------------------------
    # Final safety cleaning
    # -----------------------------------------------------

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )


    valid_rows = X.notna().all(
        axis=1
    )


    group = group.loc[
        valid_rows
    ].copy()


    X = X.loc[
        valid_rows
    ]


    # -----------------------------------------------------
    # Isolation Forest Model
    # -----------------------------------------------------

    model = IsolationForest(

        n_estimators=200,

        contamination=0.02,

        random_state=42,

        n_jobs=-1

    )


    # -----------------------------------------------------
    # Train Model
    # -----------------------------------------------------

    predictions = model.fit_predict(
        X
    )


    # -----------------------------------------------------
    # Anomaly Score
    # -----------------------------------------------------

    anomaly_scores = (
        model.decision_function(
            X
        )
    )


    # -----------------------------------------------------
    # Store Results
    # -----------------------------------------------------

    group[
        "AI_Anomaly_Flag"
    ] = np.where(

        predictions == -1,

        1,

        0
    )


    group[
        "AI_Anomaly_Score"
    ] = anomaly_scores


    results.append(
        group
    )


# =========================================================
# 14. COMBINE RESULTS
# =========================================================

if not results:

    print()
    print(
        "ERROR: No valid data available for AI analysis."
    )

    raise SystemExit


ai_results = pd.concat(
    results,
    ignore_index=True
)


# =========================================================
# 15. CLASSIFY ANOMALY TYPE
# =========================================================

def classify_anomaly(row):

    if row[
        "AI_Anomaly_Flag"
    ] == 0:

        return "Normal"


    # Extreme price movement
    if abs(
        row["Daily_Return"]
    ) >= 4:

        return "Extreme Price Movement"


    # High volume activity
    if row[
        "Volume_Change_Percent"
    ] >= 100:

        return "Volume Spike"


    # High volatility
    if row[
        "Volatility_20D"
    ] >= 3:

        return "High Volatility"


    # Extreme RSI
    if (
        row["RSI_14"] <= 20
        or
        row["RSI_14"] >= 80
    ):

        return "Extreme RSI"


    return "Other Anomaly"


ai_results[
    "AI_Anomaly_Type"
] = ai_results.apply(
    classify_anomaly,
    axis=1
)


# =========================================================
# 16. MARKET EVENT CLASSIFICATION
# =========================================================

ai_results[
    "AI_Market_Event"
] = np.select(

    [

        ai_results[
            "Daily_Return"
        ] >= 3,

        ai_results[
            "Daily_Return"
        ] <= -3,

        ai_results[
            "Volume_Change_Percent"
        ] >= 100

    ],

    [

        "Strong Positive Movement",

        "Strong Negative Movement",

        "Unusual Volume Activity"

    ],

    default="Normal Movement"
)


# =========================================================
# 17. AI RISK LEVEL
# =========================================================

ai_results[
    "AI_Risk_Level"
] = np.select(

    [

        (
            ai_results[
                "AI_Anomaly_Flag"
            ] == 1
        )
        &
        (
            abs(
                ai_results[
                    "Daily_Return"
                ]
            ) >= 5
        ),

        (
            ai_results[
                "AI_Anomaly_Flag"
            ] == 1
        ),

    ],

    [

        "High",

        "Medium"

    ],

    default="Low"
)


# =========================================================
# 18. ROUND AI SCORE
# =========================================================

ai_results[
    "AI_Anomaly_Score"
] = ai_results[
    "AI_Anomaly_Score"
].round(4)


# =========================================================
# 19. SAVE AI DATASET
# =========================================================

ai_results.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================================================
# 20. AI SUMMARY
# =========================================================

total_records = len(
    ai_results
)


total_anomalies = int(
    ai_results[
        "AI_Anomaly_Flag"
    ].sum()
)


if total_records > 0:

    anomaly_percentage = (

        total_anomalies /
        total_records

    ) * 100

else:

    anomaly_percentage = 0


print()
print("----------------------------------------------")
print("AI ANOMALY SUMMARY")
print("----------------------------------------------")

print()

print(
    "Total AI Records:",
    total_records
)

print(
    "AI Anomalies:",
    total_anomalies
)

print(
    "Anomaly Percentage:",
    round(
        anomaly_percentage,
        2
    ),
    "%"
)


# =========================================================
# 21. ANOMALIES BY STOCK
# =========================================================

print()
print("----------------------------------------------")
print("ANOMALIES BY STOCK")
print("----------------------------------------------")


stock_anomalies = (

    ai_results[
        ai_results[
            "AI_Anomaly_Flag"
        ] == 1
    ]

    .groupby(
        "Ticker"
    )

    .size()

    .reset_index(
        name="Anomaly_Count"
    )

    .sort_values(
        "Anomaly_Count",
        ascending=False
    )
)


print(
    stock_anomalies
)


# =========================================================
# 22. ANOMALY TYPES
# =========================================================

print()
print("----------------------------------------------")
print("ANOMALY TYPES")
print("----------------------------------------------")


anomaly_types = (

    ai_results[
        ai_results[
            "AI_Anomaly_Flag"
        ] == 1
    ]

    [
        "AI_Anomaly_Type"
    ]

    .value_counts()
)


print(
    anomaly_types
)


# =========================================================
# 23. TOP AI-DETECTED EVENTS
# =========================================================

print()
print("----------------------------------------------")
print("TOP AI-DETECTED MARKET EVENTS")
print("----------------------------------------------")


top_anomalies = (

    ai_results[
        ai_results[
            "AI_Anomaly_Flag"
        ] == 1
    ]

    .sort_values(
        "AI_Anomaly_Score",
        ascending=True
    )

    [
        [
            "Date",
            "Ticker",
            "Close",
            "Daily_Return",
            "Volume_Change_Percent",
            "RSI_14",
            "AI_Anomaly_Type",
            "AI_Risk_Level",
            "AI_Anomaly_Score"
        ]
    ]

    .head(20)
)


print(
    top_anomalies.to_string(
        index=False
    )
)


# =========================================================
# 24. FINAL DATA QUALITY CHECK
# =========================================================

print()
print("----------------------------------------------")
print("FINAL OUTPUT CHECK")
print("----------------------------------------------")


print(
    "Output Records:",
    len(ai_results)
)


print(
    "Output Columns:",
    len(ai_results.columns)
)


print(
    "Output File:"
)

print(
    OUTPUT_FILE
)


# =========================================================
# 25. COMPLETION
# =========================================================

print()
print("==============================================")
print("      AI ANALYSIS COMPLETED SUCCESSFULLY")
print("==============================================")

print()

print(
    "Created Dataset:"
)

print(
    "AI_Market_Anomalies.csv"
)

print()

print(
    "AI workflow completed:"
)

print(
    "Technical Features"
)

print(
    "        ↓"
)

print(
    "Data Validation"
)

print(
    "        ↓"
)

print(
    "Isolation Forest"
)

print(
    "        ↓"
)

print(
    "Anomaly Detection"
)

print(
    "        ↓"
)

print(
    "Market Event Classification"
)

print(
    "        ↓"
)

print(
    "AI Risk Level"
)

print()