import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


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


# =========================================================
# 2. INPUT FILES
# =========================================================

MULTI_FACTOR_FILE = os.path.join(
    DATA_DIR,
    "Multi_Factor_Market_Analysis.csv"
)

AI_FILE = os.path.join(
    DATA_DIR,
    "AI_Market_Anomalies.csv"
)

NEWS_FILE = os.path.join(
    DATA_DIR,
    "News_Sentiment_Analysis.csv"
)

NEWS_SUMMARY_FILE = os.path.join(
    DATA_DIR,
    "Stock_News_Sentiment_Summary.csv"
)

STOCK_FILE = os.path.join(
    DATA_DIR,
    "Stock_Prices_Features.csv"
)


# =========================================================
# 3. OUTPUT FOLDER
# =========================================================

OUTPUT_DIR = os.path.join(
    DATA_DIR,
    "Final_Analysis"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# =========================================================
# 4. CHECK FILES
# =========================================================

required_files = [

    MULTI_FACTOR_FILE,

    AI_FILE,

    NEWS_FILE,

    NEWS_SUMMARY_FILE,

    STOCK_FILE

]


print()
print("==============================================")
print("     FINAL MARKET INTELLIGENCE ANALYSIS")
print("==============================================")
print()


for file_path in required_files:

    if not os.path.exists(file_path):

        print(
            "ERROR: File not found:"
        )

        print(
            file_path
        )

        raise SystemExit


print(
    "All required files found."
)


# =========================================================
# 5. LOAD DATA
# =========================================================

multi_factor = pd.read_csv(
    MULTI_FACTOR_FILE
)

ai_data = pd.read_csv(
    AI_FILE
)

news_data = pd.read_csv(
    NEWS_FILE
)

news_summary = pd.read_csv(
    NEWS_SUMMARY_FILE
)

stock_data = pd.read_csv(
    STOCK_FILE
)


# =========================================================
# 6. STANDARDIZE TICKER
# =========================================================

for df in [

    multi_factor,

    ai_data,

    news_data,

    news_summary,

    stock_data

]:

    if "Ticker" not in df.columns:

        possible_columns = [

            "ticker",

            "Symbol",

            "symbol"

        ]

        for column in possible_columns:

            if column in df.columns:

                df.rename(

                    columns={
                        column: "Ticker"
                    },

                    inplace=True

                )

                break


# =========================================================
# 7. QUESTION 1
# =========================================================

print()
print("================================================")
print("Q1. STRONGEST OVERALL RETURNS")
print("================================================")


return_columns = [

    "Total_Return_Percent",

    "Total_Return",

    "Return_Percent"

]


return_column = None


for column in return_columns:

    if column in multi_factor.columns:

        return_column = column

        break


if return_column:

    q1 = (

        multi_factor[
            [
                "Ticker",
                return_column
            ]
        ]

        .sort_values(
            return_column,
            ascending=False
        )

    )

    print(
        q1.to_string(
            index=False
        )
    )

    q1.to_csv(

        os.path.join(
            OUTPUT_DIR,
            "Q1_Return_Analysis.csv"
        ),

        index=False
    )

else:

    print(
        "Return column not available."
    )


# =========================================================
# 8. QUESTION 2
# =========================================================

print()
print("================================================")
print("Q2. HIGHEST VOLATILITY AND DOWNSIDE RISK")
print("================================================")


risk_columns = [

    "Annualized_Volatility_Percent",

    "Volatility_Percent",

    "Maximum_Drawdown_Percent"

]


available_risk = [

    column

    for column in risk_columns

    if column in multi_factor.columns

]


if available_risk:

    q2_columns = [
        "Ticker"
    ] + available_risk

    q2 = multi_factor[
        q2_columns
    ].copy()

    print(
        q2.to_string(
            index=False
        )
    )

    q2.to_csv(

        os.path.join(
            OUTPUT_DIR,
            "Q2_Risk_Analysis.csv"
        ),

        index=False
    )


# =========================================================
# 9. QUESTION 3
# =========================================================

print()
print("================================================")
print("Q3. BENCHMARK PERFORMANCE")
print("================================================")


benchmark_columns = [

    "Benchmark_Excess_Return",

    "Excess_Return",

    "Alpha_vs_NIFTY",

    "NIFTY_Correlation"

]


available_benchmark = [

    column

    for column in benchmark_columns

    if column in multi_factor.columns

]


if available_benchmark:

    q3 = multi_factor[
        [
            "Ticker"
        ] + available_benchmark
    ]

    print(
        q3.to_string(
            index=False
        )
    )

    q3.to_csv(

        os.path.join(
            OUTPUT_DIR,
            "Q3_Benchmark_Analysis.csv"
        ),

        index=False
    )


# =========================================================
# 10. QUESTION 4
# =========================================================

print()
print("================================================")
print("Q4. FUNDAMENTAL GROWTH + TECHNICAL MOMENTUM")
print("================================================")


q4_columns = [

    "Ticker",

    "Revenue_Growth",

    "Profit_Growth",

    "Growth_Profile",

    "Technical_Profile"

]


available_q4 = [

    column

    for column in q4_columns

    if column in multi_factor.columns

]


q4 = multi_factor[
    available_q4
].copy()


print(
    q4.to_string(
        index=False
    )
)


q4.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "Q4_Growth_Technical_Analysis.csv"
    ),

    index=False
)


# =========================================================
# 11. QUESTION 5
# =========================================================

print()
print("================================================")
print("Q5. VALUATION VS GROWTH")
print("================================================")


q5_columns = [

    "Ticker",

    "PE_Ratio",

    "Revenue_Growth",

    "Profit_Growth",

    "Valuation_Profile",

    "Growth_Profile"

]


available_q5 = [

    column

    for column in q5_columns

    if column in multi_factor.columns

]


q5 = multi_factor[
    available_q5
].copy()


print(
    q5.to_string(
        index=False
    )
)


q5.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "Q5_Valuation_Growth_Analysis.csv"
    ),

    index=False
)


# =========================================================
# 12. QUESTION 6
# =========================================================

print()
print("================================================")
print("Q6. AI-DETECTED MARKET ANOMALIES")
print("================================================")


q6 = (

    ai_data

    .groupby(
        "Ticker"
    )

    .agg(

        AI_Anomaly_Count=(
            "AI_Anomaly_Flag",
            "sum"
        ),

        Average_AI_Score=(
            "AI_Anomaly_Score",
            "mean"
        )

    )

    .reset_index()

)


q6 = q6.sort_values(
    "AI_Anomaly_Count",
    ascending=False
)


print(
    q6.to_string(
        index=False
    )
)


q6.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "Q6_AI_Anomaly_Analysis.csv"
    ),

    index=False
)


# =========================================================
# 13. QUESTION 7
# =========================================================

print()
print("================================================")
print("Q7. MOST FREQUENT AI ANOMALY TYPES")
print("================================================")


q7 = (

    ai_data[
        ai_data[
            "AI_Anomaly_Flag"
        ] == 1
    ]

    .groupby(
        "AI_Anomaly_Type"
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
    q7.to_string(
        index=False
    )
)


q7.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "Q7_Anomaly_Type_Analysis.csv"
    ),

    index=False
)


# =========================================================
# 14. QUESTION 8
# =========================================================

print()
print("================================================")
print("Q8. NEWS SENTIMENT VS MARKET MOVEMENT")
print("================================================")


news_by_stock = (

    news_summary[
        [
            "Ticker",
            "Average_Sentiment",
            "Positive_News_Percent",
            "Negative_News_Percent",
            "Overall_Sentiment"
        ]
    ]

)


# Detect return column

stock_return_column = None


for column in return_columns:

    if column in multi_factor.columns:

        stock_return_column = column

        break


if stock_return_column:

    return_data = multi_factor[
        [
            "Ticker",
            stock_return_column
        ]
    ]


    q8 = pd.merge(

        news_by_stock,

        return_data,

        on="Ticker",

        how="left"

    )


else:

    q8 = news_by_stock.copy()


print(
    q8.to_string(
        index=False
    )
)


q8.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "Q8_News_vs_Market_Analysis.csv"
    ),

    index=False
)


# =========================================================
# 15. QUESTION 9
# =========================================================

print()
print("================================================")
print("Q9. NEGATIVE SENTIMENT + HIGH RISK + AI ACTIVITY")
print("================================================")


q9_columns = [

    "Ticker",

    "Overall_Sentiment",

    "Negative_News_Percent",

    "Annualized_Volatility_Percent",

    "AI_Anomaly_Count",

    "AI_High_Risk_Events"

]


available_q9 = [

    column

    for column in q9_columns

    if column in multi_factor.columns
]


# Add news information

q9 = multi_factor[
    [
        "Ticker"
    ]

    +

    [
        column

        for column in q9_columns

        if column != "Ticker"
        and column in multi_factor.columns
    ]

].copy()


q9 = pd.merge(

    q9,

    news_summary[
        [
            "Ticker",
            "Overall_Sentiment",
            "Negative_News_Percent"
        ]
    ],

    on="Ticker",

    how="left",

    suffixes=(
        "",
        "_News"
    )

)


print(
    q9.to_string(
        index=False
    )
)


q9.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "Q9_Negative_Sentiment_Risk_AI.csv"
    ),

    index=False
)


# =========================================================
# 16. QUESTION 10
# =========================================================

print()
print("================================================")
print("Q10. COMPLETE MULTI-FACTOR MARKET PROFILE")
print("================================================")


profile_columns = [

    "Ticker",

    "Growth_Profile",

    "Profitability_Profile",

    "Leverage_Profile",

    "Technical_Profile",

    "Valuation_Profile",

    "Risk_Profile",

    "AI_Profile"

]


available_profile = [

    column

    for column in profile_columns

    if column in multi_factor.columns
]


q10 = multi_factor[
    available_profile
].copy()


q10 = pd.merge(

    q10,

    news_summary[
        [
            "Ticker",
            "Overall_Sentiment",
            "Average_Sentiment"
        ]
    ],

    on="Ticker",

    how="left"

)


print(
    q10.to_string(
        index=False
    )
)


q10.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "Q10_Multi_Factor_Profile.csv"
    ),

    index=False
)


# =========================================================
# 17. SENTIMENT DISTRIBUTION
# =========================================================

print()
print("================================================")
print("NEWS SENTIMENT DISTRIBUTION")
print("================================================")


sentiment_distribution = (

    news_data[
        "Sentiment"
    ]

    .value_counts()

    .reset_index()

)


sentiment_distribution.columns = [

    "Sentiment",

    "News_Count"

]


print(
    sentiment_distribution.to_string(
        index=False
    )
)


sentiment_distribution.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "Sentiment_Distribution.csv"
    ),

    index=False
)


# =========================================================
# 18. AI ANOMALY DISTRIBUTION
# =========================================================

print()
print("================================================")
print("AI ANOMALY DISTRIBUTION")
print("================================================")


ai_distribution = (

    ai_data[
        ai_data[
            "AI_Anomaly_Flag"
        ] == 1
    ]

    [
        "AI_Anomaly_Type"
    ]

    .value_counts()

    .reset_index()

)


ai_distribution.columns = [

    "AI_Anomaly_Type",

    "Anomaly_Count"

]


print(
    ai_distribution.to_string(
        index=False
    )
)


# =========================================================
# 19. FINAL SUMMARY
# =========================================================

print()
print("==============================================")
print("     STEP 7 COMPLETED SUCCESSFULLY")
print("==============================================")
print()

print(
    "10 integrated market questions analyzed."
)

print()

print(
    "Output folder:"
)

print(
    OUTPUT_DIR
)

print()

print(
    "Generated analytical files:"
)

for file_name in os.listdir(
    OUTPUT_DIR
):

    print(
        "-",
        file_name
    )

print()