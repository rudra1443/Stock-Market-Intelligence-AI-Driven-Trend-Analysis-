SELECT * FROM company_fundamentals cf ,
            multi_factor_market_analysis mfma ,
            stock_benchmark_analysis sba ,
            stock_news_sentiment_summary snss ,
            stock_prices_features spf ,
            stock_risk_summary srs ;


-- Question_01 How many companies and trading records are available in the dataset?

SELECT 
    COUNT(DISTINCT Ticker) AS Total_Companies,
    COUNT(*) AS total_Trading_Records 
    FROM stock_prices_features spf ;

-- Question_02 Which stocks generated the highest cumulative return during the analysis period?

SELECT 
   Ticker,
   ROUND(
   (EXP(SUM(LN(1 + Daily_Return / 100))) - 1)
   * 100,
       2
   ) AS Cumulative_Return_Percentage
   FROM stock_prices_features spf 
   WHERE Daily_Return > -100 
   GROUP BY spf.Ticker 
   ORDER BY Cumulative_Return_Percentage ;

-- Question_03 Which stocks experienced the highest average 20-day volatility?

SELECT 
  Ticker,
  ROUND(AVG(Volatility_20D),2) AS 
  Avrage_20D_Volatility 
  FROM stock_prices_features spf 
  GROUP BY spf.Ticker 
  ORDER BY Avrage_20D_Volatility DESC ;

-- Question_04 :- Which stocks experienced the largest maximum drawdown?

SELECT 
   Ticker,
   ROUND(MAX(spf.Drawdown_Percent),2) AS Maximum_Drawdown_Percent
   FROM stock_prices_features spf 
   GROUP BY spf.Ticker 
   ORDER BY Maximum_Drawdown_Percent ASC;

-- Question_05 :- How do company growth metrics compare with their valuation?

SELECT 
   cf.Ticker ,
   cf.Company_Name ,
   cf.Revenue_Growth ,
   cf.Profit_Margin ,
   cf.PE_Ratio ,
   cf.ROE  AS Profit_Growth
   FROM company_fundamentals cf  
   ORDER BY Profit_Growth ;

-- Question_06:- Which stocks have the highest number of high-risk AI-detected market events?

SELECT
    Ticker,
    COUNT(*) AS High_Risk_AI_Events
FROM ai_market_anomalies
WHERE AI_Risk_Level = 'High'
GROUP BY Ticker
ORDER BY High_Risk_AI_Events DESC;

-- Question_07:- Which stocks generated the highest number of AI_detected abnormal market events?

SELECT 
   Ticker,
   COUNT(*) AS AI_Anomaly_Count
   FROM ai_market_anomalies ama 
   WHERE ama.AI_Anomaly_Flag = 1
   GROUP BY ama.Ticker 
   ORDER BY  AI_Anomaly_Count DESC ;

-- Question_08:- What types of abnormal market event occured most frequently?

SELECT 
  AI_Anomaly_type,
  COUNT(*) AS Anomay_Count
  FROM ai_market_anomalies ama 
  WHERE ama.AI_Anomaly_Flag = 1
  GROUP BY ai_anomaly_type 
  ORDER BY anomay_count ;

-- Question_09 :- Which Stocks had the highest negative financial-news exposure?

SELECT 
  Ticker,
  News_Count,
  Average_Sentiment,
  Positive_News,
  Neutral_News,
  Negative_News,
  Negative_News_Percent,
  Overall_Sentiment
  FROM stock_news_sentiment_summary snss 
  ORDER BY negative_news_percent DESC ;

-- Question_10:- What is the multi-dimensional market profiles of each stock across fundamentals, technical indicators,
    -- valuation, risk and AI activity?

SELECT 
   Ticker,
   Growth_Profile,
   mfma.Profitability_Profile ,
   Leverage_Profile,
   Technical_Profile
   Valuation_Profile,
   Risk_Profile,
   AI_Profile
   FROM multi_factor_market_analysis mfma 
   ORDER BY Ticker;














   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
























