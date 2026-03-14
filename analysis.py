import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('charts', exist_ok=True)

sentiment_df = pd.read_csv('sentiment.csv')
trader_df = pd.read_csv('trader_data.csv')

print("--- PART A: Data Preparation ---")
print("Sentiment Shape:", sentiment_df.shape)
print("Trader Data Shape:", trader_df.shape)

print("\nMissing Values (Sentiment):\n", sentiment_df.isnull().sum())
print("\nMissing Values (Trader):\n", trader_df.isnull().sum())

print("\nDuplicates (Sentiment):", sentiment_df.duplicated().sum())
print("Duplicates (Trader):", trader_df.duplicated().sum())

sentiment_df['date'] = pd.to_datetime(sentiment_df['date'], format='mixed').dt.date
trader_df['date'] = pd.to_datetime(trader_df['Timestamp'], unit='ms').dt.date

daily_sentiment = sentiment_df.groupby('date').agg({'value': 'mean', 'classification': 'last'}).reset_index()

merged_df = pd.merge(trader_df, daily_sentiment, on='date', how='inner')

daily_trader_metrics = merged_df.groupby(['date', 'Account']).agg(
    daily_pnl=('Closed PnL', 'sum'),
    num_trades=('Trade ID', 'count'),
    total_volume=('Size USD', 'sum'),
    win_trades=('Closed PnL', lambda x: (x > 0).sum()),
    sentiment_val=('value', 'first'),
    sentiment_class=('classification', 'first')
).reset_index()

daily_trader_metrics['win_rate'] = (daily_trader_metrics['win_trades'] / daily_trader_metrics['num_trades']).fillna(0)
daily_trader_metrics['avg_trade_size'] = daily_trader_metrics['total_volume'] / daily_trader_metrics['num_trades']

long_short = merged_df.groupby('date')['Side'].value_counts().unstack().fillna(0)
buy_col = long_short['BUY'] if 'BUY' in long_short.columns else 0
sell_col = long_short['SELL'] if 'SELL' in long_short.columns else 1
long_short['ls_ratio'] = buy_col / np.maximum(sell_col, 1)

daily_metrics_summary = daily_trader_metrics.groupby('date').agg(
    avg_daily_pnl=('daily_pnl', 'mean'),
    avg_win_rate=('win_rate', 'mean'),
    avg_trade_size=('avg_trade_size', 'mean'),
    avg_num_trades=('num_trades', 'mean'),
    sentiment_class=('sentiment_class', 'first')
).join(long_short[['ls_ratio']])

print("\n--- Key Metrics Head ---")
print(daily_metrics_summary.head())

print("\n--- PART B: Analysis & Insights ---")

perf_by_sentiment = daily_trader_metrics.groupby('sentiment_class').agg(
    avg_pnl=('daily_pnl', 'mean'),
    median_pnl=('daily_pnl', 'median'),
    avg_win_rate=('win_rate', 'mean'),
    avg_num_trades=('num_trades', 'mean'),
    avg_trade_size=('avg_trade_size', 'mean')
)

print("\nPerformance by Sentiment:")
print(perf_by_sentiment)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

perf_by_sentiment[['avg_pnl']].plot(kind='bar', ax=axes[0, 0], title='Average PnL by Sentiment', color=['skyblue'])
axes[0, 0].set_ylabel('Avg PnL')

perf_by_sentiment[['avg_win_rate']].plot(kind='bar', ax=axes[0, 1], title='Average Win Rate by Sentiment', color=['lightgreen'])
axes[0, 1].set_ylabel('Win Rate')

perf_by_sentiment[['avg_num_trades']].plot(kind='bar', ax=axes[1, 0], title='Avg Num Trades by Sentiment', color=['salmon'])
axes[1, 0].set_ylabel('Num Trades')

perf_by_sentiment[['avg_trade_size']].plot(kind='bar', ax=axes[1, 1], title='Avg Trade Size by Sentiment', color=['gold'])
axes[1, 1].set_ylabel('Trade Size (USD)')

plt.tight_layout()
plt.savefig('charts/sentiment_behavior.png')
plt.close()

daily_trader_metrics['activity_segment'] = pd.qcut(daily_trader_metrics['num_trades'], q=2, labels=['Infrequent', 'Frequent'])
daily_trader_metrics['size_segment'] = pd.qcut(daily_trader_metrics['avg_trade_size'], q=2, labels=['Small', 'Large'])

segment_perf = daily_trader_metrics.groupby(['activity_segment', 'sentiment_class']).agg(
    avg_pnl=('daily_pnl', 'mean')
).unstack()
print("\nSegment Performance (Infrequent vs Frequent):")
print(segment_perf)

segment_perf.plot(kind='bar', figsize=(10, 6), title='PnL by Activity Segment and Sentiment')
plt.ylabel('Avg PnL')
plt.tight_layout()
plt.savefig('charts/segment_analysis.png')
plt.close()

print("\n--- PART C: Actionable Output ---")
print("1. Strategy Idea: Adjust sizing based on market regime. During 'Extreme Fear', win rates and PnL might slightly dip or be more volatile; sizing down could preserve capital.")
print("2. Strategy Idea: Frequency tuning. Frequent traders might see diminished returns during specific sentiment regimes compared to infrequent traders. Align trade frequency with high-conviction setups only during highly emotional market periods.")
