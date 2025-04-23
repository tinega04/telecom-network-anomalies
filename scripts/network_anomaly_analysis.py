import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------- Setup Paths --------------------
RAW_DATA_PATH = r'data/raw/ip_addresses_sample/agg_10_minutes/11.csv'
TIME_DATA_PATH = r'data/raw/times/times_10_minutes.csv'
FIGURES_DIR = 'figures'
RESULTS_DIR = 'results'

# Create folders if they don't exist
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------- Load Data --------------------
df = pd.read_csv(RAW_DATA_PATH)
time_df = pd.read_csv(TIME_DATA_PATH)

# Merge on id_time
df = df.merge(time_df, on='id_time', how='left')

# Convert 'time' to datetime with UTC
df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce')

# -------------------- Data Overview --------------------
print("Missing values:\n", df.isnull().sum())
print("\nData types:\n", df.dtypes)
print("\nSummary statistics:\n", df.describe())

# -------------------- Visualizations --------------------
plt.figure(figsize=(14, 6))
plt.plot(df['time'], df['n_flows'], label='Number of Flows')
plt.plot(df['time'], df['n_packets'], label='Number of Packets')
plt.title('Flows and Packets Over Time')
plt.xlabel('Time')
plt.ylabel('Count')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/flows_packets_over_time.png')
plt.close()

# Heatmap of correlation
plt.figure(figsize=(12, 8))
sns.heatmap(df.drop(columns=['id_time', 'time']).corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/correlation_heatmap.png')
plt.close()

# -------------------- Anomaly Detection --------------------
# Z-score based anomaly detection
def detect_anomalies(df, column, threshold=3):
    z_scores = (df[column] - df[column].mean()) / df[column].std()
    return df[np.abs(z_scores) > threshold]

anomalies_flows = detect_anomalies(df, 'n_flows')
anomalies_packets = detect_anomalies(df, 'n_packets')
anomalies_combined = pd.concat([anomalies_flows, anomalies_packets]).drop_duplicates()

print(f"Anomalies detected: {len(anomalies_combined)}")

# Save anomaly data
anomalies_combined.to_csv(f'{RESULTS_DIR}/anomalies.csv', index=False)

# Plot anomalies on flows
plt.figure(figsize=(14, 6))
plt.plot(df['time'], df['n_flows'], label='Flows', color='blue')
plt.scatter(anomalies_flows['time'], anomalies_flows['n_flows'], color='red', label='Anomalies')
plt.title('Anomalies in Network Flows')
plt.xlabel('Time')
plt.ylabel('Flow Count')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/anomalies_in_flows.png')
plt.close()

# -------------------- Export Summary --------------------
summary = {
    "total_records": len(df),
    "total_anomalies": len(anomalies_combined),
    "anomalies_flows": len(anomalies_flows),
    "anomalies_packets": len(anomalies_packets),
}
pd.Series(summary).to_csv(f'{RESULTS_DIR}/summary.csv')

print("\n✅ Analysis complete. Figures and results saved.")
