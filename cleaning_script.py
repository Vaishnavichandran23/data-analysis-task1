import pandas as pd

# Load dataset
df = pd.read_csv("Customer_Data.csv")

# Clean column names (remove spaces)
df.columns = df.columns.str.strip()

# Check missing values before cleaning
print("Missing values before cleaning:\n", df.isnull().sum())

# Handle missing values (fill with mean)
df['minimum_payments'] = df['minimum_payments'].fillna(df['minimum_payments'].mean())
df['credit_limit'] = df['credit_limit'].fillna(df['credit_limit'].mean())

# Remove duplicate rows
df = df.drop_duplicates()

# Handle outliers (remove top 1% extreme values in balance)
df = df[df['balance'] < df['balance'].quantile(0.99)]

# Feature Engineering / Data Transformation

# Average purchase per transaction
df['avg_purchase'] = df['purchases'] / df['purchases_trx']

# Replace infinite values (division by zero)
df['avg_purchase'] = df['avg_purchase'].replace([float('inf'), -float('inf')], 0)

# Total transactions
df['total_transactions'] = df['purchases_trx'] + df['cash_advance_trx']

# Balance categories
df['balance_group'] = pd.cut(
    df['balance'],
    bins=[0, 1000, 5000, 10000, 50000],
    labels=['Low', 'Medium', 'High', 'Very High']
)

# Normalize balance (0 to 1 scale)
df['normalized_balance'] = (
    (df['balance'] - df['balance'].min()) /
    (df['balance'].max() - df['balance'].min())
)

# Final check
print("\nMissing values after cleaning:\n", df.isnull().sum())

# Save cleaned dataset
df.to_csv("cleaned_data.csv", index=False)

print("\nData Cleaning and Transformation Completed Successfully!")
