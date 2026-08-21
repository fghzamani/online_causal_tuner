import pandas as pd

data_path = "/home/forough/phd_projects/online_tuner/rct_data_campaign_a_pal_office/rct_results.csv"
df = pd.read_csv(data_path)

print("Total Rows:", len(df))

for col in ["param__local_costmap__footprint", "arm_requested_label", "param_actual__local_costmap__footprint"]:
    if col in df.columns:
        print(f"\n--- {col} ---")
        print(df[col].value_counts().head(5))

print("\n--- risk__r_width ---")
if "risk__r_width" in df.columns:
    print(df["risk__r_width"].describe())

print("\n--- risk__r_min ---")
if "risk__r_min" in df.columns:
    print(df["risk__r_min"].describe())
