import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

RAW_DATA_DIR = r"F:\data\raw"
PROCESSED_DATA_PATH = r"F:\data\processed\final_dataset.csv"

def merge_datasets():
    all_data = []
    for file_name in os.listdir(RAW_DATA_DIR):
        if file_name.endswith(".csv"):
            file_path = os.path.join(RAW_DATA_DIR, file_name)
            df = pd.read_csv(file_path)
            all_data.append(df)
    merged_df = pd.concat(all_data, ignore_index=True)
    merged_df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"✅ Merged dataset saved to {PROCESSED_DATA_PATH}")
    return merged_df

def preprocess_data():
    df = pd.read_csv(PROCESSED_DATA_PATH)

    # Drop duplicates and rows with all NaNs
    df.drop_duplicates(inplace=True)
    df.dropna(how="all", inplace=True)

    # Example: drop irrelevant columns if they exist
    drop_cols = ["Patient ID", "Timestamp", "IP Address"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # Separate target variable if it exists
    target_col = "Target" if "Target" in df.columns else None
    X = df.drop(columns=[target_col]) if target_col else df
    y = df[target_col] if target_col else None

    # Identify numeric and categorical columns
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    # Define preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ]
    )

    X_processed = preprocessor.fit_transform(X)

    print("✅ Preprocessing completed successfully!")
    return X_processed, y

if __name__ == "__main__":
    merge_datasets()
    preprocess_data()
