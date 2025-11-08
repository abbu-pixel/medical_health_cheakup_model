import pandas as pd
import numpy as np
from faker import Faker
import random
import os

fake = Faker()

def generate_medical_data(num_records: int, output_file: str):
    np.random.seed(42)
    records = []

    for _ in range(num_records):
        age = np.random.randint(18, 90)
        gender = random.choice(['Male', 'Female'])
        heart_rate = np.random.randint(50, 120)
        temperature = round(np.random.uniform(36.0, 40.0), 1)
        blood_pressure = f"{np.random.randint(90, 160)}/{np.random.randint(60, 100)}"
        oxygen_level = np.random.randint(85, 100)
        glucose_level = np.random.randint(70, 200)
        cholesterol = np.random.randint(100, 300)
        device_id = fake.uuid4()
        checkup_date = fake.date_between(start_date='-2y', end_date='today')

        # health status label (1 = normal, 0 = needs attention)
        status = 1 if (heart_rate < 100 and 36 <= temperature <= 38 and oxygen_level > 92 and glucose_level < 140) else 0

        records.append([
            age, gender, heart_rate, temperature, blood_pressure,
            oxygen_level, glucose_level, cholesterol, device_id,
            checkup_date, status
        ])

    df = pd.DataFrame(records, columns=[
        'age', 'gender', 'heart_rate', 'temperature', 'blood_pressure',
        'oxygen_level', 'glucose_level', 'cholesterol',
        'device_id', 'checkup_date', 'status'
    ])

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"✅ Generated {len(df)} records and saved to {output_file}")

if __name__ == "__main__":
    # You can adjust this number for MBs or GBs of data
    num_records = 1_000_000  # ~200MB (increase for GBs)
    generate_medical_data(num_records, "data/medical_checkup_data.csv")
