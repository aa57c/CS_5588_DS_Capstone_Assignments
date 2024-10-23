import pandas as pd
import numpy as np
import random

# Parameters
num_patients = 1  # Number of unique patients
num_entries_per_patient = 100  # Number of CGM entries per patient
time_interval = 3  # Time interval between measurements in seconds
start_time = pd.Timestamp('00:00:00')  # Start time

# Function to generate random CGM data
def generate_cgm_data(num_patients, num_entries_per_patient, start_time, time_interval):
    data = []
    
    for patient_id in range(1, num_patients + 1):
        # Generate T2DM value once for the patient
        t2dm = random.choice([True, False])  # Random True/False for type 2 diabetes
        
        current_time = start_time
        for i in range(num_entries_per_patient):
            glucemia = np.random.randint(70, 180)  # Random glucose value between 70 and 180 mg/dL
            age = np.random.randint(20, 80)  # Random age between 20 and 80
            bmi = round(np.random.uniform(18, 35), 1)  # Random BMI between 18 and 35
            glycaemia = np.random.randint(90, 130)  # Random glycaemia value
            hba1c = round(np.random.uniform(4.5, 8.0), 1)  # Random HbA1c between 4.5 and 8.0
            follow_up = np.random.randint(100, 500)  # Random follow-up time
            
            # Append the row to data with fixed T2DM value
            data.append([patient_id, current_time.time(), glucemia, 1, age, bmi, glycaemia, hba1c, follow_up, t2dm])
            
            # Increment the time for the next entry
            current_time += pd.Timedelta(seconds=time_interval)
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=['Patient_ID', 'Hora', 'Glucemia', 'gender', 'age', 'BMI', 'glycaemia', 'HbA1c', 'follow.up', 'T2DM'])
    return df

# Generate data
synthetic_cgm_data = generate_cgm_data(num_patients, num_entries_per_patient, start_time, time_interval)

# Show the first few rows of the generated data
print(synthetic_cgm_data.head())

# Save to CSV
synthetic_cgm_data.to_csv('synthetic_cgm_data.csv', index=False)
