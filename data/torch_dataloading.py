'''Adapted from the original codebase:
J. Yoon, D. Jarrett, M. van der Schaar, "Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.
'''
## Necessary Packages
import numpy as np
import os
import pandas as pd
import dask.dataframe as dd
from sklearn.preprocessing import LabelEncoder


def MinMaxScaler(data):
    """Apply Min-Max normalization to the given data.
    Args:
      - data (np.ndarray): raw data
    Returns:
      - norm_data (np.ndarray): normalized data
      - min_val (np.ndarray): minimum values (for renormalization)
      - max_val (np.ndarray): maximum values (for renormalization)
    """    
    min_val = np.min(np.min(data, axis = 0), axis = 0)
    data = data - min_val
      
    max_val = np.max(np.max(data, axis = 0), axis = 0)
    norm_data = data / (max_val + 1e-7)
      
    return norm_data, min_val, max_val


def sine_data_generation (no, seq_len, dim):
    """Sine data generation.

    Args:
    - no: the number of samples
    - seq_len: sequence length of the time-series
    - dim: feature dimensions

    Returns:
    - data: generated data
    """  
    # Initialize the output
    data = list()

    # Generate sine data
    for i in range(no):      
    # Initialize each time-series
        temp = list()
        # For each feature
        for k in range(dim):
            # Randomly drawn frequency and phase
            freq = np.random.uniform(0, 0.1)            
            phase = np.random.uniform(0, 0.1)
                
            # Generate sine signal based on the drawn frequency and phase
            temp_data = [np.sin(freq * j + phase) for j in range(seq_len)] 
            temp.append(temp_data)
        
    # Align row/column
    temp = np.transpose(np.asarray(temp))        
    # Normalize to [0,1]
    temp = (temp + 1)*0.5
    # Stack the generated data
    data.append(temp)
                
    return data
    
def disease_categories(categories, types):
    group = ""
    if set(types).issubset(set(categories)):
        group = f"{types[0]}/{types[1]}"
    elif types[0] in categories:
        group = types[0]
    elif types[1] in categories:
        group = types[1]
    else:
        group = "no disease"
    return group

def real_data_loading (data_name, seq_len=None, resample_rate=None, save=False, norm=False):
    """Load and preprocess real-world datasets. It assumes that the data is in a
    separate directory outside of the repository, particularly since eICU has restricted
    access

    Args:
    - data_name: eICU or CKD
    - seq_len: sequence length
    - resample_rate: resample spacing as per Pandas format - for eICU only

    Returns:
    - data: preprocessed data.
    - labels: labels for downstream usage
    """  
    assert data_name in ['ckd', "eicu"]
    dir = os.path.dirname(__file__)
    data_dir = os.path.join(dir, "..", "..", "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)


    if data_name == "ckd":
        # Make file paths

        saved_npy_file_path = os.path.join(data_dir, "ckd_sequences.npy")
        saved_label_npy_file_path = os.path.join(data_dir, "ckd_labels_hypertension.npy")
        data_path = os.path.join(dir, 'timeseries2_CKD_dataset.csv')

        #Loading and filtering the Japanese CKD dataset
        print("CKD dataframe loading")
        df_ckd = pd.read_csv(data_path, delimiter=",")
        columns_used = ['eGFR', 'age', 'BMI', 'Hb', 'Alb', 'Cr', 'UPCR', "ID"]
        df_labels = df_ckd.loc[:, ["ID", "hypertension"]]
        df_ckd = df_ckd[columns_used].dropna(axis=0)

        #Convert to Numpy array and reshape into time series sub-arrays
        ckd_array = df_ckd.values
        ckd_sequences_array = ckd_array.reshape(-1, 7, len(columns_used))

        #Identify & remove sub-arrays where any value in the final position (eGFR) is zero
        #(currently only considering full sequences)
        mask = np.all(ckd_sequences_array[:, :, 0] != 0, axis=1)            #middle index is sequence number? last index 0 for egfr?
        ori_data = ckd_sequences_array[mask]

        #Removing patient ID from NumPy sequences
        patients = ori_data[:, 0, -1].astype(np.int32)
        ori_data = np.delete(ori_data, 7, axis=2)

        #Picking patients still in sequences
        print("Getting labels")
        all_labels = df_labels.query("ID in @patients")
        labels = all_labels.drop_duplicates()

        #Encoding labels
        encoder = LabelEncoder()
        encoder = encoder.fit(labels["CKD_stage"])
        labels = encoder.transform(labels["CKD_stage"])

        #Normalization
        if norm:
            ori_data, min_val, max_val = MinMaxScaler(ori_data)
            print("Data normalized with MinMax")

        if save:
            print("Saving data")
            np.save(saved_npy_file_path, ori_data)
            np.save(saved_label_npy_file_path, labels)

    if data_name == "eicu":
        #Pathing to files
        eicu_directory = os.path.join(data_dir, "physionet.org", "files", "eicu-crd", "2.0")
        save_file_name = f"eicuVitalPeriodic_resample{resample_rate}_seqlen{seq_len}"
        save_label_file_name = f"eicuLabels_resample{resample_rate}_seqlen{seq_len}"
        save_patient_file_name = f"eicuPatients_resample{resample_rate}_seqlen{seq_len}"
        save_npy_file_path = os.path.join(data_dir, save_file_name + ".npy")
        save_label_file_path = os.path.join(data_dir, save_label_file_name + ".npy")
        save_patient_file_path = os.path.join(data_dir, save_patient_file_name + ".npy")
        save_csv_file_path = os.path.join(data_dir, save_file_name + ".csv")
        vital_periodic_path = os.path.join(eicu_directory, "vitalPeriodic.csv")
        diagnosis_path = os.path.join(eicu_directory, "diagnosis.csv")

        #Columns used from the diagnosis csv
        diagnosis_cols = [
            "patientunitstayid",
            "diagnosisoffset",
            "diagnosisstring",
        ]

        #Columns, features and datatypes from the vitalPeriodic.csv
        vitals_periodic_cols = ["patientunitstayid",
                                "observationoffset", 
                                "temperature", 
                                "sao2",
                                "heartrate",
                                "systemicmean",
                                "respiration",
        ]

        vitals_periodic_dtypes = {"patientunitstayid": "int32",
                                "observationoffset": "int32",
                                "temperature": "float32", 
                                "sao2": "float32",
                                "heartrate": "float32",
                                "systemicmean": "float32",
                                "respiration": "float32"}
        features = [
            "temperature",
            "spo2",
            "heartrate",
            "systemicmean",
            "respiration"
        ]

        #Importing in dataframes
        print("Diagnosis dataframe loading")
        diagnosis_df = pd.read_csv(
            diagnosis_path,
            usecols=diagnosis_cols
        )
        diagnosis_df.dropna(inplace=True)
        diagnosis_df.rename(columns={"patientunitstayid":"patientid"}, inplace=True)
        diagnosis_df = diagnosis_df[diagnosis_df["diagnosisoffset"] > 0]
        print("Periodic vitals dataframe loading - will take a while - ~3mins on my system")
        df = dd.read_csv(
            vital_periodic_path,
            dtype=vitals_periodic_dtypes,
            usecols=vitals_periodic_cols,
        )
        df = df.dropna().compute()
        df.rename(columns={"patientunitstayid":"patientid", "sao2":"spo2"}, inplace=True)
        df = df[df["observationoffset"] > 0]

        #Resampling to get even timesteps
        print("Begin resampling")
        df["resampleobservationoffset"] = pd.to_timedelta(df["observationoffset"], unit="m")
        df_timestamp = df.set_index("resampleobservationoffset")
        df_resampled = df_timestamp.groupby("patientid")[features].resample(resample_rate).median() #resampling rate should be a parameter in function
        df_resampled.reset_index(inplace=True)
        df_resampled["observationoffset"] = np.int32(df_resampled["resampleobservationoffset"].dt.total_seconds()/60)
        df_resampled.drop("resampleobservationoffset", axis=1, inplace=True)

        #Removing all patients with resample results that have NaNs - uneven sequences
        print("Remove patients with unevenly spaced vitals")
        df_no_nans = df_resampled.groupby("patientid").filter(lambda x: (np.all(x.loc[:, features].isna().sum() == 0)))
        
        #Setting the sequence length of each patient
        print("Finding patients with long enough sequence lengths")
        vital_count = df_no_nans["patientid"].value_counts()
        good_patients = vital_count[vital_count >= seq_len]
        good_patients = pd.Series(good_patients.index.values)
        good_patients = good_patients[good_patients.isin(diagnosis_df["patientid"])]
        df_good_patients = df_no_nans.query("patientid in @good_patients")

        del vital_count, df_no_nans, df_resampled, df_timestamp, df #for memory constraints

        #Getting set sequence lengths for each patient
        print("Rearranging dataframe into NumPy sequence arrays")
        df_sequences = df_good_patients.groupby("patientid").apply(lambda x: x.sort_values("observationoffset").head(seq_len), include_groups=False)
        del df_good_patients
        df_sequences.drop("observationoffset", axis=1, inplace=True)

        #Arranging into NumPy sequences with shape (samples, seq_len, features)
        features_array = df_sequences.values
        ori_data = features_array.reshape(-1, 30, len(features))
        if norm:
            ori_data, min_val, max_val = MinMaxScaler(ori_data)
        print("Data normalized with MinMax")

        #need to save early for memory
        if save:
            print(f"Saving resampled patients with {seq_len} sequence length and {resample_rate} resample rate")
            df_sequences.to_csv(save_csv_file_path)
            print("Saving NumPy sequences array")
            np.save(save_npy_file_path, ori_data)
            print("Saving patients")
            np.save(save_patient_file_path, good_patients)

        #Finding labels in diagnosis that match up with patients used above
        del features_array, df_sequences
        print("Finding disease categories for diseases found in each patient above")
        diagnosis_df["diagnosisstring"] = diagnosis_df["diagnosisstring"].str.partition("|")[0]

        #Removing duplicate diagnoses
        diagnosis_df = diagnosis_df.groupby("patientid").apply((lambda x: x["diagnosisstring"].drop_duplicates()), include_groups=False).reset_index()
        diagnosis_df.drop(columns="level_1", inplace=True)

        #Pulling patients from above 
        diagnosis_df_good_patients = diagnosis_df.query("patientid in @good_patients")

        #Memory issues
        del diagnosis_df
        #Disease categories used for classification
        print("Encoding labels for classification")
        types = ["cardiovascular", "pulmonary"]
        cardio_resp = diagnosis_df_good_patients.groupby("patientid")["diagnosisstring"].agg(list).reset_index()
        cardio_resp["diagnosisstring"] = cardio_resp["diagnosisstring"].apply(disease_categories, args=(types,))
        
        #Label encoding for classification
        encoder = LabelEncoder()
        encoder = encoder.fit(cardio_resp["diagnosisstring"].values)
        labels = encoder.transform(cardio_resp["diagnosisstring"].values)
        if save:
            print("Saving labels")
            np.save(save_label_file_path, labels)

    print(ori_data[0, :, 0])
    return ori_data, labels

if __name__ == "__main__":
  __, __ = real_data_loading("ckd", seq_len=30, resample_rate="15min", save=True, norm=False)