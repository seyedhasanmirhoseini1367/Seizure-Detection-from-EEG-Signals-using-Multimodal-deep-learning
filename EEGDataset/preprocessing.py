
 
 
def list_file_name(path):
    # List all files in the train directory
    paths = os.listdir(path)
    file_names = [filename.split('.')[0] for filename in paths]

    return file_names
 
 
train_path = 'D:/Thesis/hms-harmful-brain-activity-classification/train_eegs'
train_ids = list_file_name(train_path)
 
def finding_Nan(directory_path, id_list, file_names, df):
    valid_list = []  # Stores IDs with no NaN values
    remove_ids = []  # Stores IDs to be removed
    for file_id in file_names:
        if file_id in id_list:
            file_path = os.path.join(directory_path, f'{file_id}.parquet')
            try:
                data = pd.read_parquet(file_path)
                if data.isna().any().any():  # Check if any NaN exists
                    remove_ids.append(file_id)
                    os.remove(file_path)  # Remove file with NaN values
                else:
                    valid_list.append(file_id)
            except Exception:
                pass  # If there's an error reading the file, skip it
        else:
            # File ID not in `df`, so remove it
            remove_ids.append(file_id)
            file_path = os.path.join(directory_path, f'{file_id}.parquet')
            os.remove(file_path)
 
    # Filter `total_df` to only keep rows with IDs in `valid_list`
    df = df.copy()
    df = df[df['eeg_id'].isin(valid_list)].reset_index(drop=True)
    return valid_list, df


df = pd.read_csv('D:/Thesis/hms-harmful-brain-activity-classification/final_dataset.csv')
ids_list = [str(eeg_id) for eeg_id in total_df['eeg_id']]
nan_train_files, filtered_df = finding_Nan(train_path, ids_list, train_ids, df)
valid_train_files = [int(eeg_id) for eeg_id in nan_train_files]
df = df[df['eeg_id'].isin(valid_train_files)].reset_index(drop=True)
 
df['target'].value_counts()
df['target'].value_counts()/ df.shape[0] * 100
# Now, `filtered_df` has rows matching non-NaN files in the directory, and both are in sync
print(f"Valid files: {len(nan_train_files)}")
print(f"Filtered total_df length: {len(df)}")
 
Valid files: 16164
Filtered df length: 16164

 
df['target'].value_counts()
Other      6790
Seizure    2637
LPD        2433
GPD        1711
GRDA       1710
LRDA        883

df['target'].value_counts()/ df.shape[0] * 100
Other      42.00 %
Seizure    16.31 %
LPD          15.05 %
GPD         10.58 %
GRDA       10.57 %
LRDA        5.46 %

 
