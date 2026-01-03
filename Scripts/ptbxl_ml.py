import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import ast

def plot_signal(ecg_id: int, signals: np.ndarray, fields: dict) -> None:
    '''
    Plots lead signal 2 for the corresponding record using the provided signals and fields.
    
    Args:
        ecg_id: The record to plot
        signals: A matrix consisting of the record's ECG signals
        fields: Data that determines the labels of the axes
        
    Returns:
        None:
    '''
    plt.figure(figsize=(12,6))
    plt.plot(signals, label=fields['sig_name'][1])
    plt.title(f"ECG Record {ecg_id}: Lead Signal 2")
    plt.xlabel("Time Points")
    plt.ylabel("Amplitude (mV)")
    plt.legend()
    plt.show()


def load_database(file_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Loads a compressed ".npz" database from the specified path. Two identically indexed DataFrames are returned.
    
    Args:
        file_path: The path containing the compressed ".npz" file
        
    Returns:
        tuple: A tuple consisting of two DataFrames, one containing the PTB-XL metadata and the second containing
        all corresponding ECG signals
    '''
    # Loading the compressed file
    database = np.load(file_path, allow_pickle=True)
    
    # Extracting all the data from the loaded file
    ecg_ids = database['ecg_ids']
    signals = database['signals']
    fields = database['fields']
    metadata = database['metadata']
    col_names = database['col_names']
    
    data = {}
    
    for i, col in enumerate(col_names):
        data[col] = pd.Series(metadata[:, i], index=ecg_ids)
    
    # Casting data in fields to literal dictionaries
    fields = np.array([ast.literal_eval(str(field)) for field in fields])
    
    # Adding fields and superclasses to the data
    data['fields'] = fields
    
    ptbxl_df = pd.DataFrame(data=data, index=ecg_ids)
    signals_df = pd.DataFrame(data=signals, index=ecg_ids, dtype=float)
    
    return (ptbxl_df, signals_df)