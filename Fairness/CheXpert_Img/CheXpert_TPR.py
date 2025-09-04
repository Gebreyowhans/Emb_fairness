import pandas as pd
import numpy as np
import math
import random as python_random
import io
import os
import sys
import glob
import matplotlib.pyplot as plt
from numpy import linalg as LA
import matplotlib.pyplot as plt

from IPython.display import clear_output
import warnings

def main():
    
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")))
    sys.path.append(os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "MIMIC_CXR_EMB")))

    from Fairness.MIMIC_CXR_EMB.MIMIC_Actual_TPR import TPR_Disparities
    from MIMIC_CXR_EMB.config_MIMIC import get_diseases, get_diseases_abbr, get_seeds, get_patient_groups, get_utility_variables
    
    diseases = get_diseases()
    # diseases_abbr = get_diseases_abbr()

    patient_groups = get_patient_groups()

    gender = patient_groups["sex"]
    age_decile = patient_groups["age"]
    race = patient_groups["race"]
    insurance = patient_groups["insurance"]

    factor = [gender, age_decile]

    factor_str = ['gender', 'age_decile']

    seeds = get_seeds()
    
    
    race_df=pd.read_csv("../CheXpert_Emb/Prediction_results/bipred_19.csv")
    for seed in seeds:

        np.random.seed(seed)
        python_random.seed(seed)
        
        base_path = "./Prediction_results/"
        
        # Create directory model weightes saving
        tpr_gaps_path = "./TPR_GAPS/"
        os.makedirs(os.path.dirname(
            tpr_gaps_path), exist_ok=True)

        true_labels_df = pd.read_csv(f"{base_path}True_withMeta.csv")
        true_labels_df.rename(
            columns={'Airspace Opacity': 'Lung Opacity','Path':'path','Sex':'gender','Age':'age_decile'}, inplace=True)

        bipred_df = pd.read_csv(f"{base_path}bipred_{seed}.csv")
        bipred_df.rename(
            columns={'bi_Airspace Opacity': 'bi_Lung Opacity','Path':'path'}, inplace=True)
        
        # Step 1: Remove the prefix from 'path' in true_labels_df
        true_labels_df = true_labels_df.copy()
        true_labels_df['clean_path'] = true_labels_df['path'].apply(lambda p: '/'.join(p.split('/')[1:]).replace('.jpg', ''))
        
        # Step 3: Merge on the cleaned path
        # full_true_labels_df = true_labels_df.merge(
        #     race_df[['path', 'race']],
        #     left_on='clean_path',
        #     right_on='path',
        #     how='left')
        
        # Step 3: Clean up columns (optional)
        # full_true_labels_df.drop(columns=['clean_path', 'path_y'], inplace=True)
        # full_true_labels_df.rename(columns={'path_x': 'path'}, inplace=True)
        
        # # Save the full DataFrame with
        # full_true_labels_df.to_csv(f"{base_path}full_true_labels_df_{seed}.csv", index=False)
        # full_true_labels_df = []
        # for idx, row  in true_labels_df.iterrows():
            
        #     row_copy = row.copy()
        #     true_labels_df['path'] = true_labels_df['path'].apply(lambda p: '/'.join(p.split('/')[1:]))
            #path = row_copy["path"].split('/')[1:]
            
            # race=race_df[race_df['path']==path]['race'].values[0]
            
            # row_copy["race"]=race
            
            # full_true_labels_df.append(row_copy)
            
            
        # break

        df = true_labels_df.merge(bipred_df, on="path", how="inner")

        print(f'Sample : {df.head()}')
        

        ''' TPR Disparities '''

        for i in range(len(factor)):
            TPR_Disparities(
                df, diseases, factor[i], factor_str[i], seed, tpr_gaps_path)

        print(f'SEED : {seed}')


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
