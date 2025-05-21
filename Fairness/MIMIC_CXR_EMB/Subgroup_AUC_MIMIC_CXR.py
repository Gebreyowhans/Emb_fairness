
from sklearn.metrics import roc_auc_score

import pandas as pd
import numpy as np
import math
import random as python_random
import io
import os
import glob
from IPython.display import clear_output
import warnings

from config_MIMIC import get_diseases, get_diseases_abbr, get_seeds, get_patient_groups, get_utility_variables,get_patient_groups_abbr
from auc_diparitis import get_Sex_Auc_Disparities,get_Insurance_Auc_Disparities,get_Age_Auc_Disparities,get_Race_Auc_Disparities

def get_auc(df,tru_col, prob_col):
    """
    Calculate AUC for a given true label and predicted probabilities.
    
    Parameters:
    - df: DataFrame containing binary labels and probabilities.
    - tru_col: Column name for the true labels.
    - prob_col: Column name for the predicted probabilities.
    
    Returns:
    - auc_score: AUC score for the given true label and predicted probabilities.
    """
    return round(roc_auc_score(df[tru_col].to_numpy().astype(int), df[prob_col].to_numpy().astype(float)), 3)
    
    
def subgroup_auc(df, diseases, sub_group, sub_group_name, seed,group_abrv):
    
    """
    Calculate AUC for each disease within each subgroup (e.g., gender, race).
    
    Parameters:
    - df: DataFrame containing binary labels and probabilities.
    - diseases: List of disease names (e.g., ['Cardiomegaly', 'Edema', ...]).
    - sub_group: List of unique values in the subgroup column (e.g., ['M', 'F']).
    - sub_group_name: Column name for the subgroup (e.g., 'gender').
    - seed: For reproducibility (not used in AUC itself, but could be for logging/shuffling).
    
    Returns:
    - auc_df: DataFrame containing AUC scores for each disease and subgroup.
    """
    
    # if sub_group_name == 'gender':
    #     auc_sex = pd.DataFrame(diseases, columns=["diseases"])
    # Initialize an empty list to store AUC values
    
    result_rows = []

    for disease in diseases:
        
        if disease == 'No Finding':
            continue  # Skip 'No Finding' disease
        
        tru_col = f'{disease}'
        prob_col = f'prob_{disease}'
        

        if tru_col not in df.columns or prob_col not in df.columns:
            continue  # Skip if columns not found
        
        row = {'diseases': disease}
        for group_value in sub_group:
            
            group_df = df[df[sub_group_name] == group_value]
            n_group_df = df[df[sub_group_name] != group_value]
            
       
            if group_df[tru_col].nunique() < 2:
                auc_score = np.nan  # AUC can't be calculated with only one class
            
            else:
                auc_score = get_auc(group_df,tru_col,prob_col)
                n_group_auc_score = get_auc(n_group_df,tru_col,prob_col)
            
        
            if sub_group_name != 'gender':
                
                
                temp_auc = []
                for group_value_1 in sub_group:
                    
                    group_df_1 = df[df[sub_group_name] == group_value_1]
                    
                    _auc=0.0
                    if group_df_1[tru_col].nunique() < 2:
                        _auc = np.nan  # AUC can't be calculated with only one class
            
                    else:
                        _auc = get_auc(group_df_1,tru_col,prob_col)
                    
                    if _auc >=0 and _auc <= 1:
                        temp_auc.append(_auc)
                    
                    temp_auc_filtered = [x for x in temp_auc if not math.isnan(x)]
                    temp_auc_filtered.sort()
                    
                    
                    if len(temp_auc_filtered) % 2 == 0:
                        median = (temp_auc_filtered[(len(temp_auc_filtered) // 2) - 1]
                                + temp_auc_filtered[(len(temp_auc_filtered) // 2)])/2
                    else:
                        median = temp_auc_filtered[(len(temp_auc_filtered) // 2)]
                
                #print(f" label : {disease} Group name : {group_value} temp_auc_filtered: {temp_auc_filtered}")
                GAP = auc_score - median
                
            else:
                GAP= auc_score - n_group_auc_score
        
            gr_name=group_abrv[group_value]
            row[f'Auc_{gr_name}'] = auc_score
            row[f'Gap_{gr_name}'] = GAP
        
        result_rows.append(row)
            
    auc_df = pd.DataFrame(result_rows)
    
    return auc_df
    
        
def main():
    
    diseases = get_diseases()
    diseases_abbr = get_diseases_abbr()
    patient_groups = get_patient_groups()

    gender = patient_groups["sex"]
    age_decile = patient_groups["age"]
    race = patient_groups["race"]
    insurance = patient_groups["insurance"]

    utility_variables = get_utility_variables()
    number_of_runs = utility_variables['number_of_runs']
    significance_level = utility_variables['significance_level']
    group_abrv=get_patient_groups_abbr()
    
    seeds = get_seeds()
    
    # factor = [gender]
    factor = [gender, age_decile, race, insurance]
    # factor_str = ['gender']
    factor_str = ['gender', 'age_decile', 'race', 'insurance']
    
    auc_base_path = "./AUC_GAPS/"
    os.makedirs(os.path.dirname(auc_base_path), exist_ok=True)
    # Create directory FPR saving
    
    for seed in seeds:
        
        np.random.seed(seed)
        python_random.seed(seed)
        
        base_path = "./Prediction_results/"
        
        df = pd.read_csv(f"{base_path}bipred_{seed}.csv").rename(
            columns={'age decile': 'age_decile'})

        ''' AUC '''

        for i in range(len(factor)):
            auc_df=subgroup_auc(df, diseases, factor[i], factor_str[i], seed,group_abrv)
            auc_df.to_csv(f"{auc_base_path}Run_seed{str(seed)}_{factor_str[i]}_AUC_GAP.csv", index=False)
        
        
        # auc Disparities for SEX
    get_Sex_Auc_Disparities(auc_base_path, diseases, diseases_abbr,
                            number_of_runs, significance_level)     
    
    # auc Disparities for AGE
    get_Age_Auc_Disparities(auc_base_path, diseases, diseases_abbr,
                            number_of_runs, significance_level)  
    
    # auc Disparities for Race
    get_Race_Auc_Disparities(auc_base_path, diseases, diseases_abbr,
                            number_of_runs, significance_level)      
    
    # auc Disparities for Race
    get_Insurance_Auc_Disparities(auc_base_path, diseases, diseases_abbr,
                            number_of_runs, significance_level)   

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()