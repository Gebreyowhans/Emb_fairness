
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
import sys


def main():
    
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")))
    
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "MIMIC_CXR_EMB")))
    
    from MIMIC_CXR_EMB.config_MIMIC import get_diseases, get_diseases_abbr, get_seeds, get_patient_groups, get_utility_variables,get_patient_groups_abbr
    from MIMIC_CXR_EMB.Subgroup_AUC_MIMIC_CXR import subgroup_auc 
    from MIMIC_CXR_EMB.auc_diparitis import get_Age_Auc_Disparities,get_Race_Auc_Disparities,get_Sex_Auc_Disparities
    
    diseases = get_diseases()
    diseases_abbr = get_diseases_abbr()
    patient_groups = get_patient_groups()

    gender = patient_groups["sex"]
    age_decile = patient_groups["age"]
    race = patient_groups["race"]
  
  
    utility_variables = get_utility_variables()
    number_of_runs = utility_variables['number_of_runs']
    significance_level = utility_variables['significance_level']
    group_abrv=get_patient_groups_abbr()
    
    
    seeds = [32, 40, 56, 60, 90]
    
    # factor = [gender]
    factor = [gender, age_decile]
    # factor_str = ['gender']
    factor_str = ['gender', 'age_decile']
    
    auc_base_path = "./AUC_GAPS/"
    os.makedirs(os.path.dirname(auc_base_path), exist_ok=True)
    # Create directory FPR saving
    
    for seed in seeds:
        
        np.random.seed(seed)
        python_random.seed(seed)
        
        base_path = "./Prediction_results/"
        
        true_df = pd.read_csv(f"{base_path}True_withMeta.csv").rename(columns={"Airspace Opacity": "Lung Opacity",
                                                                               "Sex":"gender","Age": "age_decile"})
        bipred_df = pd.read_csv(f"{base_path}bipred_{seed}.csv").rename(columns={"bi_Airspace Opacity": "bi_Lung Opacity"})
        pred_df = pd.read_csv(f"{base_path}preds_{seed}.csv").rename(columns={"prob_Airspace Opacity": "prob_Lung Opacity"})
        
        merged_df = true_df.merge(bipred_df, on="Path")
        df = merged_df.merge(pred_df, on="Path")
        
        # print(f"Columns : {df.columns}   Shape : {df.shape}")


        ''' AUC '''

        for i in range(len(factor)):
            auc_df=subgroup_auc(df, diseases, factor[i], factor_str[i], seed,group_abrv)
            auc_df.to_csv(f"{auc_base_path}Run_seed{str(seed)}_{factor_str[i]}_AUC_GAP.csv", index=False)
        
        
    # auc Disparities for SEX
    get_Sex_Auc_Disparities(auc_base_path, diseases, diseases_abbr,
                           number_of_runs, significance_level,"CheXpert_Img")     
    
    # auc Disparities for AGE
    get_Age_Auc_Disparities(auc_base_path, diseases, diseases_abbr,
                            number_of_runs, significance_level,"CheXpert_Img")  
    
    # auc Disparities for Race
    # get_Race_Auc_Disparities(auc_base_path, diseases, diseases_abbr,
    #                          number_of_runs, significance_level,"CheXpert_Img")      
    
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()