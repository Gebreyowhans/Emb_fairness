
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
    
    seeds = get_seeds()
    
    # factor = [gender]
    factor = [gender, age_decile, race]
    # factor_str = ['gender']
    factor_str = ['gender', 'age_decile', 'race']
    
    auc_base_path = "./AUC_GAPS/"
    os.makedirs(os.path.dirname(auc_base_path), exist_ok=True)
    # Create directory FPR saving
    
    for seed in seeds:
        
        np.random.seed(seed)
        python_random.seed(seed)
        
        base_path = "./Prediction_results/"
        
        df = pd.read_csv(f"{base_path}bipred_{seed}.csv")

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

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()