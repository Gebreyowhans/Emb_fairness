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

    # from Fairness.MIMIC_CXR_EMB.MIMIC_Actual_TPR import TPR_Disparities
    from MIMIC_CXR_EMB.config_MIMIC import get_diseases, get_diseases_abbr, get_seeds, get_patient_groups, get_utility_variables
    from MIMIC_CXR_EMB.Actual_TPR import TPR_14
    
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

    for seed in seeds:

        np.random.seed(seed)
        python_random.seed(seed)
        
        base_path = "./Prediction_results/"
        
        # Create directory model weightes saving
        tpr_gaps_path = "./Actual_TPR/"
        os.makedirs(os.path.dirname(
            tpr_gaps_path), exist_ok=True)

        true_labels_df = pd.read_csv(f"{base_path}True_withMeta.csv")
        true_labels_df.rename(
            columns={'Airspace Opacity': 'Lung Opacity','Path':'path','Sex':'gender','Age':'age_decile'}, inplace=True)

        bipred_df = pd.read_csv(f"{base_path}bipred_{seed}.csv")
        bipred_df.rename(
            columns={'bi_Airspace Opacity': 'bi_Lung Opacity','Path':'path'}, inplace=True)

        df = true_labels_df.merge(bipred_df, on="path", how="inner")

        # print(f'Sample : {df.head()}')
        

        ''' TPR Disparities '''

        for i in range(len(factor)):
            TPR_14(
                df, diseases, factor[i], factor_str[i], seed, tpr_gaps_path)

        print(f'SEED : {seed}')


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
