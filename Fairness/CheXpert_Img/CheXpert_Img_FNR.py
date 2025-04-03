
import pandas as pd
import numpy as np
import math
import random as python_random
from IPython.display import clear_output
import warnings
import sys
import os
from FNR_Disparities_Chexpert_Img import get_Age_FNR_Disparities, get_Race_FNR_Disparities, get_Sex_FNR_Disparities


def main():

    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")))
    sys.path.append(os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "MIMIC_CXR_EMB")))

    from MIMIC_CXR_EMB.MIMIC_FNR import FNR_GAPs
    from MIMIC_CXR_EMB.config_MIMIC import get_diseases, get_diseases_abbr, get_seeds, get_patient_groups, get_utility_variables

    diseases = get_diseases()
    diseases_abbr = get_diseases_abbr()
    patient_groups = get_patient_groups()
    gender = patient_groups["sex"]
    age_decile = patient_groups["age"]
    race = patient_groups["race"]

    utility_variables = get_utility_variables()

    number_of_runs = utility_variables['number_of_runs']
    significance_level = utility_variables['significance_level']
    height = utility_variables['height']
    font_size = utility_variables['font_size']
    rotation_degree = utility_variables['rotation_degree']

    # factor = [gender, age_decile, race]
    factor = [gender, age_decile]
    # factor_str = ['gender', 'age_decile', 'race']
    factor_str = ['gender', 'age_decile']

    seeds = [32, 40, 56, 60, 90]

    fnr_base_path = "./FNR_GAPS/"
    # Create directory fnr saving FNR gaps
    os.makedirs(os.path.dirname(fnr_base_path), exist_ok=True)

    for seed in seeds:

        np.random.seed(seed)
        python_random.seed(seed)

        base_path = "./Prediction_results/"

        true_labels_df = pd.read_csv(f"{base_path}True_withMeta.csv").rename(
            columns={"Sex": 'gender', 'Age': 'age_decile'})

        bipred_df = pd.read_csv(f"{base_path}bipred_{seed}.csv")

        df = true_labels_df.merge(bipred_df, on="Path", how="inner")

        print(f'Sample : {df.head()}')

        ''' FNR Disparities '''

        for i in range(len(factor)):
            FNR_GAPs(
                df, diseases, factor[i], factor_str[i], seed, fnr_base_path)

        print(f'SEED : {seed}')

    # FNR Disparities for SEX
    get_Sex_FNR_Disparities(fnr_base_path, diseases, diseases_abbr,
                            number_of_runs, significance_level, height, font_size, rotation_degree)

    # FNR Disparities for AGE
    get_Age_FNR_Disparities(fnr_base_path, diseases, diseases_abbr,
                            number_of_runs, significance_level, height, font_size, rotation_degree)

    # FNR Disparities for RACE
    # get_Race_FNR_Disparities(fnr_base_path, diseases, diseases_abbr,
    #                          number_of_runs, significance_level, height, font_size, rotation_degree)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
