import pandas as pd
import numpy as np
import math
import random as python_random
import io
import os
import glob
import matplotlib.pyplot as plt
from numpy import linalg as LA
import matplotlib.pyplot as plt

from IPython.display import clear_output
import warnings

from config_MIMIC import get_diseases, get_diseases_abbr, get_seeds, get_patient_groups

from Actual_TPR import TPR_14

def main():

    diseases = get_diseases()
    # diseases_abbr = get_diseases_abbr()

    patient_groups = get_patient_groups()

    gender = patient_groups["sex"]
    age_decile = patient_groups["age"]
    race = patient_groups["race"]
    insurance = patient_groups["insurance"]

    factor = [gender, age_decile, race, insurance]

    factor_str = ['gender', 'age_decile', 'race', 'insurance']

    seeds = get_seeds()

    for seed in seeds:

        np.random.seed(seed)
        python_random.seed(seed)

        base_path = "./Prediction_results/"

        # Create directory model weightes saving
        tpr_gaps_path = "./Actual_TPR/"
        os.makedirs(os.path.dirname(
            tpr_gaps_path), exist_ok=True)

        df = pd.read_csv(f"{base_path}bipred_{seed}.csv").rename(
            columns={'age decile': 'age_decile'})

        ''' TPR Disparities '''

        for i in range(len(factor)):
            TPR_14(
                df, diseases, factor[i], factor_str[i], seed, tpr_gaps_path)

        print(f'SEED : {seed}')

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
