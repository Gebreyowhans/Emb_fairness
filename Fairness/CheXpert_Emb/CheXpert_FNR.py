
import pandas as pd
import numpy as np
import math
import random as python_random
from IPython.display import clear_output
import warnings
import sys
import os


def main():

    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")))
    sys.path.append(os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "MIMIC_CXR_EMB")))

    from MIMIC_CXR_EMB.MIMIC_FNR import FNR_GAPs
    from MIMIC_CXR_EMB.config_MIMIC import get_diseases, get_seeds, get_patient_groups

    diseases = get_diseases()
    patient_groups = get_patient_groups()
    gender = patient_groups["sex"]
    age_decile = patient_groups["age"]
    race = patient_groups["race"]

    factor = [gender, age_decile, race]

    factor_str = ['gender', 'age_decile', 'race']

    seeds = get_seeds()

    for seed in seeds:

        np.random.seed(seed)
        python_random.seed(seed)

        base_path = "./Prediction_results/"

        # Create directory model weightes saving
        fnr_gaps_path = "./FNR_GAPS/"
        os.makedirs(os.path.dirname(fnr_gaps_path), exist_ok=True)

        df = pd.read_csv(f"{base_path}bipred_{seed}.csv").rename(
            columns={'age decile': 'age_decile'})

        ''' TPR Disparities '''

        for i in range(len(factor)):
            FNR_GAPs(
                df, diseases, factor[i], factor_str[i], seed, fnr_gaps_path)

        print(f'SEED : {seed}')


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
