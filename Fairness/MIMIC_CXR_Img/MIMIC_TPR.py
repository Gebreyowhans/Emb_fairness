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

def tpr(df, d, c, category_name):

    pred_disease = "bi_" + d
    gt = df.loc[(df[d] == 1) & (df[category_name] == c), :]
    pred = df.loc[(df[pred_disease] == 1) & (
        df[d] == 1) & (df[category_name] == c), :]

    if len(gt) != 0:
        TPR = len(pred) / len(gt)
        return TPR
    else:
        # print("Disease", d, "in category", c, "has zero division error")
        return np.NAN


def TPR_Disparities(df, diseases, category, category_name, seed=19, tpr_gaps_results_path_dir_dir="default"):

    plt.rcParams.update({'font.size': 18})
    GAP_total = []
    percentage_total = []
    cate = []

    if category_name == 'gender':
        Run1_sex = pd.DataFrame(diseases, columns=["diseases"])

    if category_name == 'age_decile':
        Run1_age = pd.DataFrame(diseases, columns=["diseases"])

    if category_name == 'race':
        Run1_race = pd.DataFrame(diseases, columns=["diseases"])

    if category_name == 'insurance':
        Run1_insurance = pd.DataFrame(diseases, columns=["diseases"])

    for c in category:

        GAP_y = []
        percentage_y = []

        for d in diseases:

            pred_disease = "bi_" + d

            gt = df.loc[(df[d] == 1) & (df[category_name] == c), :]
            pred = df.loc[(df[pred_disease] == 1) & (
                df[d] == 1) & (df[category_name] == c), :]

            n_gt = df.loc[(df[d] == 1) & (df[category_name] != c)
                          & (df[category_name] != 0), :]

            n_pred = df.loc[(df[pred_disease] == 1) & (df[d] == 1) & (
                df[category_name] != c) & (df[category_name] != 0), :]

            pi_gy = df.loc[(df[d] == 1) & (df[category_name] == c), :]
            pi_y = df.loc[(df[d] == 1) & (df[category_name] != 0), :]

            if len(gt) != 0 and len(n_gt) != 0 and len(pi_y) != 0:

                TPR = len(pred) / len(gt)

                n_TPR = len(n_pred) / len(n_gt)

                percentage = len(pi_gy) / len(pi_y)

                if category_name != 'gender':
                    temp_TPR = []

                    for c1 in category:
                        _tpr = tpr(df, d, c1, category_name)

                        if _tpr != -1:
                            temp_TPR.append(_tpr)

                    temp_TPR_Filtered = [
                        x for x in temp_TPR if not math.isnan(x)]
                    temp_TPR_Filtered.sort()

                    if len(temp_TPR_Filtered) % 2 == 0:

                        median = (temp_TPR_Filtered[(len(temp_TPR_Filtered) // 2) - 1]
                                  + temp_TPR_Filtered[(len(temp_TPR_Filtered) // 2)])/2

                    else:
                        median = temp_TPR_Filtered[(
                            len(temp_TPR_Filtered) // 2)]

                    GAP = TPR - median

                else:

                    GAP = TPR - n_TPR

                """ This portion of code is used for debuging purpose only """
                # if category_name=='age_decile' and c=='60-80' and d=='Fracture':
                #   print(f'Current category : {c}')
                #   print(f'Current disease : {d}')
                #   pdb.set_trace()

                GAP_y.append(GAP)
                percentage_y.append(percentage)

            else:

                GAP_y.append(np.NAN)
                percentage_y.append(0)

        GAP_total.append(GAP_y)
        percentage_total.append(percentage_y)

        c = c.replace(' ', '_', 3)
        c = c.replace('/', '_', 3)
        cate.append(c)

    GAP_total = np.array(GAP_total)

    if category_name == 'age_decile':
        print(f'GAP_total: {GAP_total}')

    # Create a new array of x-values for the non-NaN diseases
    x = np.arange(len(diseases))

    print("len(GAP_total): ", len(GAP_total))
    for i in range(len(GAP_total)):

        if category_name == 'age_decile':

            if i == 0:

                Percent0 = pd.DataFrame(
                    percentage_total[i], columns=["%60-80"])
                Run1_age = pd.concat(
                    [Run1_age, Percent0.reindex(Run1_age.index)], axis=1)

                Gap0 = pd.DataFrame(GAP_total[i], columns=["Gap_60-80"])
                Run1_age = pd.concat(
                    [Run1_age, Gap0.reindex(Run1_age.index)], axis=1)

            if i == 1:

                Percent1 = pd.DataFrame(
                    percentage_total[i], columns=["%40-60"])
                Run1_age = pd.concat(
                    [Run1_age, Percent1.reindex(Run1_age.index)], axis=1)

                Gap1 = pd.DataFrame(GAP_total[i], columns=["Gap_40-60"])
                Run1_age = pd.concat(
                    [Run1_age, Gap1.reindex(Run1_age.index)], axis=1)

            if i == 2:

                Percent2 = pd.DataFrame(
                    percentage_total[i], columns=["%20-40"])
                Run1_age = pd.concat(
                    [Run1_age, Percent2.reindex(Run1_age.index)], axis=1)

                Gap2 = pd.DataFrame(GAP_total[i], columns=["Gap_20-40"])
                Run1_age = pd.concat(
                    [Run1_age, Gap2.reindex(Run1_age.index)], axis=1)

            if i == 3:

                Percent3 = pd.DataFrame(percentage_total[i], columns=["%80+"])
                Run1_age = pd.concat(
                    [Run1_age, Percent3.reindex(Run1_age.index)], axis=1)

                Gap3 = pd.DataFrame(GAP_total[i], columns=["Gap_80+"])
                Run1_age = pd.concat(
                    [Run1_age, Gap3.reindex(Run1_age.index)], axis=1)

            if i == 4:

                Percent4 = pd.DataFrame(percentage_total[i], columns=["%0-20"])
                Run1_age = pd.concat(
                    [Run1_age, Percent4.reindex(Run1_age.index)], axis=1)

                Gap4 = pd.DataFrame(GAP_total[i], columns=["Gap_0-20"])
                Run1_age = pd.concat(
                    [Run1_age, Gap4.reindex(Run1_age.index)], axis=1)

            Run1_age.to_csv(tpr_gaps_results_path_dir_dir +
                            "Run_seed"+str(seed)+"_TPR_GAP_Age.csv")

        if category_name == 'gender':

            if i == 0:

                MalePercent = pd.DataFrame(percentage_total[i], columns=["%M"])
                Run1_sex = pd.concat(
                    [Run1_sex, MalePercent.reindex(Run1_sex.index)], axis=1)

                MaleGap = pd.DataFrame(GAP_total[i], columns=["Gap_M"])
                Run1_sex = pd.concat(
                    [Run1_sex, MaleGap.reindex(Run1_sex.index)], axis=1)

            else:

                FeMalePercent = pd.DataFrame(
                    percentage_total[i], columns=["%F"])
                Run1_sex = pd.concat(
                    [Run1_sex, FeMalePercent.reindex(Run1_sex.index)], axis=1)

                FeMaleGap = pd.DataFrame(GAP_total[i], columns=["Gap_F"])
                Run1_sex = pd.concat(
                    [Run1_sex, FeMaleGap.reindex(Run1_sex.index)], axis=1)

            Run1_sex.to_csv(tpr_gaps_results_path_dir_dir +
                            "Run_seed"+str(seed)+"_TPR_GAP_sex.csv")

        if category_name == 'race':

            if i == 0:

                Percent_White = pd.DataFrame(
                    percentage_total[i], columns=["%White"])
                Run1_race = pd.concat(
                    [Run1_race, Percent_White.reindex(Run1_race.index)], axis=1)

                Gap_White = pd.DataFrame(GAP_total[i], columns=["Gap_White"])
                Run1_race = pd.concat(
                    [Run1_race, Gap_White.reindex(Run1_race.index)], axis=1)

            if i == 1:

                Percent_Black = pd.DataFrame(
                    percentage_total[i], columns=["%Black"])
                Run1_race = pd.concat(
                    [Run1_race, Percent_Black.reindex(Run1_race.index)], axis=1)

                Gap_Black = pd.DataFrame(GAP_total[i], columns=["Gap_Black"])
                Run1_race = pd.concat(
                    [Run1_race, Gap_Black.reindex(Run1_race.index)], axis=1)

            if i == 2:

                Percent_Hisp = pd.DataFrame(
                    percentage_total[i], columns=["%Hisp"])
                Run1_race = pd.concat(
                    [Run1_race, Percent_Hisp.reindex(Run1_race.index)], axis=1)

                Gap_Hisp = pd.DataFrame(GAP_total[i], columns=["Gap_Hisp"])
                Run1_race = pd.concat(
                    [Run1_race, Gap_Hisp.reindex(Run1_race.index)], axis=1)

            if i == 3:

                Percent_Other = pd.DataFrame(
                    percentage_total[i], columns=["%Other"])
                Run1_race = pd.concat(
                    [Run1_race, Percent_Other.reindex(Run1_race.index)], axis=1)

                Gap_Other = pd.DataFrame(GAP_total[i], columns=["Gap_Other"])
                Run1_race = pd.concat(
                    [Run1_race, Gap_Other.reindex(Run1_race.index)], axis=1)

            if i == 4:

                Percent_Asian = pd.DataFrame(
                    percentage_total[i], columns=["%Asian"])
                Run1_race = pd.concat(
                    [Run1_race, Percent_Asian.reindex(Run1_race.index)], axis=1)

                Gap_Asian = pd.DataFrame(GAP_total[i], columns=["Gap_Asian"])
                Run1_race = pd.concat(
                    [Run1_race, Gap_Asian.reindex(Run1_race.index)], axis=1)

            if i == 5:

                Percent_American = pd.DataFrame(
                    percentage_total[i], columns=["%American"])
                Run1_race = pd.concat(
                    [Run1_race, Percent_American.reindex(Run1_race.index)], axis=1)

                Gap_American = pd.DataFrame(
                    GAP_total[i], columns=["Gap_American"])
                Run1_race = pd.concat(
                    [Run1_race, Gap_American.reindex(Run1_race.index)], axis=1)

            Run1_race.to_csv(tpr_gaps_results_path_dir_dir +
                             "Run_seed"+str(seed)+"_TPR_GAP_race.csv")

        if category_name == 'insurance':
            if i == 0:
                Percent_Medicare = pd.DataFrame(
                    percentage_total[i], columns=["%Medicare"])
                Run1_insurance = pd.concat([Run1_insurance,
                                            Percent_Medicare.reindex(Run1_insurance.index)], axis=1)

                Gap_Medicare = pd.DataFrame(
                    GAP_total[i], columns=["Gap_Medicare"])
                Run1_insurance = pd.concat([Run1_insurance,
                                            Gap_Medicare.reindex(Run1_insurance.index)], axis=1)
            if i == 1:
                Percent_Other = pd.DataFrame(
                    percentage_total[i], columns=["%Other"])
                Run1_insurance = pd.concat([Run1_insurance,
                                            Percent_Other.reindex(Run1_insurance.index)], axis=1)

                Gap_Other = pd.DataFrame(GAP_total[i], columns=["Gap_Other"])
                Run1_insurance = pd.concat([Run1_insurance,
                                            Gap_Other.reindex(Run1_insurance.index)], axis=1)
            if i == 2:
                Percent_Medicaid = pd.DataFrame(
                    percentage_total[i], columns=["%Medicaid"])
                Run1_insurance = pd.concat([Run1_insurance,
                                            Percent_Medicaid.reindex(Run1_insurance.index)], axis=1)

                Gap_Medicaid = pd.DataFrame(
                    GAP_total[i], columns=["Gap_Medicaid"])
                Run1_insurance = pd.concat([Run1_insurance,
                                            Gap_Medicaid.reindex(Run1_insurance.index)], axis=1)

            Run1_insurance.to_csv(
                tpr_gaps_results_path_dir_dir+"Run_seed"+str(seed)+"_TPR_GAP_insurance.csv")


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

    factor = [gender, age_decile, race, insurance]

    factor_str = ['gender', 'age_decile', 'race', 'insurance']

    seeds = get_seeds()

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
            columns={'Airspace Opacity': 'Lung Opacity'}, inplace=True)

        bipred_df = pd.read_csv(f"{base_path}bipred_{seed}.csv")
        bipred_df.rename(
            columns={'bi_Airspace Opacity': 'bi_Lung Opacity'}, inplace=True)

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
