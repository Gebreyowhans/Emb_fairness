from config_MIMIC import get_diseases, get_diseases_abbr, get_seeds, get_patient_groups, get_utility_variables

from FPR_Disparities_MIMIC_CXR import get_Sex_FPR_Disparities, get_Age_FPR_Disparities, get_Race_FPR_Disparities, get_Insurance_FPR_Disparities

import pandas as pd
import numpy as np
import math
import random as python_random
import io
import os
import glob
from IPython.display import clear_output
import warnings

def fpr(df, d, c, category_name):
    
    pred_disease = "bi_" + d
    gt = df.loc[(df[d] == 0) & (df[category_name] == c), :]
    pred = df.loc[(df[pred_disease] == 1) & (df[d] == 0) & (df[category_name] == c), :]
    
    if len(gt) != 0:
        FPR = len(pred) / len(gt)
        return FPR
    else:
        # print("Disease", d, "in category", c, "has zero division error")
        return np.NAN


def FPR_GAPs(df, diseases, category, category_name, seed=19, FPR_gaps_results_path="default"):

    GAP_total = []
    img_count_total = []
    FN_total = []

    cate = []

    if category_name == 'gender':
        FPR_Sex = pd.DataFrame(diseases, columns=["diseases"])

    if category_name == 'age_decile' or category_name == 'age decile':
        FPR_Age = pd.DataFrame(diseases, columns=["diseases"])

    if category_name == 'race':
        FPR_Race = pd.DataFrame(diseases, columns=["diseases"])

    if category_name == 'insurance':
        FPR_insu = pd.DataFrame(diseases, columns=["diseases"])

    for c in category:

        img_count_cate = []
        FN_cate = []
        GAP_y = []

        for d in diseases:

            pred_disease = "bi_" + d
            # print(f'Disease : {pred_disease}  category : {c} ')
            
            gt_fp = df.loc[(df[d] == 0) & (df[category_name] == c), :]
            pred_fp = df.loc[(df[pred_disease] == 1) & (df[d] == 0) & (df[category_name] == c), :]
            
            
            n_gt_fp = df.loc[(df[d] == 0) & (df[category_name] != c) & (df[category_name] != 0), :]
            
            n_pred_fp = df.loc[(df[pred_disease] == 1) & (df[d] == 0) & (df[category_name] != c) & (df[category_name] != 0), :]
            
            # print(f'pred_fp: {len(pred_fp)} gt_fp: {len(gt_fp)} n_pred_fp: {len(n_pred_fp)} n_gt_fp: {len(n_gt_fp)}')
            if len(gt_fp) != 0:

                FPR = len(pred_fp) / len(gt_fp)
                n_FPR = len(n_pred_fp) / len(n_gt_fp)

                FN_cate.append(round(FPR, 3))
                img_count_cate.append(round(len(gt_fp), 3))

                if category_name != 'gender':

                    temp_FPR = []
                    for c1 in category:

                        _FPR = fpr(df, d, c1, category_name)

                        if _FPR != -1:
                            temp_FPR.append(_FPR)

                    temp_FPR_Filtered = [
                        x for x in temp_FPR if not math.isnan(x)]
                    temp_FPR_Filtered.sort()

                    if len(temp_FPR_Filtered) % 2 == 0:

                        median = (temp_FPR_Filtered[(len(temp_FPR_Filtered) // 2) - 1]
                                  + temp_FPR_Filtered[(len(temp_FPR_Filtered) // 2)])/2

                    else:
                        median = temp_FPR_Filtered[(
                            len(temp_FPR_Filtered) // 2)]

                    GAP = FPR - median

                else:
                    GAP = FPR - n_FPR

                GAP_y.append(GAP)

            else:
                FN_cate.append(np.NaN)
                GAP_y.append(np.NAN)

        # FPR of all 14 diseases and categories
        img_count_total.append(img_count_cate)
        FN_total.append(FN_cate)
        GAP_total.append(GAP_y)

        c = c.replace(' ', '_', 3)
        c = c.replace('/', '_', 3)
        cate.append(c)

    GAP_total = np.array(GAP_total)

    print(f'GAP_total: {GAP_total}')

    # Create a new array of x-values for the non-NaN diseases

    for i in range(len(GAP_total)):

        if category_name == 'age_decile' or category_name == 'age decile':

            if i == 0:
                img_count_0 = pd.DataFrame(
                    img_count_total[i], columns=["#60-80"])
                FPR_Age = pd.concat([FPR_Age, img_count_0.reindex(FPR_Age.index)],
                                    axis=1)

                FPR_0 = pd.DataFrame(FN_total[i], columns=["FPR_60-80"])
                FPR_Age = pd.concat([FPR_Age, FPR_0.reindex(FPR_Age.index)],
                                    axis=1)

                Gap0 = pd.DataFrame(GAP_total[i], columns=["Gap_60-80"])
                FPR_Age = pd.concat(
                    [FPR_Age, Gap0.reindex(FPR_Age.index)], axis=1)

            if i == 1:
                img_count_1 = pd.DataFrame(
                    img_count_total[i], columns=["#40-60"])
                FPR_Age = pd.concat([FPR_Age, img_count_1.reindex(FPR_Age.index)],
                                    axis=1)

                FPR_1 = pd.DataFrame(FN_total[i], columns=["FPR_40-60"])
                FPR_Age = pd.concat([FPR_Age, FPR_1.reindex(FPR_Age.index)],
                                    axis=1)

                Gap1 = pd.DataFrame(GAP_total[i], columns=["Gap_40-60"])
                FPR_Age = pd.concat(
                    [FPR_Age, Gap1.reindex(FPR_Age.index)], axis=1)

            if i == 2:
                img_count_2 = pd.DataFrame(
                    img_count_total[i], columns=["#20-40"])
                FPR_Age = pd.concat([FPR_Age, img_count_2.reindex(FPR_Age.index)],
                                    axis=1)

                FPR_2 = pd.DataFrame(FN_total[i], columns=["FPR_20-40"])
                FPR_Age = pd.concat([FPR_Age, FPR_2.reindex(FPR_Age.index)],
                                    axis=1)

                Gap2 = pd.DataFrame(GAP_total[i], columns=["Gap_20-40"])
                FPR_Age = pd.concat(
                    [FPR_Age, Gap2.reindex(FPR_Age.index)], axis=1)

            if i == 3:

                img_count_3 = pd.DataFrame(
                    img_count_total[i], columns=["#80-"])
                FPR_Age = pd.concat([FPR_Age, img_count_3.reindex(FPR_Age.index)],
                                    axis=1)

                FPR_3 = pd.DataFrame(FN_total[i], columns=["FPR_80-"])
                FPR_Age = pd.concat([FPR_Age, FPR_3.reindex(FPR_Age.index)],
                                    axis=1)

                Gap3 = pd.DataFrame(GAP_total[i], columns=["Gap_80-"])
                FPR_Age = pd.concat(
                    [FPR_Age, Gap3.reindex(FPR_Age.index)], axis=1)

            if i == 4:
                img_count_4 = pd.DataFrame(
                    img_count_total[i], columns=["#0-20"])
                FPR_Age = pd.concat([FPR_Age, img_count_4.reindex(FPR_Age.index)],
                                    axis=1)

                FPR_4 = pd.DataFrame(FN_total[i], columns=["FPR_0-20"])
                FPR_Age = pd.concat([FPR_Age, FPR_4.reindex(FPR_Age.index)],
                                    axis=1)

                Gap4 = pd.DataFrame(GAP_total[i], columns=["Gap_0-20"])
                FPR_Age = pd.concat(
                    [FPR_Age, Gap4.reindex(FPR_Age.index)], axis=1)

                FPR_Age.to_csv(FPR_gaps_results_path+"Run_seed" +
                               str(seed)+"_FPR_GAP_Age.csv", index=False)

        if category_name == 'gender':

            if i == 0:

                img_count_0 = pd.DataFrame(img_count_total[i], columns=["#M"])
                FPR_Sex = pd.concat([FPR_Sex, img_count_0.reindex(FPR_Sex.index)],
                                    axis=1)

                FPR_0 = pd.DataFrame(FN_total[i], columns=["FPR_M"])
                FPR_Sex = pd.concat([FPR_Sex, FPR_0.reindex(FPR_Sex.index)],
                                    axis=1)

                Gap0 = pd.DataFrame(GAP_total[i], columns=["Gap_M"])
                FPR_Sex = pd.concat(
                    [FPR_Sex, Gap0.reindex(FPR_Sex.index)], axis=1)

            else:
                img_count_1 = pd.DataFrame(img_count_total[i], columns=["#F"])
                FPR_Sex = pd.concat([FPR_Sex, img_count_1.reindex(FPR_Sex.index)],
                                    axis=1)

                FPR_1 = pd.DataFrame(FN_total[i], columns=["FPR_F"])
                FPR_Sex = pd.concat([FPR_Sex, FPR_1.reindex(FPR_Sex.index)],
                                    axis=1)

                Gap1 = pd.DataFrame(GAP_total[i], columns=["Gap_F"])
                FPR_Sex = pd.concat(
                    [FPR_Sex, Gap1.reindex(FPR_Sex.index)], axis=1)

                FPR_Sex.to_csv(FPR_gaps_results_path+"Run_seed" +
                               str(seed)+"_FPR_GAP_sex.csv", index=False)

        if category_name == 'race':

            if i == 0:
                img_count_0 = pd.DataFrame(
                    img_count_total[i], columns=["#White"])
                FPR_Race = pd.concat([FPR_Race, img_count_0.reindex(FPR_Race.index)],
                                     axis=1)

                FPR_0 = pd.DataFrame(FN_total[i], columns=["FPR_White"])
                FPR_Race = pd.concat([FPR_Race, FPR_0.reindex(FPR_Race.index)],
                                     axis=1)

                Gap0 = pd.DataFrame(GAP_total[i], columns=["Gap_White"])
                FPR_Race = pd.concat(
                    [FPR_Race, Gap0.reindex(FPR_Race.index)], axis=1)

            if i == 1:
                img_count_1 = pd.DataFrame(
                    img_count_total[i], columns=["#Black"])
                FPR_Race = pd.concat([FPR_Race, img_count_1.reindex(FPR_Race.index)],
                                     axis=1)

                FPR_1 = pd.DataFrame(FN_total[i], columns=["FPR_Black"])
                FPR_Race = pd.concat([FPR_Race, FPR_1.reindex(FPR_Race.index)],
                                     axis=1)

                Gap1 = pd.DataFrame(GAP_total[i], columns=["Gap_Black"])
                FPR_Race = pd.concat(
                    [FPR_Race, Gap1.reindex(FPR_Race.index)], axis=1)

            if i == 2:
                img_count_2 = pd.DataFrame(
                    img_count_total[i], columns=["#Hisp"])
                FPR_Race = pd.concat([FPR_Race, img_count_2.reindex(FPR_Race.index)],
                                     axis=1)

                FPR_2 = pd.DataFrame(FN_total[i], columns=["FPR_Hisp"])
                FPR_Race = pd.concat([FPR_Race, FPR_2.reindex(FPR_Race.index)],
                                     axis=1)

                Gap2 = pd.DataFrame(GAP_total[i], columns=["Gap_Hisp"])
                FPR_Race = pd.concat(
                    [FPR_Race, Gap2.reindex(FPR_Race.index)], axis=1)

            if i == 3:
                img_count_3 = pd.DataFrame(
                    img_count_total[i], columns=["#Other"])
                FPR_Race = pd.concat([FPR_Race, img_count_3.reindex(FPR_Race.index)],
                                     axis=1)

                FPR_3 = pd.DataFrame(FN_total[i], columns=["FPR_Other"])
                FPR_Race = pd.concat([FPR_Race, FPR_3.reindex(FPR_Race.index)],
                                     axis=1)

                Gap3 = pd.DataFrame(GAP_total[i], columns=["Gap_Other"])
                FPR_Race = pd.concat(
                    [FPR_Race, Gap3.reindex(FPR_Race.index)], axis=1)

            if i == 4:

                img_count_4 = pd.DataFrame(
                    img_count_total[i], columns=["#Asian"])
                FPR_Race = pd.concat([FPR_Race, img_count_4.reindex(FPR_Race.index)],
                                     axis=1)

                FPR_4 = pd.DataFrame(FN_total[i], columns=["FPR_Asian"])
                FPR_Race = pd.concat([FPR_Race, FPR_4.reindex(FPR_Race.index)],
                                     axis=1)

                Gap4 = pd.DataFrame(GAP_total[i], columns=["Gap_Asian"])
                FPR_Race = pd.concat(
                    [FPR_Race, Gap4.reindex(FPR_Race.index)], axis=1)

            if i == 5:
                img_count_5 = pd.DataFrame(
                    img_count_total[i], columns=["#American"])
                FPR_Race = pd.concat([FPR_Race, img_count_5.reindex(FPR_Race.index)],
                                     axis=1)

                FPR_5 = pd.DataFrame(FN_total[i], columns=["FPR_American"])
                FPR_Race = pd.concat([FPR_Race, FPR_5.reindex(FPR_Race.index)],
                                     axis=1)

                Gap5 = pd.DataFrame(GAP_total[i], columns=["Gap_American"])
                FPR_Race = pd.concat(
                    [FPR_Race, Gap5.reindex(FPR_Race.index)], axis=1)

                FPR_Race.to_csv(FPR_gaps_results_path +
                                "Run_seed"+str(seed)+"_FPR_GAP_race.csv", index=False)

        if category_name == 'insurance':

            if i == 0:

                img_count_0 = pd.DataFrame(
                    img_count_total[i], columns=["#medicare"])
                FPR_insu = pd.concat([FPR_insu, img_count_0.reindex(FPR_insu.index)],
                                     axis=1)

                FPR_0 = pd.DataFrame(FN_total[i], columns=["FPR_medicare"])
                FPR_insu = pd.concat([FPR_insu, FPR_0.reindex(FPR_insu.index)],
                                     axis=1)

                Gap0 = pd.DataFrame(GAP_total[i], columns=["Gap_medicare"])
                FPR_insu = pd.concat(
                    [FPR_insu, Gap0.reindex(FPR_insu.index)], axis=1)

            if i == 1:
                img_count_1 = pd.DataFrame(
                    img_count_total[i], columns=["#other"])
                FPR_insu = pd.concat([FPR_insu, img_count_1.reindex(FPR_insu.index)],
                                     axis=1)

                FPR_1 = pd.DataFrame(FN_total[i], columns=["FPR_other"])
                FPR_insu = pd.concat([FPR_insu, FPR_1.reindex(FPR_insu.index)],
                                     axis=1)

                Gap1 = pd.DataFrame(GAP_total[i], columns=["Gap_other"])
                FPR_insu = pd.concat(
                    [FPR_insu, Gap1.reindex(FPR_insu.index)], axis=1)

            if i == 2:

                img_count_2 = pd.DataFrame(
                    img_count_total[i], columns=["#medicaid"])
                FPR_insu = pd.concat([FPR_insu, img_count_2.reindex(FPR_insu.index)],
                                     axis=1)

                FPR_2 = pd.DataFrame(FN_total[i], columns=["FPR_medicaid"])
                FPR_insu = pd.concat([FPR_insu, FPR_2.reindex(FPR_insu.index)],
                                     axis=1)

                Gap2 = pd.DataFrame(GAP_total[i], columns=["Gap_medicaid"])
                FPR_insu = pd.concat(
                    [FPR_insu, Gap2.reindex(FPR_insu.index)], axis=1)

                FPR_insu.to_csv(FPR_gaps_results_path+"Run_seed" +
                                str(seed)+"_FPR_GAP_insurance.csv", index=False)


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
    height = utility_variables['height']
    font_size = utility_variables['font_size']
    rotation_degree = utility_variables['rotation_degree']

    factor = [gender, age_decile, race, insurance]
    factor_str = ['gender', 'age_decile', 'race', 'insurance']
    seeds = get_seeds()

    FPR_base_path = "./FPR_GAPS/"
    # Create directory FPR saving
    os.makedirs(os.path.dirname(FPR_base_path), exist_ok=True)

    for seed in seeds:

        np.random.seed(seed)
        python_random.seed(seed)

        base_path = "./Prediction_results/"

        df = pd.read_csv(f"{base_path}bipred_{seed}.csv").rename(
            columns={'age decile': 'age_decile'})

        ''' FPR Disparities '''

        for i in range(len(factor)):
            FPR_GAPs(
                df, diseases, factor[i], factor_str[i], seed, FPR_base_path)

    print(f'SEED : {seed}')

    # FPR Disparities for SEX
    get_Sex_FPR_Disparities(FPR_base_path, diseases, diseases_abbr,
                            number_of_runs, significance_level, height, font_size, rotation_degree)

    # FPR Disparities for INSURANCE
    get_Insurance_FPR_Disparities(FPR_base_path, diseases, diseases_abbr,
                                  number_of_runs, significance_level, height, font_size, rotation_degree)

    # FPR Disparities for RACE
    get_Race_FPR_Disparities(FPR_base_path, diseases, diseases_abbr,
                             number_of_runs, significance_level, height, font_size, rotation_degree)

    # FPR Disparities for AGE
    get_Age_FPR_Disparities(FPR_base_path, diseases, diseases_abbr,
                            number_of_runs, significance_level, height, font_size, rotation_degree)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
