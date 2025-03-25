from config_MIMIC import get_diseases, get_diseases_abbr, get_seeds, get_patient_groups

import pandas as pd
import numpy as np
import math
import random as python_random
import io
import os
import glob
from IPython.display import clear_output
import warnings


def fnr(df, d, c, category_name):

    pred_disease = "bi_" + d

    gt_fn = df.loc[(df[d] == 1) & (df[category_name] == c), :]
    pred_fn = df.loc[(df[pred_disease] == 0) & (
        df[d] == 1) & (df[category_name] == c), :]

    if len(gt_fn) != 0:
        FNR = len(pred_fn) / len(gt_fn)
        return FNR

    else:
        return np.NAN


def FNR_GAPs(df, diseases, category, category_name, seed=19, fnr_gaps_results_path="default"):

    GAP_total = []
    img_count_total = []
    FN_total = []

    cate = []

    if category_name == 'gender':
        FNR_Sex = pd.DataFrame(diseases, columns=["diseases"])

    if category_name == 'age_decile':
        FNR_Age = pd.DataFrame(diseases, columns=["diseases"])

    if category_name == 'race':
        FNR_Race = pd.DataFrame(diseases, columns=["diseases"])

    if category_name == 'insurance':
        FNR_insu = pd.DataFrame(diseases, columns=["diseases"])

    for c in category:

        img_count_cate = []
        FN_cate = []
        GAP_y = []

        for d in diseases:
            pred_disease = "bi_" + d

            pred_fn = df.loc[(df[pred_disease] == 0) & (
                df[d] == 1) & (df[category_name] == c), :]

            gt_fn = df.loc[(df[d] == 1) & (df[category_name] == c), :]

            n_gt_fn = df.loc[(df[d] == 1) & (df[category_name] != c)
                             & (df[category_name] != 0), :]

            n_pred_fn = df.loc[(df[pred_disease] == 0) & (df[d] == 1) & (
                df[category_name] != c) & (df[category_name] != 0), :]

            if len(gt_fn) != 0:

                FNR = len(pred_fn) / len(gt_fn)
                n_FNR = len(n_pred_fn) / len(n_gt_fn)

                FN_cate.append(round(FNR, 3))
                img_count_cate.append(round(len(gt_fn), 3))

                if category_name != 'gender':

                    temp_FNR = []
                    for c1 in category:

                        _fnr = fnr(df, d, c1, category_name)

                        if _fnr != -1:
                            temp_FNR.append(_fnr)

                    temp_FNR_Filtered = [
                        x for x in temp_FNR if not math.isnan(x)]
                    temp_FNR_Filtered.sort()

                    if len(temp_FNR_Filtered) % 2 == 0:

                        median = (temp_FNR_Filtered[(len(temp_FNR_Filtered) // 2) - 1]
                                  + temp_FNR_Filtered[(len(temp_FNR_Filtered) // 2)])/2

                    else:
                        median = temp_FNR_Filtered[(
                            len(temp_FNR_Filtered) // 2)]

                    GAP = FNR - median

                else:
                    GAP = FNR - n_FNR

                GAP_y.append(GAP)

            else:
                FN_cate.append(np.NaN)
                GAP_y.append(np.NAN)

        # FNR of all 14 diseases and categories
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

        if category_name == 'age_decile':

            if i == 0:
                img_count_0 = pd.DataFrame(
                    img_count_total[i], columns=["#60-80"])
                FNR_Age = pd.concat([FNR_Age, img_count_0.reindex(FNR_Age.index)],
                                    axis=1)

                fnr_0 = pd.DataFrame(FN_total[i], columns=["FNR_60-80"])
                FNR_Age = pd.concat([FNR_Age, fnr_0.reindex(FNR_Age.index)],
                                    axis=1)

                Gap0 = pd.DataFrame(GAP_total[i], columns=["Gap_60-80"])
                FNR_Age = pd.concat(
                    [FNR_Age, Gap0.reindex(FNR_Age.index)], axis=1)

            if i == 1:
                img_count_1 = pd.DataFrame(
                    img_count_total[i], columns=["#40-60"])
                FNR_Age = pd.concat([FNR_Age, img_count_1.reindex(FNR_Age.index)],
                                    axis=1)

                fnr_1 = pd.DataFrame(FN_total[i], columns=["FNR_40-60"])
                FNR_Age = pd.concat([FNR_Age, fnr_1.reindex(FNR_Age.index)],
                                    axis=1)

                Gap1 = pd.DataFrame(GAP_total[i], columns=["Gap_40-60"])
                FNR_Age = pd.concat(
                    [FNR_Age, Gap1.reindex(FNR_Age.index)], axis=1)

            if i == 2:
                img_count_2 = pd.DataFrame(
                    img_count_total[i], columns=["#20-40"])
                FNR_Age = pd.concat([FNR_Age, img_count_2.reindex(FNR_Age.index)],
                                    axis=1)

                fnr_2 = pd.DataFrame(FN_total[i], columns=["FNR_20-40"])
                FNR_Age = pd.concat([FNR_Age, fnr_2.reindex(FNR_Age.index)],
                                    axis=1)

                Gap2 = pd.DataFrame(GAP_total[i], columns=["Gap_20-40"])
                FNR_Age = pd.concat(
                    [FNR_Age, Gap2.reindex(FNR_Age.index)], axis=1)

            if i == 3:

                img_count_3 = pd.DataFrame(
                    img_count_total[i], columns=["#80-"])
                FNR_Age = pd.concat([FNR_Age, img_count_3.reindex(FNR_Age.index)],
                                    axis=1)

                fnr_3 = pd.DataFrame(FN_total[i], columns=["FNR_80-"])
                FNR_Age = pd.concat([FNR_Age, fnr_3.reindex(FNR_Age.index)],
                                    axis=1)

                Gap3 = pd.DataFrame(GAP_total[i], columns=["Gap_80-"])
                FNR_Age = pd.concat(
                    [FNR_Age, Gap3.reindex(FNR_Age.index)], axis=1)

            if i == 4:
                img_count_4 = pd.DataFrame(
                    img_count_total[i], columns=["#0-20"])
                FNR_Age = pd.concat([FNR_Age, img_count_4.reindex(FNR_Age.index)],
                                    axis=1)

                fnr_4 = pd.DataFrame(FN_total[i], columns=["FNR_0-20"])
                FNR_Age = pd.concat([FNR_Age, fnr_4.reindex(FNR_Age.index)],
                                    axis=1)

                Gap4 = pd.DataFrame(GAP_total[i], columns=["Gap_0-20"])
                FNR_Age = pd.concat(
                    [FNR_Age, Gap4.reindex(FNR_Age.index)], axis=1)

                FNR_Age.to_csv(fnr_gaps_results_path+"Run_seed" +
                               str(seed)+"_FNR_GAP_Age.csv", index=False)

        if category_name == 'gender':

            if i == 0:

                img_count_0 = pd.DataFrame(img_count_total[i], columns=["#M"])
                FNR_Sex = pd.concat([FNR_Sex, img_count_0.reindex(FNR_Sex.index)],
                                    axis=1)

                fnr_0 = pd.DataFrame(FN_total[i], columns=["FNR_M"])
                FNR_Sex = pd.concat([FNR_Sex, fnr_0.reindex(FNR_Sex.index)],
                                    axis=1)

                Gap0 = pd.DataFrame(GAP_total[i], columns=["Gap_M"])
                FNR_Sex = pd.concat(
                    [FNR_Sex, Gap0.reindex(FNR_Sex.index)], axis=1)

            else:
                img_count_1 = pd.DataFrame(img_count_total[i], columns=["#F"])
                FNR_Sex = pd.concat([FNR_Sex, img_count_1.reindex(FNR_Sex.index)],
                                    axis=1)

                fnr_1 = pd.DataFrame(FN_total[i], columns=["FNR_F"])
                FNR_Sex = pd.concat([FNR_Sex, fnr_1.reindex(FNR_Sex.index)],
                                    axis=1)

                Gap1 = pd.DataFrame(GAP_total[i], columns=["Gap_F"])
                FNR_Sex = pd.concat(
                    [FNR_Sex, Gap1.reindex(FNR_Sex.index)], axis=1)

                FNR_Sex.to_csv(fnr_gaps_results_path+"Run_seed" +
                               str(seed)+"_FNR_GAP_sex.csv", index=False)

        if category_name == 'race':

            if i == 0:
                img_count_0 = pd.DataFrame(
                    img_count_total[i], columns=["#White"])
                FNR_Race = pd.concat([FNR_Race, img_count_0.reindex(FNR_Race.index)],
                                     axis=1)

                fnr_0 = pd.DataFrame(FN_total[i], columns=["FNR_White"])
                FNR_Race = pd.concat([FNR_Race, fnr_0.reindex(FNR_Race.index)],
                                     axis=1)

                Gap0 = pd.DataFrame(GAP_total[i], columns=["Gap_White"])
                FNR_Race = pd.concat(
                    [FNR_Race, Gap0.reindex(FNR_Race.index)], axis=1)

            if i == 1:
                img_count_1 = pd.DataFrame(
                    img_count_total[i], columns=["#Black"])
                FNR_Race = pd.concat([FNR_Race, img_count_1.reindex(FNR_Race.index)],
                                     axis=1)

                fnr_1 = pd.DataFrame(FN_total[i], columns=["FNR_Black"])
                FNR_Race = pd.concat([FNR_Race, fnr_1.reindex(FNR_Race.index)],
                                     axis=1)

                Gap1 = pd.DataFrame(GAP_total[i], columns=["Gap_Black"])
                FNR_Race = pd.concat(
                    [FNR_Race, Gap1.reindex(FNR_Race.index)], axis=1)

            if i == 2:
                img_count_2 = pd.DataFrame(
                    img_count_total[i], columns=["#Hisp"])
                FNR_Race = pd.concat([FNR_Race, img_count_2.reindex(FNR_Race.index)],
                                     axis=1)

                fnr_2 = pd.DataFrame(FN_total[i], columns=["FNR_Hisp"])
                FNR_Race = pd.concat([FNR_Race, fnr_2.reindex(FNR_Race.index)],
                                     axis=1)

                Gap2 = pd.DataFrame(GAP_total[i], columns=["Gap_Hisp"])
                FNR_Race = pd.concat(
                    [FNR_Race, Gap2.reindex(FNR_Race.index)], axis=1)

            if i == 3:
                img_count_3 = pd.DataFrame(
                    img_count_total[i], columns=["#Other"])
                FNR_Race = pd.concat([FNR_Race, img_count_3.reindex(FNR_Race.index)],
                                     axis=1)

                fnr_3 = pd.DataFrame(FN_total[i], columns=["FNR_Other"])
                FNR_Race = pd.concat([FNR_Race, fnr_3.reindex(FNR_Race.index)],
                                     axis=1)

                Gap3 = pd.DataFrame(GAP_total[i], columns=["Gap_Other"])
                FNR_Race = pd.concat(
                    [FNR_Race, Gap3.reindex(FNR_Race.index)], axis=1)

            if i == 4:

                img_count_4 = pd.DataFrame(
                    img_count_total[i], columns=["#Asian"])
                FNR_Race = pd.concat([FNR_Race, img_count_4.reindex(FNR_Race.index)],
                                     axis=1)

                fnr_4 = pd.DataFrame(FN_total[i], columns=["FNR_Asian"])
                FNR_Race = pd.concat([FNR_Race, fnr_4.reindex(FNR_Race.index)],
                                     axis=1)

                Gap4 = pd.DataFrame(GAP_total[i], columns=["Gap_Asian"])
                FNR_Race = pd.concat(
                    [FNR_Race, Gap4.reindex(FNR_Race.index)], axis=1)

            if i == 5:
                img_count_5 = pd.DataFrame(
                    img_count_total[i], columns=["#American"])
                FNR_Race = pd.concat([FNR_Race, img_count_5.reindex(FNR_Race.index)],
                                     axis=1)

                fnr_5 = pd.DataFrame(FN_total[i], columns=["FNR_American"])
                FNR_Race = pd.concat([FNR_Race, fnr_5.reindex(FNR_Race.index)],
                                     axis=1)

                Gap5 = pd.DataFrame(GAP_total[i], columns=["Gap_American"])
                FNR_Race = pd.concat(
                    [FNR_Race, Gap5.reindex(FNR_Race.index)], axis=1)

                FNR_Race.to_csv(fnr_gaps_results_path +
                                "Run_seed"+str(seed)+"_FNR_GAP_race.csv", index=False)

        if category_name == 'insurance':

            if i == 0:

                img_count_0 = pd.DataFrame(
                    img_count_total[i], columns=["#medicare"])
                FNR_insu = pd.concat([FNR_insu, img_count_0.reindex(FNR_insu.index)],
                                     axis=1)

                fnr_0 = pd.DataFrame(FN_total[i], columns=["FNR_medicare"])
                FNR_insu = pd.concat([FNR_insu, fnr_0.reindex(FNR_insu.index)],
                                     axis=1)

                Gap0 = pd.DataFrame(GAP_total[i], columns=["Gap_medicare"])
                FNR_insu = pd.concat(
                    [FNR_insu, Gap0.reindex(FNR_insu.index)], axis=1)

            if i == 1:
                img_count_1 = pd.DataFrame(
                    img_count_total[i], columns=["#other"])
                FNR_insu = pd.concat([FNR_insu, img_count_1.reindex(FNR_insu.index)],
                                     axis=1)

                fnr_1 = pd.DataFrame(FN_total[i], columns=["FNR_other"])
                FNR_insu = pd.concat([FNR_insu, fnr_1.reindex(FNR_insu.index)],
                                     axis=1)

                Gap1 = pd.DataFrame(GAP_total[i], columns=["Gap_other"])
                FNR_insu = pd.concat(
                    [FNR_insu, Gap1.reindex(FNR_insu.index)], axis=1)

            if i == 2:

                img_count_2 = pd.DataFrame(
                    img_count_total[i], columns=["#medicaid"])
                FNR_insu = pd.concat([FNR_insu, img_count_2.reindex(FNR_insu.index)],
                                     axis=1)

                fnr_2 = pd.DataFrame(FN_total[i], columns=["FNR_medicaid"])
                FNR_insu = pd.concat([FNR_insu, fnr_2.reindex(FNR_insu.index)],
                                     axis=1)

                Gap2 = pd.DataFrame(GAP_total[i], columns=["Gap_medicaid"])
                FNR_insu = pd.concat(
                    [FNR_insu, Gap2.reindex(FNR_insu.index)], axis=1)

                FNR_insu.to_csv(fnr_gaps_results_path+"Run_seed" +
                                str(seed)+"_FNR_GAP_insurance.csv", index=False)


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
