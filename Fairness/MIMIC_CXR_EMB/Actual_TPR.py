import pandas as pd
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import random
from scipy.optimize import curve_fit



def TPR_14(df, diseases, category, category_name, seed=19, tpr_gaps_results_path_dir_dir="default"):
    GAP_total = []
    percentage_total = []
    Total_total = []
    Positive_total = []
    Negetive_total = []    
    
    
    cate = []

    # print(diseases)

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
        Total_y = []
        Positive_y = []
        Negetive_y = []
        for d in diseases:
            pred_disease = "bi_" + d
            gt = df.loc[(df[d] == 1) & (df[category_name] == c), :]
            pred = df.loc[(df[pred_disease] == 1) & (df[d] == 1) & (df[category_name] == c), :]
            n_gt = df.loc[(df[d] == 1) & (df[category_name] != c) & (df[category_name] != 0), :]
            n_pred = df.loc[
                     (df[pred_disease] == 1) & (df[d] == 1) & (df[category_name] != c) & (df[category_name] != 0), :]
            
        #within category (e.g Male) with positive for disease d    
            pi_gy = df.loc[(df[d] == 1) & (df[category_name] == c), :] 
        
        #within category (e.g Male) with negative for disease d
            Ne_gy = df.loc[(df[d] == 0) & (df[category_name] == c), :] 
        
        # All subgroups wihin a category that have disease (e.g male and female with disease d)
        #df[category_name] != 0 means all becouse we need to not to consider an image if we do not have the its meta-data
            pi_y = df.loc[(df[d] == 1) & (df[category_name] != 0), :]
            

            Total = len(pi_gy) + len(Ne_gy)
            Positive = len(pi_gy)
            Negetive = len(Ne_gy)
                        
            
            if len(gt) != 0 and len(n_gt) != 0 and len(pi_y) != 0:
                TPR = len(pred) / len(gt)
                percentage = len(pi_gy) / len(pi_y)
                GAP = TPR  # just to not to update parameter name later
                GAP_y.append(GAP)
                percentage_y.append(percentage)
                
            else:
                GAP_y.append(np.NaN)
                percentage_y.append(0)
                
            
            Total_y.append(Total)
            Positive_y.append(Positive)
            Negetive_y.append(Negetive) 

        # Gaps of all 14 diseases and categories
        GAP_total.append(GAP_y)
        percentage_total.append(percentage_y)
        Total_total.append(Total_y)
        Positive_total.append(Positive_y)
        Negetive_total.append(Negetive_y) 




    for i in range(len(GAP_total)):




        if category_name == 'age_decile':

            if i == 0:

                Gap4 = pd.DataFrame(GAP_total[i], columns=["TPR_60-80"])
                Run1_age = pd.concat([Run1_age, Gap4.reindex(Run1_age.index)], axis=1)

            if i == 1:

                Gap6 = pd.DataFrame(GAP_total[i], columns=["TPR_40-60"])
                Run1_age = pd.concat([Run1_age, Gap6.reindex(Run1_age.index)], axis=1)

            if i == 2:

                Gap2 = pd.DataFrame(GAP_total[i], columns=["TPR_20-40"])
                Run1_age = pd.concat([Run1_age, Gap2.reindex(Run1_age.index)], axis=1)

            if i == 3:
                
                Gap8 = pd.DataFrame(GAP_total[i], columns=["TPR_80-"])
                Run1_age = pd.concat([Run1_age, Gap8.reindex(Run1_age.index)], axis=1)

            if i == 4:
              
                Gap0 = pd.DataFrame(GAP_total[i], columns=["TPR_0-20"])
                Run1_age = pd.concat([Run1_age, Gap0.reindex(Run1_age.index)], axis=1)

            # Run1_age.to_csv("./results/Run1_TPR_Age.csv")
            Run1_age.to_csv(tpr_gaps_results_path_dir_dir +
                            "Run_seed"+str(seed)+"_TPR_Age.csv")

        if category_name == 'gender':

            if i == 0:
    
                MaleGap = pd.DataFrame(GAP_total[i], columns=["TPR_M"])
                Run1_sex = pd.concat([Run1_sex, MaleGap.reindex(Run1_sex.index)], axis=1)

            else:
                 
                FeMaleGap = pd.DataFrame(GAP_total[i], columns=["TPR_F"])
                Run1_sex = pd.concat([Run1_sex, FeMaleGap.reindex(Run1_sex.index)], axis=1)

            # Run1_sex.to_csv("./results/Run1_TPR_sex.csv")
            Run1_sex.to_csv(tpr_gaps_results_path_dir_dir +
                            "Run_seed"+str(seed)+"_TPR_sex.csv")

        if category_name == 'race':
            if i == 0:
                  
                WhGap = pd.DataFrame(GAP_total[i], columns=["TPR_White"])
                Run1_race = pd.concat([Run1_race, WhGap.reindex(Run1_race.index)], axis=1)

            if i == 1:
                BlGap = pd.DataFrame(GAP_total[i], columns=["TPR_Black"])
                Run1_race = pd.concat([Run1_race, BlGap.reindex(Run1_race.index)], axis=1)

            if i == 2:               
                
                BlGap = pd.DataFrame(GAP_total[i], columns=["TPR_Hisp"])
                Run1_race = pd.concat([Run1_race, BlGap.reindex(Run1_race.index)], axis=1)

            if i == 3:
   
                OtGap = pd.DataFrame(GAP_total[i], columns=["TPR_Other"])
                Run1_race = pd.concat([Run1_race, OtGap.reindex(Run1_race.index)], axis=1)

            if i == 4:
  
                AsGap = pd.DataFrame(GAP_total[i], columns=["Gap_Asian"])
                Run1_race = pd.concat([Run1_race, AsGap.reindex(Run1_race.index)], axis=1)

            if i == 5:
                 
                AmGap = pd.DataFrame(GAP_total[i], columns=["TPR_American"])
                Run1_race = pd.concat([Run1_race, AmGap.reindex(Run1_race.index)], axis=1)

            # Run1_race.to_csv("./results/Run1_TPR_race.csv")
            Run1_race.to_csv(tpr_gaps_results_path_dir_dir +
                             "Run_seed"+str(seed)+"_TPR_race.csv")

        if category_name == 'insurance':
            if i == 0:
                
                CareGap = pd.DataFrame(GAP_total[i], columns=["TPR_Medicare"])
                Run1_insurance = pd.concat([Run1_insurance, CareGap.reindex(Run1_insurance.index)], axis=1)

            if i == 1:
                 
                OtherGap = pd.DataFrame(GAP_total[i], columns=["TPR_Other"])
                Run1_insurance = pd.concat([Run1_insurance, OtherGap.reindex(Run1_insurance.index)], axis=1)

            if i == 2:
                                    
                AidGap = pd.DataFrame(GAP_total[i], columns=["TPR_Medicaid"])
                Run1_insurance = pd.concat([Run1_insurance, AidGap.reindex(Run1_insurance.index)], axis=1)

            # Run1_insurance.to_csv("./results/Run1_TPR_insurance.csv")
            
            Run1_insurance.to_csv(
                tpr_gaps_results_path_dir_dir+"Run_seed"+str(seed)+"_TPR_insurance.csv")


if __name__ == '__main__':
    
    diseases = ['Airspace Opacity', 'Atelectasis', 'Cardiomegaly',
           'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
           'Lung Lesion', 'No Finding', 'Pleural Effusion', 'Pleural Other',
           'Pneumonia', 'Pneumothorax', 'Support Devices']

    age_decile = ['60-80', '40-60', '20-40', '80-', '0-20']

    gender = ['M', 'F']
    race = ['WHITE', 'BLACK/AFRICAN AMERICAN',
            'HISPANIC/LATINO', 'OTHER', 'ASIAN',
            'AMERICAN INDIAN/ALASKA NATIVE']


    insurance = ['Medicare', 'Other', 'Medicaid']    
    
    TrueWithMeta = pd.read_csv("./True_withMeta.csv")
    pred = pd.read_csv("./results/bipred.csv")
    factor = [gender, age_decile, race, insurance]
    factor_str = ['gender', 'age_decile', 'race', 'insurance']
    
    for i in range(len(factor)):
        TPR_14(TrueWithMeta,pred, diseases, factor[i], factor_str[i]) 