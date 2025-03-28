
def get_seeds():
    return [19, 31, 38, 47, 77]


def get_patient_groups():
    groups = {
        "sex": ['M', 'F'],
        "age": ['60-80', '40-60', '20-40', '80+', '0-20'],
        "race": ['WHITE', 'BLACK/AFRICAN AMERICAN',
                 'HISPANIC/LATINO', 'OTHER', 'ASIAN',
                 'AMERICAN INDIAN/ALASKA NATIVE'],
        "insurance": ['Medicare', 'Other', 'Medicaid']
    }

    return groups


def get_diseases():
    return ['Enlarged Cardiomediastinum',
            'Cardiomegaly',
            'Lung Opacity',
            'Lung Lesion',
            'Edema',
            'Consolidation',
            'Pneumonia',
            'Atelectasis',
            'Pneumothorax',
            'Pleural Effusion',
            'Pleural Other',
            'Fracture',
            'Support Devices',
            'No Finding']


def get_diseases_abbr():

    return {'Cardiomegaly': 'Cardiomegaly',
            'Effusion': 'Effusion',
            'Enlarged Cardiomediastinum': 'Enlarged Card.',
            'Lung Lesion': 'Lung Lesion',
            'Atelectasis': 'Atelectasis',
            'Pneumonia': 'Pneumonia',
            'Pneumothorax': 'Pneumothorax',
            'Consolidation': 'Consolidation',
            'Edema': 'Edema',
            'Pleural Effusion': 'Effusion',
            'Pleural Other': 'Pleural Other',
            'Fracture': 'Fracture',
            'Support Devices': 'Sup. Devices',
            'Lung Opacity': 'Air. Opacity',
            'No Finding': 'No Finding'
            }


def get_utility_variables():
    return {'number_of_runs': 5,
            'significance_level': 1.96,
            'height': 6,
            'font_size': 11,
            'rotation_degree': 15
            }
