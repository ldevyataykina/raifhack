import pandas as pd
from raifhack.settings import SPECIFIC_FLOORS
from raifhack.utils import UNKNOWN_VALUE


def prepare_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Заполняет пропущенные категориальные переменные
    :param df: dataframe, обучающая выборка
    :return: dataframe
    """
    df_new = df.copy()
    fillna_cols = ['region', 'city', 'street', 'realty_type']
    df_new[fillna_cols] = df_new[fillna_cols].fillna(UNKNOWN_VALUE)
    return df_new

def get_number_floors(floor):
    floor = str(floor)
    if floor == 'подвал, 1-3':
        n_floors = 4
    elif floor == '1-3':
        n_floors = 3
    elif floor == '1.2':
        n_floors = 2
    elif floor == '1 + подвал (без отделки)':
        n_floors = 2
    elif floor == 'подвал, 3. 4 этаж':
        n_floors = 3
    elif floor == 'подвал, 1-4 этаж':
        n_floors = 5
    elif floor == 'подвал, 1-7, техэтаж':
        n_floors = 9
    elif floor == 'подва, 1.2 этаж':
        n_floors = 3
    elif floor == '1-7':
        n_floors = 7
    elif floor == ' 1-2, подвальный':
        n_floors = 3
    elif floor == '1-3 этажи, цоколь (188,4 кв.м), подвал (104 кв.м)':
        n_floors = 5
    elif ',' in floor:
        n_floors = len(floor.split(','))
    else:
        n_floors = 1
    return n_floors

def normalize_floor(floor):
    floor = str(floor).strip()
    floor = floor.replace('.0', '')
    floor = floor.replace(' этаж', '')
    floor = floor.replace('-й', '')
    if floor == '1, подвал':
        floor = 'подвал, 1'
    elif floor == '1 + подвал (без отделки)':
        floor == 'подвал, 1'
    elif floor == 'подвал , 1':
        floor = 'подвал, 1'
    elif floor == 'подвал,1':
        floor = 'подвал, 1'
    elif floor == 'подва, 1.2':
        floor = 'подвал, 1.2'
    return floor

def is_specific_floor(floor):
    if any(s in floor for s in SPECIFIC_FLOORS):
        return 1
    else:
        return 0