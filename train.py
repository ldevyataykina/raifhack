import argparse
import logging.config
import pandas as pd
from traceback import format_exc
import geopy.distance

from raifhack.model import BenchmarkModel
from raifhack.settings import MODEL_PARAMS, LOGGING_CONFIG, NUM_FEATURES, CATEGORICAL_OHE_FEATURES, \
    CATEGORICAL_STE_FEATURES, TARGET, CENTER_MSK_LAT, CENTER_MSK_LNG, ENCODED_NUM_CAT_FEATURES
from raifhack.utils import PriceTypeEnum, UNKNOWN_VALUE
from raifhack.metrics import metrics_stat
from raifhack.features import prepare_categorical, get_number_floors, normalize_floor, is_specific_floor

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def parse_args():

    parser = argparse.ArgumentParser(
        description="""
    Бенчмарк для хакатона по предсказанию стоимости коммерческой недвижимости от "Райффайзенбанк"
    Скрипт для обучения модели
     
     Примеры:
        1) с poetry - poetry run python3 train.py --train_data /path/to/train/data --model_path /path/to/model
        2) без poetry - python3 train.py --train_data /path/to/train/data --model_path /path/to/model
    """,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--train_data", "-d", type=str, dest="d", required=True, help="Путь до обучающего датасета")
    parser.add_argument("--model_path", "-mp", type=str, dest="mp", required=True, help="Куда сохранить обученную ML модель")

    return parser.parse_args()

if __name__ == "__main__":

    try:
        logger.info('START train.py')
        args = vars(parse_args())
        logger.info('Load train df')
        train_df = pd.read_csv(args['d'])
        logger.info(f'Input shape: {train_df.shape}')
        # Sort values
        train_df = train_df.sort_values('date', ascending=True)
        train_df = prepare_categorical(train_df)

        # Add new features
        train_df['n_floors'] = train_df['floor'].apply(lambda x: get_number_floors(x))
        train_df['norm_floor'] = train_df['floor'].apply(lambda x: normalize_floor(x))
        train_df['specific_floor'] = train_df['norm_floor'].apply(lambda x: is_specific_floor(x))
        train_df['low_floor'] = train_df['norm_floor'].apply(lambda x: 1 if x.startswith('-') else 0)
        train_df['basement'] = train_df['norm_floor'].apply(lambda x: 1 if 'подвал' in x else 0)
        train_df['basement1'] = train_df['norm_floor'].apply(lambda x: 1 if 'цоколь' in x else 0)
        train_df['distance_from_moscow_center'] = train_df.apply(
            lambda x: geopy.distance.distance((x['lat'], x['lng']), (CENTER_MSK_LAT, CENTER_MSK_LNG)).km, axis=1)
        
        # create future columns with 0
        for en_num in ENCODED_NUM_CAT_FEATURES:
            train_df[en_num] = 0
        
        # Data preparation
        X_offer = train_df[train_df.price_type == PriceTypeEnum.OFFER_PRICE][NUM_FEATURES+CATEGORICAL_OHE_FEATURES+CATEGORICAL_STE_FEATURES]
        y_offer = train_df[train_df.price_type == PriceTypeEnum.OFFER_PRICE][TARGET]
        X_manual = train_df[train_df.price_type == PriceTypeEnum.MANUAL_PRICE][NUM_FEATURES+CATEGORICAL_OHE_FEATURES+CATEGORICAL_STE_FEATURES]
        y_manual = train_df[train_df.price_type == PriceTypeEnum.MANUAL_PRICE][TARGET]
        logger.info(f'X_offer {X_offer.shape}  y_offer {y_offer.shape}\tX_manual {X_manual.shape} y_manual {y_manual.shape}')
        model = BenchmarkModel(
            numerical_features=NUM_FEATURES,
            ohe_categorical_features=CATEGORICAL_OHE_FEATURES,
            ste_categorical_features=CATEGORICAL_STE_FEATURES,
            model_params=MODEL_PARAMS,
        )
        logger.info('Fit model')
        model.fit(X_offer, y_offer, X_manual, y_manual)
        logger.info('Save model')
        model.save(args['mp'])

        predictions_offer = model.predict(X_offer)
        metrics = metrics_stat(y_offer.values, predictions_offer / (1 + model.corr_coef)) # для обучающей выборки с ценами из объявлений смотрим качество без коэффициента
        logger.info(f'Metrics stat for training data with offers prices: {metrics}')

        predictions_manual = model.predict(X_manual)
        metrics = metrics_stat(y_manual.values, predictions_manual)
        logger.info(f'Metrics stat for training data with manual prices: {metrics}')


    except Exception as e:
        err = format_exc()
        logger.error(err)
        raise(e)
    logger.info('END train.py')