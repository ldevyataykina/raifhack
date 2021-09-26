import argparse
import logging.config
import pandas as pd
from traceback import format_exc

import geopy.distance

from raif_hack.features import prepare_categorical, get_number_floors, normalize_floor, is_specific_floor, get_distance_to_reg_center
from raif_hack.model import BenchmarkModel
from raif_hack.settings import LOGGING_CONFIG, NUM_FEATURES, CATEGORICAL_OHE_FEATURES, \
    CATEGORICAL_STE_FEATURES, CENTER_MSK_LAT, CENTER_MSK_LNG

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="""
    Бенчмарк для хакатона по предсказанию стоимости коммерческой недвижимости от "Райффайзенбанк"
    Скрипт для предсказания модели
     
     Примеры:
        1) с poetry - poetry run python3 predict.py --test_data /path/to/test/data --model_path /path/to/model --output /path/to/output/file.csv.gzip
        2) без poetry - python3 predict.py --test_data /path/to/test/data --model_path /path/to/model --output /path/to/output/file.csv.gzip
    """,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--test_data", "-d", type=str, dest="d", required=True, help="Путь до отложенной выборки")
    parser.add_argument("--model_path", "-mp", type=str, dest="mp", required=True,
                        help="Пусть до сериализованной ML модели")
    parser.add_argument("--output", "-o", type=str, dest="o", required=True, help="Путь до выходного файла")

    return parser.parse_args()

if __name__ == "__main__":

    try:
        logger.info('START predict.py')
        args = vars(parse_args())
        logger.info('Load test df')
        test_df = pd.read_csv(args['d'])
        logger.info(f'Input shape: {test_df.shape}')
        test_df = test_df.sort_values('date', ascending=True)
        test_df = prepare_categorical(test_df)
        # Add new features
        test_df['n_floors'] = test_df['floor'].apply(lambda x: get_number_floors(x))
        test_df['norm_floor'] = test_df['floor'].apply(lambda x: normalize_floor(x))
        test_df['specific_floor'] = test_df['norm_floor'].apply(lambda x: is_specific_floor(x))
        test_df['low_floor'] = test_df['norm_floor'].apply(lambda x: 1 if x.startswith('-') else 0)
        test_df['basement'] = test_df['norm_floor'].apply(lambda x: 1 if 'подвал' in x else 0)
        test_df['basement1'] = test_df['norm_floor'].apply(lambda x: 1 if 'цоколь' in x else 0)
        test_df['distance_from_moscow_center'] = test_df.apply(
            lambda x: geopy.distance.distance((x['lat'], x['lng']), (CENTER_MSK_LAT, CENTER_MSK_LNG)).km, axis=1)
        #test_df['distance_from_reg_center'] = test_df.apply(lambda x: get_distance_to_reg_center(x), axis=1)

        logger.info('Load model')
        model = BenchmarkModel.load(args['mp'])
        logger.info('Predict')
        test_df['per_square_meter_price'] = model.predict(test_df[NUM_FEATURES+CATEGORICAL_OHE_FEATURES+CATEGORICAL_STE_FEATURES]) * 0.95
        test_df = test_df.sort_index()
        logger.info('Save results')
        test_df[['id','per_square_meter_price']].to_csv(args['o'], index=False)
    except Exception as e:
        err = format_exc()
        logger.error(err)
        raise (e)

    logger.info('END predict.py')