import typing
import pickle
import pandas as pd
import numpy as np
import logging

from lightgbm import LGBMRegressor

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.exceptions import NotFittedError
from raifhack.data_transformers import SmoothedTargetEncoding
from raifhack.settings import NUM_FEATURES, CATEGORICAL_STE_FEATURES

logger = logging.getLogger(__name__)


class MyLGBMRegressor():
    def __init__(self, **model_params):
        self.lgbm_model = LGBMRegressor(**model_params)
        self.sorted_columns = None
        self._estimator_type = 'regressor'
    
    def transform(self):
        return self
    
    def fit(self, X, y=None):
        print('lgbm clf fit')
        X = pd.DataFrame(X)
        x_ = len(CATEGORICAL_STE_FEATURES)
        y_ = X.shape[1]
        #X.iloc[:, x_:y_] = X.iloc[:, x_:y_].astype('string')
        self.lgbm_model.fit(
            X,
            y,
            #feature_name=[i for i in range(y_)],
            #categorical_feature=[i for i in range(x_, y_, 1)],
            eval_metric='RMSE',
        )
        return self
        
    def predict(self, X):
        print('clf predict')
        return self.lgbm_model.predict(X)
    
    def predict_proba(self, X):
        return self.lgbm_model.predict(X)
    
    def get_params(self, *args, **kwargs):
        return self.lgbm_model.get_params()


class BenchmarkModel():
    """
    Модель представляет из себя sklearn pipeline. Пошаговый алгоритм:
      1) в качестве обучения выбираются все данные с price_type=0
      1) все фичи делятся на три типа (numerical_features, ohe_categorical_features, ste_categorical_features):
          1.1) numerical_features - применяется StandardScaler
          1.2) ohe_categorical_featires - кодируются через one hot encoding
          1.3) ste_categorical_features - кодируются через SmoothedTargetEncoder
      2) после этого все полученные фичи конкатенируются в одно пространство фичей и подаются на вход модели Lightgbm
      3) делаем предикт на данных с price_type=1, считаем среднее отклонение реальных значений от предикта. Вычитаем это отклонение на финальном шаге (чтобы сместить отклонение к 0)
    :param numerical_features: list, список численных признаков из датафрейма
    :param ohe_categorical_features: list, список категориальных признаков для one hot encoding
    :param ste_categorical_features, list, список категориальных признаков для smoothed target encoding.
                                     Можно кодировать сразу несколько полей (например объединять категориальные признаки)
    :
    """

    def __init__(self, numerical_features: typing.List[str],
                 ohe_categorical_features: typing.List[str],
                 ste_categorical_features: typing.List[typing.Union[str, typing.List[str]]],
                 model_params: typing.Dict[str, typing.Union[str,int,float]]):
        self.num_features = numerical_features
        self.ohe_cat_features = ohe_categorical_features
        self.ste_cat_features = ste_categorical_features

        self.preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), self.num_features),
            ('ohe', OneHotEncoder(), self.ohe_cat_features),
            ('ste', OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1),
             self.ste_cat_features)])

        self.model = MyLGBMRegressor(**model_params)
        self.smoothed_target_encoder = SmoothedTargetEncoding(self.ste_cat_features)

        self.pipeline = Pipeline(steps=[
            ('smoothed_target_encoder', self.smoothed_target_encoder),
            ('preprocessor', self.preprocessor),
            ('model', self.model)])

        self._is_fitted = False
        self.corr_coef = 0

    def _find_corr_coefficient(self, X_manual: pd.DataFrame, y_manual: pd.Series):
        """Вычисление корректирующего коэффициента
        :param X_manual: pd.DataFrame с ручными оценками
        :param y_manual: pd.Series - цены ручника
        """
        predictions = self.pipeline.predict(X_manual)
        deviation = ((y_manual - predictions)/predictions).median()
        self.corr_coef = deviation

    def fit(
        self,
        X_offer: pd.DataFrame,
        y_offer: pd.Series,
        X_manual: pd.DataFrame, 
        y_manual: pd.Series,
    ):
        """Обучение модели.
        ML модель обучается на данных по предложениям на рынке (цены из объявления)
        Затем вычисляется среднее отклонение между руяными оценками и предиктами для корректировки стоимости
        :param X_offer: pd.DataFrame с объявлениями
        :param y_offer: pd.Series - цена предложения (в объявлениях)
        :param X_manual: pd.DataFrame с ручными оценками
        :param y_manual: pd.Series - цены ручника
        """
        logger.info('Fit lightgbm')
        self.pipeline.fit(
            X_offer,
            y_offer,
            #model__feature_name=[i for i in range(len(X_offer.columns))],
            #model__categorical_feature=[i for i in range(len(self.ste_cat_features), len(X_offer.columns))],
        )
        #self.pipeline.fit(X_offer, y_offer)
        logger.info('Find corr coefficient')
        self._find_corr_coefficient(X_manual, y_manual)
        logger.info(f'Corr coef: {self.corr_coef:.2f}')
        self.__is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.array:
        """Предсказание модели Предсказываем преобразованный таргет, затем конвертируем в обычную цену через обратное
        преобразование.
        :param X: pd.DataFrame
        :return: np.array, предсказания (цены на коммерческую недвижимость)
        """
        if self.__is_fitted:
            predictions = self.pipeline.predict(X)
            corrected_price = predictions * (1 + self.corr_coef)
            return corrected_price
        else:
            raise NotFittedError(
                "This {} instance is not fitted yet! Call 'fit' with appropriate arguments before predict".format(
                    type(self).__name__
                )
            )

    def save(self, path: str):
        """Сериализует модель в pickle.
        :param path: str, путь до файла
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(self, path: str):
        """Сериализует модель в pickle.
        :param path: str, путь до файла
        :return: Модель
        """
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model