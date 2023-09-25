import numpy as np
import pandas as pd
from datetime import datetime, time
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import xgboost as xgb
from typing import Tuple, Union, List

warnings.filterwarnings('ignore')

MORNING_START = time(5)
MORNING_END = time(12)

AFTERNOON_START = time(12)
AFTERNOON_END = time(19)

NIGHT_START = time(19)
NIGHT_END = time(23, 59)

HIGH_SEASON_RANGES = [
    ('15-Dec', '31-Dec'),
    ('1-Jan', '3-Mar'),
    ('15-Jul', '31-Jul'),
    ('11-Sep', '30-Sep')
]

HIGH_SEASON = 1
LOW_SEASON = 0

DELAY_THRESHOLD = 15
DECISION_THRESHOLD = 0.49

TOP_FEATURES = 10

class DelayModel:

    def __init__(
        self
    ):
        self._model = None # Model should be saved in this attribute.
        self._top_features_data = None

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        data = data.copy()
        data['period_day'] = data['Fecha-I'].apply(self._determine_period_of_day)
        data['high_season'] = data['Fecha-I'].apply(self._determine_season)
        data['min_diff'] = data.apply(self._calculate_time_difference, axis = 1)
        data['delay'] = np.where(data['min_diff'] > DELAY_THRESHOLD, 1, 0)
        shuffle(data[['OPERA', 'MES', 'TIPOVUELO', 'SIGLADES', 'DIANOM', 'delay']], random_state = 111)
        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'), 
            pd.get_dummies(data['MES'], prefix = 'MES')], 
            axis = 1
        )
        x_train, x_test, y_train, y_test = train_test_split(features, data['delay'], test_size = 0.33, random_state = 42)
        self._top_features_data = self._select_top_features(x_train, y_train, TOP_FEATURES)
        if target_column:
            return features[self._top_features_data], pd.DataFrame(data[target_column])
        else:
            return features[self._top_features_data]

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        x_train2, x_test2, y_train2, y_test2 = train_test_split(features, target['delay'], test_size = 0.33, random_state = 42)
        n_y0 = len(y_train2[y_train2 == 0])
        n_y1 = len(y_train2[y_train2 == 1])
        class_weight_dict = {1: n_y0 / len(y_train2), 0: n_y1 / len(y_train2)}
        self._model = LogisticRegression(class_weight=class_weight_dict, C=0.07)
        self._model.fit(x_train2, y_train2)
        return

    def predict(self, features: pd.DataFrame) -> List[int]:
        if self._model is None or self._top_features_data is None:
            print("Model not trained. Returning list of zeros.")
            return [0]*features.shape[0]
        else:
            features = pd.concat([
                pd.get_dummies(features['OPERA'], prefix = 'OPERA'),
                pd.get_dummies(features['TIPOVUELO'], prefix = 'TIPOVUELO'), 
                pd.get_dummies(features['MES'], prefix = 'MES')], 
            axis = 1
            )
            instance = self.preprocess_test_data(features)
            prediction = self._model.predict(instance)
            return prediction

    def preprocess_test_data(self, features: pd.DataFrame) -> pd.DataFrame:
        flight_data = pd.DataFrame(columns = self._top_features_data, index = [0])
        for feature in self._top_features_data:
            if feature in features.columns:
                flight_data[feature] = True
            else:
                flight_data[feature] = False
        return flight_data

    def _determine_period_of_day(self, date_str: str) -> str:
        date_time = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S').time()
        
        if MORNING_START <= date_time < MORNING_END:
            return 'morning'
        elif AFTERNOON_START <= date_time < AFTERNOON_END:
            return 'afternoon'
        elif NIGHT_START <= date_time <= NIGHT_END or time(0) <= date_time <= time(4, 59):
            return 'night'
        else:
            pass

    def _determine_season(self, date_str: str) -> int:
        date_year = int(date_str.split('-')[0])
        date = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
        
        for date_range in HIGH_SEASON_RANGES:
            min_date = datetime.strptime(date_range[0], '%d-%b').replace(year = date_year)
            max_date = datetime.strptime(date_range[1], '%d-%b').replace(year = date_year)
            
            if min_date <= date <= max_date:
                return HIGH_SEASON

        return LOW_SEASON

    def _calculate_time_difference(self, data_row: pd.Series) -> float:
        departure_date = datetime.strptime(data_row['Fecha-O'], '%Y-%m-%d %H:%M:%S')
        arrival_date = datetime.strptime(data_row['Fecha-I'], '%Y-%m-%d %H:%M:%S')
        return ((departure_date - arrival_date).total_seconds()) / 60

    def _select_top_features(self, x_train: pd.DataFrame, y_train: pd.DataFrame, top_features: int) -> List[str]:
        xgb_model = xgb.XGBClassifier(random_state=1, learning_rate=0.01)
        xgb_model.fit(x_train, y_train)
        f_scores = xgb_model.get_booster().get_score()
        top_features_keys = [feature for feature, _ in sorted(f_scores.items(), key=lambda x: x[1], reverse=True)[:top_features]]
        return top_features_keys
    
