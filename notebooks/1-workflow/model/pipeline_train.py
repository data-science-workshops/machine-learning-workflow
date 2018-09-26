from os import path, environ as env
import numpy as np
import pandas as pd
import cloudpickle as pickle
from sklearn_pandas import DataFrameMapper, cross_val_score
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import Imputer, MinMaxScaler, Normalizer, StandardScaler, LabelBinarizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, precision_score, recall_score, accuracy_score, f1_score
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler

OUTPUT_DIR = env.get('PIPELINE_OUTPUT_PATH')


def get_meta_data(df):
    data = []
    for column in df.columns:
        # Defining the role
        if column == 'target':
            role = 'target'
        elif column == 'id':
            role = 'id'
        else:
            role = 'input'

        # Defining the level
        if 'bin' in column or column == 'target':
            level = 'binary'
        elif 'cat' in column or column == 'id':
            level = 'nominal'
        elif df[column].dtype == np.dtype('float64'):
            level = 'interval'
        elif df[column].dtype == np.dtype('int64'):
            level = 'ordinal'

        # Initialize keep to True for all variables except for id
        keep = True
        if column == 'id':
            keep = False

        # Defining the data type
        dtype = df[column].dtype

        # Creating a Dict that contains all the metadata for the variable
        column_dict = {
            'column_name': column,
            'role': role,
            'level': level,
            'keep': keep,
            'dtype': dtype
        }
        data.append(column_dict)

    df_meta = pd.DataFrame(data, columns=['column_name', 'role', 'level', 'keep', 'dtype'])
    df_meta.set_index('column_name', inplace=True)
    return df_meta


def binary_cols(df_meta):
    query = df_meta[(df_meta.level == 'binary') & (df_meta.keep) & (df_meta.index != 'target')].index
    return df[query].columns.values


def nominal_cols(df_meta):
    query = df_meta[(df_meta.level == 'nominal') & (df_meta.keep) & (df_meta.index != 'id')].index
    return df[query].columns.values


def interval_cols(df_meta):
    query = df_meta[(df_meta.level == 'interval') & (df_meta.keep)].index
    return df[query].columns.values


def ordinal_cols(df_meta):
    query = df_meta[(df_meta.level == 'ordinal') & (df_meta.keep)].index
    return df[query].columns.values


def get_pipeline(df_meta):
    return Pipeline([
        ('union', FeatureUnion([
            ('binary', Pipeline([
                ('impute', DataFrameMapper([
                    (binary_cols(df_meta), Imputer(missing_values=-1, strategy='most_frequent', axis=0))
                ], input_df=True))
            ])),

            ('nominal', Pipeline([
                ('label_binarize', DataFrameMapper(
                    [(c, LabelBinarizer()) for c in nominal_cols(df_meta)]
                    , input_df=True))
            ])),

            ('interval', Pipeline([
                ('impute', DataFrameMapper([
                    (interval_cols(df_meta), Imputer(missing_values=-1, strategy='mean', axis=0))
                ], input_df=True)),
                ('scaler', StandardScaler())
            ])),

            ('ordinal', Pipeline([
                ('impute', DataFrameMapper([
                    (ordinal_cols(df_meta), Imputer(missing_values=-1, strategy='most_frequent', axis=0))
                ], input_df=True)),
                ('scaler', MinMaxScaler(feature_range=(0, 1)))
            ])),
        ])),
        ('classify', DecisionTreeClassifier())
    ])


def get_param_grid():
    return [
        {
            'classify': [DecisionTreeClassifier(criterion='gini', class_weight=None)],
            'classify__criterion': ['gini', 'entropy'],
            'classify__class_weight': [None, 'balanced']
        },
        # {
        #     'classify': [RandomForestClassifier(n_estimators=10, criterion='gini', class_weight=None, n_jobs=-1)],
        #     'classify__n_estimators': [10, 50, 100],
        #     'classify__criterion': ['gini', 'entropy'],
        #     'classify__class_weight': [None, 'balanced'],
        #     'classify__warm_start': [False, True]
        # },
    ]


if __name__ == '__main__':

    # Load the dataset
    df = pd.read_csv('safe-driver-prediction.csv')

    df_meta = get_meta_data(df)

    # Define pipeline
    pipe = get_pipeline(df_meta)

    desired_apriori = 0.30

    nb_0 = len(df.loc[df.target == 0].index)
    nb_1 = len(df.loc[df.target == 1].index)

    undersampling_rate = ((1 - desired_apriori) * nb_1) / (nb_0 * desired_apriori)
    undersampled_nb_0 = int(undersampling_rate * nb_0)

    df_X = df.drop('target', axis=1)
    df_y = df['target']

    cc = RandomUnderSampler(ratio={0: undersampled_nb_0})
    X_cc, y_cc = cc.fit_sample(df_X, df_y.ravel())

    df_X = pd.DataFrame(X_cc, columns=df_X.columns)
    df_y = pd.DataFrame(y_cc, columns=['target'])

    scoring = {  # weighted, binary, None
        'precision_score': make_scorer(precision_score, average='binary'),
        'recall_score': make_scorer(recall_score, average='binary'),
        'f1_score': make_scorer(f1_score, average='binary'),
        'accuracy_score': make_scorer(accuracy_score)
    }

    refit_score = 'f1_score'
    skf = StratifiedKFold(n_splits=2)
    param_grid = get_param_grid()

    grid = GridSearchCV(pipe, cv=skf, param_grid=param_grid, scoring=scoring, refit=refit_score,
                        return_train_score=True, n_jobs=-1)
    grid.fit(df_X, df_y)

    model_pkl_path = path.join(OUTPUT_DIR, 'model.pkl')

    with open(model_pkl_path, 'wb') as fh:
        pickle.dump(grid.best_estimator_, fh)
        print('Pickled model to "%s"' % model_pkl_path)
