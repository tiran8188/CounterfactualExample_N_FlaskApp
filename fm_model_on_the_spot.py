import numpy as np
import pandas as pd
import pickle
from counterfactual import ClassifierModel


import numpy as np
import pandas as pd


from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split,KFold
from sklearn.linear_model import LogisticRegression
from copy import deepcopy, copy
from imblearn.under_sampling import RandomUnderSampler

pd.options.display.max_columns = 30
pd.options.display.max_rows = 500

PATH = './production/'
# if the predicted probability is >= this threshold, the classifier will return
# a 1, else 0
CF_THRESHOLD = 0.2

STATES = [
    'PR', 'LA', 'GA', 'MN', 'TN', 'CO', 'DC', 'AL', 'WI', 'CA', 'MI',
    'FL', 'NY', 'others'
]
PROP_TYPES = [
    'Planned unit development', 'Manufactured housing', 'Condo', 'Co-op',
    '1-4 Fee simple'
]
PURPOSES = [
    'Cash-out refinance',
    'No cash-out refinance',
    'Purchase'
]
MCAT_CATS = {
    9 : STATES,
    7 : PROP_TYPES,
    8 : PURPOSES
}


def process_inputs(arr_x, ohc, mcat_columns):
    '''
    Encode dti into 2 variables, and do one hot encoding for the multicat
    variables
    :param arr_x: numpy array (num_raw_features,) datapoint to be processed
    :return: numpy array (num_input_vars,) array that can be fed into
       the model for training or prediction
    '''
    processed = copy(arr_x)
    # index 2 is dtl, or debt-to-loan ratio, in percentages.
    processed = np.hstack(
        (processed, ((processed[:, 2] > 65) * 1).reshape(-1, 1))
    )
    # in the original dataset, anything above 65 are recorded as not
    # applicable and encoded as 999
    processed[:, 2] = (np.apply_along_axis(
        (lambda x: (x != 999) * x), axis=0, arr=processed[:, 2]
    ))
    # one hot encoding for the multicategorical features
    processed = np.hstack((
        processed, ohc.transform(processed[:, mcat_columns])
    ))
    processed = np.delete(processed, [mcat_columns], axis=1)
    return processed

df = pd.read_csv(PATH + 'freddie_mac_cleaned.txt', sep='|')
df = df.set_index('loan_id')
df_features_attrs = pd.read_csv(
    PATH + 'feature_attrs_freddie_mac.txt', sep='\t',
    index_col='feature_index'
)
name_to_i = (
    df_features_attrs.reset_index().set_index('name')['feature_index']
    .to_dict()
)
i_to_name = df_features_attrs['name'].to_dict()
mcat_columns = (
    list(df_features_attrs[df_features_attrs.feature_type == 'mcat'].index)
)

df_x = df.drop('delq', axis=1)
df_y = df[['delq']]

rus = RandomUnderSampler(random_state=42)
x_us, y_us = rus.fit_resample(df_x, df_y)



ohc = preprocessing.OneHotEncoder(sparse=False, dtype=int, categories='auto')
ohc.fit(x_us[:, mcat_columns])

x_us = process_inputs(x_us, ohc, mcat_columns)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
rf = RandomForestClassifier(
    random_state=42, class_weight='balanced', n_estimators=10
)
lr = LogisticRegression(
    class_weight='balanced', random_state=42, solver='lbfgs', max_iter=10000
)
clf = VotingClassifier([('Logistic', lr), ('Forest', rf)], voting='soft')

scores = []
for train_index, test_index in kf.split(x_us):
    clf.fit(
        x_us[train_index],
        y_us[train_index].reshape(-1, )
    )
    proba = clf.predict_proba(x_us[test_index])
    scores.append(
        metrics.roc_auc_score(y_us[test_index].reshape(-1, ), proba[:, 1])
    )
print(f'Model trained. AUC score is {np.mean(scores)}')

class FreddieMacModel(ClassifierModel):
    '''
    A subclass just to hold a specific model. The model is a classifier model
    that predicts the probability of a home mortgage loan being delinquent for
    two or more months at least once in it's life time. The data came from
    Freddie Mac, and the classifier is an ensemble of logistic regression and
    random forest, with soft voting.

    The class contains the classifier object,one-hot encoder, feature attributes
    and the method to process input features before feeding them into the
    classifier for prediction.
    '''

    def __init__(self):

        # it's all pickled
        # dict of lists of possible values for multicat features
        self.mcat_cats = MCAT_CATS
        

        # feature attributes
        self.df_features_attrs = df_features_attrs

        # the one hot encoder
        self.ohc = ohc

        # the classifier object
        self.clf = clf

        self.mcat_columns = (list(
            self.df_features_attrs[
                self.df_features_attrs.feature_type == 'mcat'
            ].index
        ))

        self.decision_threshold = CF_THRESHOLD
        # dicts to translate to and from feature_index to feature name
        self.i_to_name = i_to_name
        self.name_to_i = name_to_i


    def __repr__(self):
        return f'Model to predict deliquency from the Freddie Mac data.'

    def process_inputs(self, arr_x):
        '''
        Encode dti into 2 variables, and do one hot encoding for the multicat
        variables
        :param arr_x: numpy array (num_raw_features,) datapoint to be processed
        :return: numpy array (num_input_vars,) array that can be fed into
           the model for training or prediction
        '''
        processed = copy(arr_x)
        # index 2 is dtl, or debt-to-loan ratio, in percentages.
        processed = np.hstack(
            (processed, ((processed[:, 2] > 65) * 1).reshape(-1, 1))
        )
        # in the original dataset, anything above 65 are recorded as not
        # applicable and encoded as 999
        processed[:, 2] = (np.apply_along_axis(
            (lambda x: (x != 999) * x), axis=0, arr=processed[:, 2]
        ))
        # one hot encoding for the multicategorical features
        processed = np.hstack((
            processed, self.ohc.transform(processed[:, self.mcat_columns])
        ))
        processed = np.delete(processed, [self.mcat_columns], axis=1)
        return processed
