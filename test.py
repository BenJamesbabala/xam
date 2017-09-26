import pandas as pd


class Model():

    def __init__(self, preprocessing_pipeline, learning_pipeline):
        self.preprocessing_pipeline = preprocessing_pipeline
        self.learning_pipeline = learning_pipeline

    def run(self, X_fit, X_val, X_test, y_fit, y_val):
        # Apply the preprocessing pipeline to all of the features
        X = pd.concat([X_fit, X_val, X_test], axis='rows')
        X = self.preprocessing_pipeline.fit_transform(X)
        # Split X back into training and test sets
        X_fit = X[:len(X_fit)]
        X_val = X[len(X_fit):-len(X_test)]
        X_test = X[-len(X_test):]
        # Train on the fit set
        self.learning_pipeline.fit(X_fit, y_fit)



import re

import pandas as pd
from sklearn import linear_model
from sklearn import model_selection
from sklearn import pipeline
from sklearn import preprocessing
import xam

train = pd.read_csv('datasets/titanic/train.csv')
test = pd.read_csv('datasets/titanic/test.csv')

X_train = train.drop(['PassengerId', 'Survived'], axis='columns')
y_train = train[['PassengerId', 'Survived']]
X_test = test.drop('PassengerId', axis='columns')

# Create a validation set with 20% of the training set
X_fit, X_val, y_fit, y_val = model_selection.train_test_split(X_train, y_train, test_size=0.2)

preprocessing_pipeline = pipeline.Pipeline([
    ('features', pipeline.FeatureUnion([
        ('cabin_letter', pipeline.Pipeline([
            ('extract', xam.preprocessing.ColumnSelector('Cabin')),
            ('transform', xam.preprocessing.SeriesTransformer(lambda x: x[0] if isinstance(x, str) else '?')),
            ('binarize', preprocessing.LabelBinarizer()),
            ('dataframe', xam.preprocessing.ToDataFrameTransformer())
        ])),
        ('title', pipeline.Pipeline([
            ('extract', xam.preprocessing.ColumnSelector('Name')),
            ('transform', xam.preprocessing.SeriesTransformer(lambda x: re.findall(r'\w+,\s(\w+)', x)[0])),
            ('binarize', preprocessing.LabelBinarizer()),
            ('dataframe', xam.preprocessing.ToDataFrameTransformer())
        ]))
    ])),
])

learning_pipeline = pipeline.Pipeline([
    ('lr', linear_model.LogisticRegression())
])

model = Model(preprocessing_pipeline, learning_pipeline)
model.run(X_fit, X_val, X_test, y_fit, y_val)
