import os
import sys
from IPython.core import display as ICD
from tqdm.notebook import tqdm

# visualisation
import numpy as np
import pandas as pd

# performance
from sklearn.metrics import accuracy_score, recall_score

# fine-tuning
from sklearn.model_selection import GridSearchCV

# model zxport
import joblib

# warnings
import warnings
warnings.filterwarnings('ignore')


# --- MODEL BUILDING --- #

class model_train():

    def classification_models(models_list: list, X_train, y_train, X_test, y_test, only_models=False):
        """
        Function to instantiate a model or a series of models in a given list, 

        Parameters
        ----------
        models_list: list
            List containing a model or series of models to instantiate
        X_train
        y_train
        X_test
        y_test

        Returns
        Plot with models metrics, and a dictionary containing trained models.
        """
        classifiers = models_list
        # names = ["RBF SVM", "Random Forest", "Extremely Randomized Trees", "XGBoost"]
        names = range(len(classifiers))

        monit_ls = []
        models_trained = {}

        for name, clf in zip(names, classifiers):
            model = clf.fit(X_train, y_train)
            models_trained[name] = model
            y_predict = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_predict)
            recall = recall_score(y_test, y_predict)
            monit_df.append([name, accuracy, recall])

        if only_models:
            monit_df = pd.DataFrame(
                monit_ds, columns=['Fault Type', 'Model', 'Accuracy', 'F1 Score'])
            monit_df.style.background_gradient(cmap='Greens')

            ICD.display(monit_df)

        else:
            return models_trained

    def grid_search(model, prm_grid: dict, metric: str, X_train, y_train, export=False):
        """
        dezdzedzedze
        """
        model = GridSearchCV(model, prm_grid, scoring=metric,
                             verbose=4, n_jobs=-1, refit=True, return_train_score=True)
        model.fit(X_train, y_train)

        return model

    def export_model(model, path: str, name: str):
        joblib.dump(value=model, filename=f"{path}/{name}.pkl")
