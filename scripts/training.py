from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


def trainModel(X_train, y_train, modelFit):
    if (modelFit == "rf"):
        
        param_grid = {
        'bootstrap': [True],
        'max_depth': [80, 90, 100, 110],
        'max_features': [2, 3],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [100, 200, 300, 1000]
        }

        # define model
        model = RandomForestRegressor()

        grid_search = GridSearchCV(estimator = model, param_grid = param_grid, 
                                cv = 3, n_jobs = -1, verbose = 2)

        grid_search.fit(X_train, y_train)
        best_grid = grid_search.best_estimator_
    return best_grid

