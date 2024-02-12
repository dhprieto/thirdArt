import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score, GridSearchCV
import re 

def fullRead(pathToTable, sep, anthro = False):

  df_renamed = pd.read_csv(pathToTable, sep = sep, encoding = "latin_1")
  
  df_name = re.sub("_ord.csv","",(re.sub("data/", "" ,pathToTable)))
  # reading and merging    
  
  if anthro == True:
    df_anthro = pd.read_csv("data/chronicAnthropometricCardiovascularData.csv", sep=";", decimal=",")
    df_renamed = df_renamed.merge(df_anthro)

    # separating by time moment and renaming

    df_renamed["Weight"] = ""
    df_renamed["BMI"] = ""
    df_renamed["Fat"] = ""
    df_renamed["CVRI"] = ""
    df_renamed["Bpmin"] = ""
    df_renamed["Bpmax"] = ""
    df_renamed["Frec"] = ""

    for i in range(len(df_renamed)):

        if df_renamed.loc[i]["Time"] == "Initial":
            df_renamed.loc[i,"Weight"] = df_renamed.loc[i]["Peso inicial"]
            df_renamed.loc[i,"BMI"] = df_renamed.loc[i]["IMC Inicial"]
            df_renamed.loc[i,"Fat"] = df_renamed.loc[i]["Grasa inicial"]
            df_renamed.loc[i,"CVRI"] = df_renamed.loc[i]["IRCV inicial"] 
            df_renamed.loc[i,"Bpmin"] = df_renamed.loc[i]["Bpmin inicial"] 
            df_renamed.loc[i,"Bpmax"] = df_renamed.loc[i]["Bpmax inicial"] 
            df_renamed.loc[i,"Frec"] = df_renamed.loc[i]["Frec inicial"] 
                
        if df_renamed.loc[i]["Time"] == "Final":
        
            df_renamed.loc[i,"Weight"] = df_renamed.loc[i]["Peso final"]
            df_renamed.loc[i,"BMI"] = df_renamed.loc[i]["IMC Final"]
            df_renamed.loc[i,"Fat"] = df_renamed.loc[i]["Grasa final"]
            df_renamed.loc[i,"CVRI"] = df_renamed.loc[i]["IRCV Final"] 
            df_renamed.loc[i,"Bpmin"] = df_renamed.loc[i]["Bpmin final"] 
            df_renamed.loc[i,"Bpmax"] = df_renamed.loc[i]["Bpmax final"] 
            df_renamed.loc[i,"Frec"] = df_renamed.loc[i]["Frec final"] 
        
    df_renamed.drop(columns = ["Peso inicial", "Peso final", "Delta Peso", "Talla", "IMC Inicial", "IMC Final", "Delta IMC", "Grasa inicial", "Grasa final", "Delta Grasa", "IRCV Final", "IRCV inicial", "Bpmin final", "Bpmin inicial", "Bpmax final", "Bpmax inicial", "Frec final", "Frec inicial",], inplace=True )
  
  df_renamed.drop(columns = ["Unnamed: 0", "grouping"], inplace=True )
  df_renamed.fillna(0, inplace=True)
  return (df_renamed, df_name)

def scaling(df_read):
   
   scaler = preprocessing.MinMaxScaler()
   numCols = df_read.select_dtypes(include=np.number).drop("numVol",1).columns
   df_read[numCols] = scaler.fit_transform(df_read[numCols])
   return df_read

def encodingSplitting(df):
  df = pd.get_dummies(df, columns = ["Sweetener", "Sex", "Time"], drop_first=False)
  X_met, y_met = df[df["Time_Initial"] == 1].drop(["numVol", "Time_Initial", "Time_Final"], axis=1), df[df["Time_Final"] == 1].drop(['Sweetener_SA', 'Sweetener_ST','Sweetener_SU', 'Sex_MAN', 'Sex_WOMAN', 'Time_Final', 'Time_Initial','numVol', 'Weight','BMI', 'Fat', 'CVRI', 'Bpmin', 'Bpmax', 'Frec'], axis = 1)
  X_metTrain, X_metTest, y_metTrain, y_metTest = train_test_split(X_met, y_met, test_size=0.2, random_state=42)

  X_full, y_full = df[df["Time_Initial"] == 1].drop(["numVol", "Time_Initial", "Time_Final"], axis=1), df[df["Time_Final"] == 1].drop(['numVol','Sweetener_SA', 'Sweetener_ST','Sweetener_SU','Time_Final', 'Time_Initial'], axis = 1)
  X_fullTrain, X_fullTest, y_fullTrain, y_fullTest = train_test_split(X_full, y_full, test_size=0.3, random_state=42)

  return(X_met, y_met, X_metTrain, X_metTest, y_metTrain, y_metTest, X_full, y_full, X_fullTrain, X_fullTest, y_fullTrain, y_fullTest)


#df_PF = pd.get_dummies(scaling(fullRead("data/plasmFlav_ord.csv",  sep = ",", anthro= True)), columns = ["Sweetener", "Sex", "Time"], drop_first=False)
#df_PA = scaling(fullRead("data/plasmAnt_ord.csv",  sep = ",", anthro= True))
#df_UF = scaling(fullRead("data/urineFlav_ord.csv",  sep = ",", anthro= True))
#df_UA = scaling(fullRead("data/urineAnt_ord.csv",  sep = ",", anthro= True))
#X_test.to_csv("X_met_test_urineAnt.csv", index=False)
#X_fulltest.to_csv("X_full_test_urineAnt.csv",index=False)

def XGBReg (df, df_name, met):

    X_met, y_met, X_metTrain, X_metTest, y_metTrain, y_metTest, X_full, y_full, X_fullTrain, X_fullTest, y_fullTrain, y_fullTest = encodingSplitting(df)

    if (met):
        
        xgbReg = XGBRegressor()

        param_grid = {'max_depth'        : [None, 1, 3, 5, 10, 20],
                    'subsample'        : [0.5, 1],
                    'learning_rate'    : [0.001, 0.01, 0.1],
                    'booster'          : ['gbtree', 'gblinear', 'dart']
                    }


        grid_search = GridSearchCV(estimator = xgbReg, param_grid = param_grid, cv= 3, n_jobs=-1,
                                verbose=2)

        grid_search.fit(X_metTrain, y_metTrain)
        best_grid = grid_search.best_estimator_


        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        n_scores = cross_val_score(best_grid, X_metTest, y_metTest,  scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
        n_scores = np.absolute(n_scores)

        print("Only metabolic model " + df_name +' MAE: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
        MAE = (np.mean(n_scores), np.std(n_scores))
        return(grid_search, MAE)
    
    else:
        xgbReg = XGBRegressor()

        param_grid = {'max_depth'        : [None, 1, 3, 5, 10, 20],
                    'subsample'        : [0.5, 1],
                    'learning_rate'    : [0.001, 0.01, 0.1],
                    'booster'          : ['gbtree', 'gblinear', 'dart']
                    }


        grid_search = GridSearchCV(estimator = xgbReg, param_grid = param_grid, cv= 3, n_jobs=-1,
                                verbose=2)

        grid_search.fit(X_fullTrain, y_fullTrain)
        best_grid = grid_search.best_estimator_

        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        n_scores = cross_val_score(best_grid, X_fullTest, y_fullTest,  scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
        n_scores = np.absolute(n_scores)

        print("Full model "+ df_name + ' MAE: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
        MAE = (np.mean(n_scores), np.std(n_scores))
        return(grid_search, MAE)        
    
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
# import pickle

# define model

def randomForestReg(df, df_name, met = True):

    param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
    
    if (met):
        
        X_met, y_met, X_metTrain, X_metTest, y_metTrain, y_metTest, X_full, y_full, X_fullTrain, X_fullTest, y_fullTrain, y_fullTest = encodingSplitting(df)

        model = RandomForestRegressor()

        grid_search = GridSearchCV(estimator = model, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

        grid_search.fit(X_metTrain, y_metTrain)
        best_grid = grid_search.best_estimator_
        # define the evaluation procedure
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        # evaluate the model and collect the scores
        n_scores = cross_val_score(best_grid, X_metTest, y_metTest, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
        # force the scores to be positive
        n_scores = np.absolute(n_scores)
        # summarize performance

        #filename = 'rf_met_plasmAnt.pkl'
        #with open(filename, 'wb') as file:
        #    pickle.dump(grid_search.best_estimator_, file)

        print('Only Metabolic model ' + df_name + ': MAE: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
        MAE = (np.mean(n_scores), np.std(n_scores))
        return(grid_search, MAE)

    else:
            
        X_met, y_met, X_metTrain, X_metTest, y_metTrain, y_metTest, X_full, y_full, X_fullTrain, X_fullTest, y_fullTrain, y_fullTest = encodingSplitting(df)

        model = RandomForestRegressor()

        grid_search = GridSearchCV(estimator = model, param_grid = param_grid, 
                        cv = 3, n_jobs = -1, verbose = 2)

        grid_search.fit(X_fullTrain, y_fullTrain)
        best_grid = grid_search.best_estimator_
        # define the evaluation procedure
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        # evaluate the model and collect the scores
        n_scores = cross_val_score(best_grid, X_fullTest, y_fullTest, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
        # force the scores to be positive
        n_scores = np.absolute(n_scores)
        # summarize performance

        #filename = 'rf_met_plasmAnt.pkl'
        #with open(filename, 'wb') as file:
        #    pickle.dump(grid_search.best_estimator_, file)

        print('Full model ' + df_name + ': MAE: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
        MAE = (np.mean(n_scores), np.std(n_scores))
        return(grid_search, MAE)

# mlp for multi-output regression

import numpy as np
import tensorflow as tf
# import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_val_score, RepeatedKFold, GridSearchCV

tf.get_logger().setLevel('ERROR')
# fix random seed for reproducibility

def MLPReg (df, df_name, met):
    
    if (met):
        seed = 7
        tf.random.set_seed(seed)

        X_met, y_met, X_metTrain, X_metTest, y_metTrain, y_metTest, X_full, y_full, X_fullTrain, X_fullTest, y_fullTrain, y_fullTest = encodingSplitting(df)


        epochs = [10, 50, 100]
        batch_size = [10, 20, 40, 60, 80, 100]

        param_grid = dict(batch_size=batch_size, epochs=epochs)

        # get the model
        def get_model(n_inputs, n_outputs):
            model_nn = Sequential()
            model_nn.add(Dense(64, input_shape=(n_inputs,),activation="relu"))
            model_nn.add(Dropout(0.5))
            model_nn.add(Dense(32, activation="relu"))
            model_nn.add(Dropout(0.5))
            model_nn.add(Dense(n_outputs, activation='linear'))
            model_nn.compile(loss='mae', optimizer=tf.keras.optimizers.Adam())

            return model_nn
        
        # evaluate a model using repeated k-fold cross-validation
        def evaluate_model(X, y):
            results = list()
            n_inputs, n_outputs = X.shape[1], y.shape[1]
            # define evaluation procedure
            cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            # define modeld
            model_nn = KerasRegressor(model = get_model(n_inputs, n_outputs), optimizer=tf.keras.optimizers.Adam(), verbose=0)
            # fit model
            grid = GridSearchCV(estimator=model_nn, param_grid=param_grid, n_jobs=-1, cv=3, verbose=0)
            grid_result = grid.fit(X_train, y_train) # evaluate model on test set
            # summarize results
            best_grid = grid_result.best_estimator_
            # define the evaluation procedure
            cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
            # evaluate the model and collect the scores
            n_scores = cross_val_score(best_grid, X_test, y_test, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
            # force the scores to be positive
            n_scores = np.absolute(n_scores)

            # store the model
            # filename = 'mlp_met_plasmAnt.pkl'
            # with open(filename, 'wb') as file:
            #    pickle.dump(grid_result.best_estimator_, file)

            # summarize performance
            print('Only Metabolic model ' + df_name +'MAE: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
            MAE = (np.mean(n_scores), np.std(n_scores))
            return(best_grid, MAE)
            # evaluate model
            
        modelMLP, MAE = evaluate_model(X_met, y_met)
    
    else:
        seed = 7
        tf.random.set_seed(seed)

        X_met, y_met, X_metTrain, X_metTest, y_metTrain, y_metTest, X_full, y_full, X_fullTrain, X_fullTest, y_fullTrain, y_fullTest = encodingSplitting(df)


        epochs = [10, 50, 100]
        batch_size = [10, 20, 40, 60, 80, 100]

        param_grid = dict(batch_size=batch_size, epochs=epochs)

        # get the model
        def get_model(n_inputs, n_outputs):
            model_nn = Sequential()
            model_nn.add(Dense(64, input_shape=(n_inputs,),activation="relu"))
            model_nn.add(Dropout(0.5))
            model_nn.add(Dense(32, activation="relu"))
            model_nn.add(Dropout(0.5))
            model_nn.add(Dense(n_outputs, activation='linear'))
            model_nn.compile(loss='mae', optimizer=tf.keras.optimizers.Adam())

            return model_nn
        
        # evaluate a model using repeated k-fold cross-validation
        def evaluate_model(X, y):
            results = list()
            n_inputs, n_outputs = X.shape[1], y.shape[1]
            # define evaluation procedure
            cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            # define modeld
            model_nn = KerasRegressor(model = get_model(n_inputs, n_outputs), optimizer=tf.keras.optimizers.Adam(), verbose=0)
            # fit model
            grid = GridSearchCV(estimator=model_nn, param_grid=param_grid, n_jobs=-1, cv=3, verbose=0)
            grid_result = grid.fit(X_train, y_train) # evaluate model on test set
            # summarize results
            best_grid = grid_result.best_estimator_
            # define the evaluation procedure
            cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
            # evaluate the model and collect the scores
            n_scores = cross_val_score(best_grid, X_test, y_test, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
            # force the scores to be positive
            n_scores = np.absolute(n_scores)

            # store the model
            # filename = 'mlp_met_plasmAnt.pkl'
            # with open(filename, 'wb') as file:
            #    pickle.dump(grid_result.best_estimator_, file)

            # summarize performance
            print('Full model ' + df_name + 'MAE: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
            MAE = (np.mean(n_scores), np.std(n_scores))
            return(best_grid, MAE)
            # evaluate model
            
        modelMLP, MAE = evaluate_model(X_full, y_full)    
    return (modelMLP, MAE)

