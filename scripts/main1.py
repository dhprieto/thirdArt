from utils import fullRead, scaling, encoding, trainTestSplit
from training import trainModel, permutation_plot, individual_plots

df1_anthro = scaling(fullRead("data/urineAnt_ord.csv",  sep = ",", anthro= True))
df1_metab = scaling(fullRead("data/urineAnt_ord.csv",  sep = ",", anthro= False))

df1_anthro = encoding(df=df1_anthro,categorical_vars=["Sweetener", "Sex"]) 

X, y, X_train, X_test, y_train, y_test = trainTestSplit(df1_anthro, dropInitial=["numVol", "Time"], dropFinal = ['Time','numVol', 'Sweetener', 'Time', 'Sex', 'Weight','BMI', 'Fat', 'CVRI', 'Bpmin', 'Bpmax', 'Frec'])



best_grid = trainModel(X_train, y_train, modelFit = "rf")
