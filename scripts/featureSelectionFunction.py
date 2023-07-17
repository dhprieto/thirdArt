from utils import fullRead, scaling, encoding, trainTestSplit
from plots import plotsImportance
from training import trainModel
from matplotlib import rcParams
import seaborn as sns
import pickle
import re
sns.set()

metabs_urAnt = ['CA', 'CA.G', 'CA.S', 'CA.GS', 'Total.CA', 'DHPAA', 'DHPAA.G',
                'DHPAA.GG', 'DHPAA.GS', 'DHPAA.SS', 'Total.DHPAA', 'TFA.G', 'TFA.S',
                'TFA.di.sulfate.1', 'Total.TFA', 'TIFA.Sulfate.1', 'VA', 'VA.GG',
                'VA.GS', 'VA.SS', 'Total.VA']
metabs_plAnt = ['CA', 'CA.G', 'CA.S', 'Total.CA', 'DHPAA', 'DHPAA.G',
                'DHPAA.GG', 'DHPAA.GS', 'DHPAA.SS', 'Total.DHPAA', 'TFA.G', 'TFA.S',
                'Total.TFA', 'VA', 'VA.GG', 'VA.S', 'VA.GS', 'VA.SS', 'Total.VA']
metabs_plFlav = ['E', 'E.S', 'Total.E', 'HE.G', 'N.G']
metabs_urFlav = ['E', 'EG.1', 'E.S', 'Total.E', 'HE', 'HE.G', 'HE.GG',
                 'Total.HE', 'N', 'N.G', 'N.GG', 'N.S', 'Total.N']

anthro = ['Weight','BMI', 'Fat', 'CVRI', 'Bpmin', 'Bpmax', 'Frec']
factors = ['Sweetener', 'Time', 'Sex']

df1_anthro = scaling(fullRead("data/urineFlav_ord.csv",  sep = ",", anthro= True))
#df1_metab = scaling(fullRead("data/urineAnt_ord.csv",  sep = ",", anthro= False))

df1_anthro = encoding(df=df1_anthro,categorical_vars=["Sweetener", "Sex"]) 

X, y, X_train, X_test, y_train, y_test = trainTestSplit(df1_anthro, dropInitial=["numVol", "Time"]+metabs_urFlav, dropFinal = factors + ['numVol']+metabs_urFlav)

best_grid = trainModel(X_train, y_train, modelFit="rf")

filename = 'rf_AnthroToAnthro_urineFlav_prueba.pkl'

with open(filename, 'wb') as file:
  pickle.dump(best_grid, file)

#with open(filename, 'rb') as f:
#  best_grid = pickle.load(f)

plotsImportance(best_grid, X_train, y_train, X_test, y_test, save = True, name = re.sub(".pkl", "", filename))

