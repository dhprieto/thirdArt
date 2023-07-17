import seaborn as sns
import pandas as pd
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import numpy as np

def plot_heatmap(best_grid, X_train, y_train, save, name):
    df_importance = pd.DataFrame(columns = y_train.columns, index = X_train.columns)
    for i in range(len(y_train.columns)):
        df_importance[y_train.columns[i]] = best_grid.estimators_[i].feature_importances_ 
    plt.figure(figsize=(10,10))
    sns.heatmap(df_importance)
    if save == True:
       plt.savefig("plots/" + name + "_heatmap.png") 
    else:
        plt.show()

def permutation_plot(best_grid, X_test, y_test, save, name):

    result = permutation_importance(
        best_grid, X_test, y_test, n_repeats=100, random_state=42, n_jobs=2
    )
    forest_importances = pd.Series(result.importances_mean, index=X_test.columns)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation on full model " + name)
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    if save == True:
       plt.savefig("plots/" + name + "_permutation.png") 
    else:
        plt.show()
def individual_plots(best_grid, X_train, y_train, save, name):    
    
    feature_importance = best_grid.estimators_[0].feature_importances_
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + 0.5
    fig = plt.figure(figsize=(35,35))
    plt.subplots_adjust(hspace=0.25)
    plt.suptitle("Feature Importance for every variable for model " + name, size = 20)

    for i,j in zip(range(len(y_train.columns)), range(len(X_train.columns))):
        feature_importance= best_grid.estimators_[i].feature_importances_
        sorted_idx = np.argsort(feature_importance)
        if (len(y_train.columns) == 21):
            ax = plt.subplot(3, 7, i+1)
        else:
            ax = plt.subplot(5, 6, i+1)
        feature_importance_ordered = feature_importance[sorted_idx]
        feature_importance_no0s = feature_importance_ordered[feature_importance_ordered != 0]
        labels = np.array(X_train.columns)[sorted_idx]
        labels_no0s = labels[feature_importance_ordered != 0]
        plt.barh(pos[feature_importance_ordered != 0], feature_importance_no0s, align="center")
        plt.yticks(pos[feature_importance_ordered != 0], labels_no0s)
        ax.set_title(y_train.columns[i].upper())
    if save == True:
       plt.savefig("plots/" + name + "_individual.png") 
    else:
        plt.show()

def plotsImportance(best_grid, X_train, y_train, X_test, y_test, save, name):
    permutation_plot(best_grid, X_test, y_test, save, name)
    individual_plots(best_grid, X_train, y_train, save, name)
    plot_heatmap(best_grid, X_train, y_train, save, name)