import geopandas as gp
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import interact
import contextily as ctx
import statsmodels.api as sm
from matplotlib.collections import PolyCollection
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

from IPython.display import Markdown, display
from itertools import combinations as combo
from matplotlib import pyplot as plt

#Linear regression
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

#Geographically weighted statistics
from pysal.model.mgwr.gwr import GWR,GWRResults
from pysal.model.mgwr.diagnostics import corr

#Clustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score,silhouette_samples
from scipy.cluster.hierarchy import dendrogram
from sklearn.ensemble import RandomForestClassifier


def correlation_arrest_col(districts, arr_column= "Arrest"):
    corrSchools = districts[[arr_column]+[col for col in districts.columns if "Schools" in col]].corr()
    corrSchools = pd.DataFrame(corrSchools.loc[[col for col in districts.columns if "Schools" in col],[arr_column]])

    plt.figure(figsize=(16,1))
    sns.heatmap(corrSchools.T, cmap = "coolwarm");
    
    corrPI = districts[[arr_column]+[col for col in districts.columns if "PI" in col]].corr()
    corrPI = pd.DataFrame(corrPI.loc[[col for col in districts.columns if "PI" in col],[arr_column]])

    plt.figure(figsize=(16,1))
    sns.heatmap(corrPI.T, cmap = "coolwarm");
    
    corrPop = districts[[arr_column]+[col for col in districts if "demo" in col]+["Population_2019"]].corr()
    corrPop = pd.DataFrame(corrPop.loc[[col for col in districts if "demo" in col]+["Population_2019"],[arr_column]])

    plt.figure(figsize=(16,1))
    sns.heatmap(corrPop.T, cmap = "coolwarm");
    
#GWR or GWRResult does not calculate geographically-weighted correlation coefficients for all variables
#So we adapt our own function
def all_corr(results,variables): #R. Henkin, “VA_brexit_practical_w7,” INM433 Visual Analytics (PRD1 A 2019/20), 2019.
    """
    Computes  local correlation coefficients (n, (((p+1)**2) + (p+1) / 2) within a geographically
    weighted design matrix
    Returns one array with the order and dimensions listed above where n
    is the number of locations used as calibrations points and p is the
    number of explanatory variables; +1 accounts for the dependent variable.
    Local correlation coefficient is not calculated for constant term.
    """
    #print(self.model)
    x = results.X
    y = results.y
    x = np.column_stack((x,y))
    w = results.W
    nvar = x.shape[1]
    nrow = len(w)
    if results.model.constant:
        ncor = (((nvar - 1)**2 + (nvar - 1)) / 2) - (nvar - 1)
        jk = list(combo(range(1, nvar), 2))
    else:
        ncor = (((nvar)**2 + (nvar)) / 2) - nvar
        jk = list(combo(range(nvar), 2))
    corr_mat = np.ndarray((nrow, int(ncor)),dtype=dict)
    
    for i in range(nrow):
        wi = w[i]
        sw = np.sum(wi)
        wi = wi / sw
        tag = 0

        for j, k in jk:
            val = corr(np.cov(x[:, j], x[:, k], aweights=wi))[0][1] 
            corr_mat[i,tag] = {"var": variables[j-1]+"_"+variables[k-1], "var_1": variables[j-1], "var_2": variables[k-1], "value": val}
            tag = tag + 1
            
    return corr_mat
    
