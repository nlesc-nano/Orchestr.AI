import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

import logging
logger = logging.getLogger(__name__)

def plot_pca(features, labels, title="PCA", filename="pca.png"):
    pca = PCA(2); red = pca.fit_transform(features)
    plt.figure(figsize=(8,6))
    cmap = ["blue","green","red","orange","purple","brown","pink","gray"]
    for lbl in np.unique(labels):
        m = (labels==lbl)
        plt.scatter(red[m,0],red[m,1],label=f"grp{lbl}",c=cmap[lbl%len(cmap)],alpha=0.7)
    plt.legend(); plt.title(title)
    plt.savefig(filename, dpi=300); plt.close()

def plot_outliers(features,labels,outliers,title,filename):
    pca = PCA(2); red = pca.fit_transform(features)
    plt.figure(figsize=(8,6))
    cmap = ["blue","green","red","orange"]
    for lbl in np.unique(labels):
        m_all = (labels==lbl)
        m_in = m_all & (outliers==1)
        m_out= m_all & (outliers==-1)
        plt.scatter(red[m_in,0],red[m_in,1],c=cmap[lbl % len(cmap)],label=f"{lbl} in")
        plt.scatter(red[m_out,0],red[m_out,1],c=cmap[lbl % len(cmap)],marker='x',s=50,label=f"{lbl} out")
    plt.title(title); plt.legend()
    plt.savefig(filename, dpi=300); plt.close()
    
def plot_final_selection(features,labels,sel,title,filename):
    pca = PCA(2); red = pca.fit_transform(features)
    plt.figure(figsize=(8,6))
    cmap = ["blue","green","red","orange"]
    for lbl in np.unique(labels):
        m = labels==lbl
        plt.scatter(red[m,0],red[m,1],c=cmap[lbl % len(cmap)],alpha=0.5,label=f"{lbl}")
    plt.scatter(red[sel,0],red[sel,1],facecolors='none',edgecolors='k',s=100,label='selected')
    plt.title(title); plt.legend()
    plt.savefig(filename, dpi=300); plt.close()
    
def detect_outliers(features, contamination: float, labels, title: str, filename: str, random_state: int = 0):
    """
    IsolationForest-based outlier detection. Returns a boolean mask of inliers.
    Also renders the outlier plot using existing plot_outliers(...).
    """
    clf = IsolationForest(contamination=contamination, random_state=random_state)
    y_pred = clf.fit_predict(features)          # -1 outlier, +1 inlier
    plot_outliers(features, labels, y_pred, title, filename)
    return (y_pred == 1)