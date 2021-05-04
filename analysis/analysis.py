import pandas as pd

import matplotlib.pyplot as plt
from kneed import KneeLocator
from scipy.cluster import hierarchy
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score, f1_score, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS, cluster_optics_dbscan
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as ex
import statsmodels.api as sm
from sklearn.mixture import GaussianMixture
import seaborn as sns

_source_dir_ = r'C:\dev\download\csvs_all'

df = pd.read_csv(_source_dir_+"\\master_list.csv")

def find_keys(keys,str,tuple_index=0,include_dist_year=True):
    _key = []
    if include_dist_year:
        _key.append('DISTRICT_CODE')
        _key.append('year')
    for k in keys:
        if type(k) == type((1, 2,)):
            if k[tuple_index].find(str) >-1:
                _key.append(k)
        elif k.find(str)>-1:
            _key.append(k)
    return _key

#salaries
df_salaries = df[find_keys(df.keys(),'salary')]
df_salaries[df_salaries.columns] = df_salaries[df_salaries.columns].apply(pd.to_numeric, errors='coerce')
df_salaries = df_salaries.fillna(df_salaries.median())
x_np = np.array(df_salaries.loc[:,'admin_salary_local_exp':])
y_np = np.array(df_salaries.loc[:,:'year'])


#overall minuse act
df_predictors = df.drop(columns=find_keys(df.keys(),'act_graduates',include_dist_year=False))
df_predictors = df_predictors.drop(columns=find_keys(df_predictors.keys(),'act_statewide',include_dist_year=False))
df_predictors = df_predictors.replace(["*",'--'],np.nan)

plt.figure(figsize=(20,6))
plt.hist(df_predictors.count())
plt.xlabel("Number of Observations")
plt.ylabel("Number of Variables")
plt.title("Variables with Fidelity")

df_predictors_fuller = df_predictors.filter(df_predictors.count()[df_predictors.count() > 2000].keys())
plt.figure(figsize=(20,6))
plt.hist(df_predictors_fuller.count())
plt.xlabel("Number of Observations")
plt.ylabel("Number of Variables")
plt.title("Variables with Fidelity")
df_predictors_fuller = df_predictors_fuller.replace(["*",'--'],np.nan)
df_predictors_fuller = df_predictors_fuller.fillna(df_predictors_fuller.median())
#df_predictors = df_predictors.replace(np.nan,0)
df_predictors_floats = df_predictors.filter(df_predictors.dtypes[df_predictors.dtypes == 'float64'].keys())
x_np_all = np.array(df_predictors.reset_index())
x_np_all_floats = np.array(df_predictors+df_predictors_floats)

x_df_fuller = df_predictors_fuller.drop(['DISTRICT_CODE','year'],axis=1)

x_np_fuller = np.array(x_df_fuller)
scaler = StandardScaler()
x_np_fuller_stand = scaler.fit_transform(x_np_fuller)

#act

df_act = df[find_keys(df.keys(),'act_graduates')]
df_act_all = df_act[find_keys(df_act.keys(),'All Students',1)]
df_act_all[df_act_all.columns] = df_act_all[df_act_all.columns].apply(pd.to_numeric, errors='coerce')
df_act_all = df_act_all.fillna(df_act_all.median())

#swod act
df_act_swod = df_act[find_keys(find_keys(find_keys(df_act.keys(),'Composite'),'Disability'),'SwoD')]

#swd act
df_act_swd = df_act[find_keys(find_keys(find_keys(df_act.keys(),'Composite'),'Disability'),'SwD')]

#objectives
# act overall
df_act_comp = df_act_all[find_keys(df_act_all.keys(),'Compos',3)]
y_np_act = np.array(df_act_all)
y_np_act_comp = np.array(df_act_comp.drop(columns=['DISTRICT_CODE','year']))


# define dataset
# X, _ = make_classification(n_samples=1000, n_features=5, n_informative=4, n_redundant=0, n_clusters_per_class=2, random_state=5)
# define the model
model = GaussianMixture(n_components=2)
model2 = GaussianMixture(n_components=4)
# fit the model
v_df = df_salaries.drop(['DISTRICT_CODE','year'],axis=1)
v_df.columns = [k.lower().replace("_"," ") for k in v_df.columns]
model.fit(v_df)
model2.fit(v_df)
# assign a cluster to each example
yhat = model.predict(v_df)
yhat2 = model2.predict(v_df)
v_df['yhat'] = yhat
# sns.pairplot(v_df,hue='yhat')
v_df['yhat'] = yhat2
# sns.pairplot(v_df,hue='yhat')

####################################
###  ACT predictor via salary
####################################


X_train, X_test, y_train, y_test = train_test_split(x_np_fuller_stand, df_act_all, test_size=0.2, random_state=0)

y_train = y_train["('act_graduates_certified', 'All Students', 'All Students', 'Combined')"]
y_test = y_test["('act_graduates_certified', 'All Students', 'All Students', 'Combined')"]

pca = PCA(n_components=193)
pca.fit(x_np_fuller_stand)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
X_pca = pca.transform(x_np_fuller_stand)

df_p = df_predictors_fuller.drop(['DISTRICT_CODE','year'],axis=1)

PCloadings = pca.components_.T * np.sqrt(pca.explained_variance_)
components=df_p.columns.tolist()
#components=components[9:21]
loadingdf=pd.DataFrame(PCloadings)
loadingdf["variable"]=components
loadingdf

fig=ex.scatter(x=loadingdf[0],y=loadingdf[1],hover_name=loadingdf['variable'], color=[f.split('_')[0].replace("('","") for f in loadingdf['variable']],)
fig.update_layout(height=600,width=500,title_text='loadings plot')
fig.update_traces(textposition='bottom center')
fig.add_shape(type="line",x0=-0, y0=-0.5,x1=-0,y1=2.5,line=dict(color="RoyalBlue",width=3))
fig.add_shape(type="line",x0=-1, y0=0,x1=1,y1=0,line=dict(color="RoyalBlue",width=3))
fig.show()

# Calculate cumulative explained variance across all PCs

cum_exp_var = []
var_exp = 0
for i in pca.explained_variance_ratio_:
    var_exp += i
    cum_exp_var.append(var_exp)

# Plot cumulative explained variance for all PCs

plt.figure(figsize=(8,6))
plt.bar(range(1,194), cum_exp_var)
plt.xlabel('# Principal Components')
plt.ylabel('% Cumulative Variance Explained');


model = LinearRegression()
model.fit(X_train, y_train)
model.score(X_test, y_test)
# 0.20757938708775936
print('debug')

scores = []
for i in range(2,193):

    pca = PCA(n_components=i)

    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    model_pca = LinearRegression()
    model_pca.fit(X_train_pca, y_train)
    scores.append(model_pca.score(X_test_pca, y_test))
# 0.17681909964374032

plt.figure(figsize=(8,6))
plt.bar(range(1,192), scores)

####################################
###  Stepwise ACT predictors
####################################

import pandas as pd
import numpy as np
import statsmodels.api as sm
X = x_df_fuller
X.columns = [c.lower().replace("'","").replace("(","").replace(")","").replace(" ","_").replace(",","") for c in X.columns]
X = X.apply(lambda col:pd.to_numeric(col, errors='coerce'))
y = df_act_all["('act_graduates_certified', 'All Students', 'All Students', 'Combined')"]
y.name = 'act_score'

def stepwise_selection(X, y,initial_list=[],thresh_in=0.01,thresh_out = 0.05):
    included = list(initial_list)
    while True:
        changed=False
        # forward
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < thresh_in:
            best_feature = new_pval.index[new_pval.argmin()]
            included.append(best_feature)
            changed=True
        #back
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()

        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()
        if worst_pval > thresh_out:
            changed=True
            worst_feature = pvalues.index[pvalues.argmax()]
            included.remove(worst_feature)
        if not changed:
            break
    return included, model

#overall

result, md1 = stepwise_selection(X, y)

print('resulting features Overall ACT:')
print(result)
print('model data')
print(md1.summary())

#overall swod
y2 = df_act_swod["('act_graduates_certified', 'Disability Status', 'SwoD', 'Composite')"]
y2.name = 'act_score_swod'
y2 = y2.fillna(y2.median())
result2, model2 = stepwise_selection(X, y2)

print('resulting features Overall ACT:')
print(result2)
print('model data')
print(model2.summary())

# #overall swd
y3 = df_act_swd["('act_graduates_certified', 'Disability Status', 'SwD', 'Composite')"]
y3.name = 'act_score_swd'
y3 = y3.fillna(y3.median())
result3, model3 = stepwise_selection(X, y3)

print('resulting features Overall ACT:')
print(result3)
print('model data')
print(model3.summary())


#
# reduced_data = X[result]
#
# model = sm.OLS(y, sm.add_constant(pd.DataFrame(X))).fit()
#
# reg = LinearRegression().fit(reduced_data,y)
# reg.score(reduced_data, y)
# reg.coef_
# reg.intercept_
# reg.predict(np.array([[3, 5]]))


####################################
###  ACT result set mixture
####################################

X1 = df_predictors_fuller.copy()

X1.columns = [c.lower().replace("'","").replace("(","").replace(")","").replace(" ","_").replace(",","") for c in X1.columns]
X2 = X1.copy()
X3 = X1.copy()
X1 = X1[result]
X2 = X2[result2]
X3 = X3[result3]
models = [X1,X2,X3]

model = GaussianMixture(n_components=2)
model2 = GaussianMixture(n_components=4)

for m in models:
    # fit the model
    v_df = m
    # v_df.columns = [k.lower().replace("_"," ") for k in v_df.columns]
    model.fit(v_df)
    model2.fit(v_df)
    # assign a cluster to each example
    yhat = model.predict(v_df)
    yhat2 = model2.predict(v_df)
    v_df['yhat'] = yhat
    sns.pairplot(v_df,hue='yhat')
    v_df['yhat'] = yhat2
    sns.pairplot(v_df,hue='yhat')


####################################
###  ACT predictor via salary
####################################

# cluster
####################################
###  Hierarchy
####################################
clusters = hierarchy.linkage(x_np_fuller_stand, method="complete")
clusters[:10]
def plot_dendrogram(clusters):
    plt.figure(figsize=(20,6))
    dendrogram = hierarchy.dendrogram(clusters, labels=y_np, orientation="top",leaf_font_size=9, leaf_rotation=360)
    plt.ylabel('Euclidean Distance');
# plot_dendrogram(clusters)



####################################
###  Optics
####################################

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


X = df_predictors_fuller.copy()

X.columns = [c.lower().replace("'","").replace("(","").replace(")","").replace(" ","_").replace(",","") for c in X.columns]

# X = x_np_all
clusts = []
xi= 0.05
for c in range(20):
    xi = xi +0.05
    clust = OPTICS(min_samples=5, cluster_method='xi',xi=xi, p=1, algorithm='brute')
    clust.fit(X[result3])
    labels_050 = cluster_optics_dbscan(reachability=clust.reachability_,
                                       core_distances=clust.core_distances_,
                                       ordering=clust.ordering_, eps=0.5)
    labels_200 = cluster_optics_dbscan(reachability=clust.reachability_,
                                       core_distances=clust.core_distances_,
                                       ordering=clust.ordering_, eps=2)

    space = np.arange(len(X))
    reachability = clust.reachability_[clust.ordering_]
    labels = clust.labels_[clust.ordering_]

    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(2, 3)
    ax1 = plt.subplot(G[0, :])
    ax2 = plt.subplot(G[1, 0])
    ax3 = plt.subplot(G[1, 1])
    ax4 = plt.subplot(G[1, 2])

    # Reachability plot
    colors = ['g.', 'r.', 'b.', 'y.', 'c.']
    for klass, color in zip(range(0, 5), colors):
        Xk = space[labels == klass]
        Rk = reachability[labels == klass]
        ax1.plot(Xk, Rk, color, alpha=0.3)
    ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
    ax1.plot(space, np.full_like(space, 2., dtype=float), 'k-', alpha=0.5)
    ax1.plot(space, np.full_like(space, 0.5, dtype=float), 'k-.', alpha=0.5)
    ax1.set_ylabel('Reachability (epsilon distance)')
    ax1.set_title('Reachability Plot')

    # OPTICS
    # colors = ['g.', 'r.', 'b.', 'y.', 'c.']
    # for klass, color in zip(range(0, 5), colors):
    #     Xk = X[clust.labels_ == klass]
    #     ax2.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    # ax2.plot(X[clust.labels_ == -1, 0], X[clust.labels_ == -1, 1], 'k+', alpha=0.1)
    # ax2.set_title('Automatic Clustering\nOPTICS')
    #
    # # DBSCAN at 0.5
    # colors = ['g', 'greenyellow', 'olive', 'r', 'b', 'c']
    # for klass, color in zip(range(0, 6), colors):
    #     Xk = X[labels_050 == klass]
    #     ax3.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3, marker='.')
    # ax3.plot(X[labels_050 == -1, 0], X[labels_050 == -1, 1], 'k+', alpha=0.1)
    # ax3.set_title('Clustering at 0.5 epsilon cut\nDBSCAN')
    #
    # # DBSCAN at 2.
    # colors = ['g.', 'm.', 'y.', 'c.']
    # for klass, color in zip(range(0, 4), colors):
    #     Xk = X[labels_200 == klass]
    #     ax4.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    # ax4.plot(X[labels_200 == -1, 0], X[labels_200 == -1, 1], 'k+', alpha=0.1)
    # ax4.set_title('Clustering at 2.0 epsilon cut\nDBSCAN')

    plt.tight_layout()
    plt.show()

    print('stop')