################################################################
# TALENT SCOUTING CLASSIFICATION WITH MACHINE LEARNING
################################################################

################################################################
# Business Problem
################################################################

# Predicting which class (average, highlighted) players are according to the scores
# given to the characteristics of the players watched by the scouts.

################################################################
# Data Set Story:
################################################################

# The dataset consists of information from Scoutium that includes the attributes and scores
# of the football players rated by the scouts according to the attributes of the
# football players observed in the matches.
# Attributes: It contains the scores given by the users evaluating the players
# to the attributes of each player they watched and evaluated in a match (independent variables).
# Potential_labels: Contains the potential labels of the users who evaluated the players,
# including their final opinion about the players in each match (target variable)
# 9 Variables, 10730 Observations, 0.65 mb

################################################################
# Variables
################################################################

# task_response_id: Set of a scout's evaluations of all players in a team's squad in a match.

# match_id: The id of the match.

# evaluator_id: The id of the evaluator (scout).

# player_id: id of the respective player.

# position_id: id of the position the player played in that match.

# 1- Goalkeeper
# 2- Stopper
# 3- Right back
# 4- Left back
# 5- Defensive midfield
# 6- Central midfield
# 7- Right wing
# 8- Left wing
# 9- Offensive midfield
# 10- Striker

# analysis_id: Set containing a scout's attribute evaluations of a player in a match.

# attribute_id: The id of each attribute by which players are evaluated.

# attribute_value: The value (score) given by a scout to an attribute of a player.

# potential_label: Label indicating a scout's final decision on a player in a match (target variable).

#####################
#EXPLORATORY DATA ANALYSIS
#####################

####################
#Needed libraries und Functions
####################

import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
import datetime as dt
import warnings
warnings.simplefilter(action="ignore")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import AgglomerativeClustering

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df2 = pd.read_csv("DataScience/datasets/scoutium_attributes.csv",sep=";")
df1 = pd.read_csv("DataScience/datasets/scoutium_potential_labels.csv",sep=";")
df1.head()
df2.head()

df = pd.merge(df1, df2, how='left', on=["task_response_id", "match_id", "evaluator_id", "player_id"])
df.head()

#Remove the Goalkeeper (1) class in position_id from the dataset.

df = df[df["position_id"] != 1]
df.head()

#Remove the below_average class in potential_label from the dataset.
# (the below_average class makes up 1% of the entire dataset)

df = df[df["potential_label"] != "below_average"]

#####################
#Pivot Table
#####################

pt = pd.pivot_table(df, values="attribute_value", columns="attribute_id", index=["player_id","position_id","potential_label"])

pt = pt.reset_index(drop=False)
pt.columns = pt.columns.map(str)
pt.head()

#Numeric Variables

num_cols = pt.columns[3:]

####################
#Observations
####################

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())

check_df(pt)

######################
# Numeric and categorical variables.
######################

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(pt, col, plot=True)

##########
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in ["position_id","potential_label"]:
    cat_summary(pt, col, plot=True)

#####################
#Target variable analysis
#####################

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(pt, "potential_label", col)

#######################
#Correlation
#######################

pt[num_cols].corr()

f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(pt[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

########################
#Feature Extraction
########################

pt["min"] = pt[num_cols].min(axis=1)
pt["max"] = pt[num_cols].max(axis=1)
pt["sum"] = pt[num_cols].sum(axis=1)
pt["mean"] = pt[num_cols].mean(axis=1)
pt["median"] = pt[num_cols].median(axis=1)

pt["mentality"] = pt["position_id"].apply(lambda x: "defender" if (x == 2) | (x == 5) | (x == 3) | (x == 4) else "attacker")

pt.head()

###############################
# Label Encoder
###############################

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


labelEncoderCols = ["potential_label","mentality"]

for col in labelEncoderCols:
    pt = label_encoder(pt, col)

##########################
#Standard Scaler
##########################

lst = ["min","max","sum","mean","median"]
num_cols = list(num_cols)

for i in lst:
    num_cols.append(i)

scaler = StandardScaler()
pt[num_cols] = scaler.fit_transform(pt[num_cols])

pt.head()

#######################
#Modelling
#######################

y = pt["potential_label"]
X = pt.drop(["potential_label", "player_id"], axis=1)


models = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ("LightGBM", LGBMClassifier())]



for name, model in models:
    print(name)
    for score in ["roc_auc", "f1", "precision", "recall", "accuracy"]:
        cvs = cross_val_score(model, X, y, scoring=score, cv=10).mean()
        print(score+" score:"+str(cvs))

pt.head()

#############################
#Hyperparameter Optimization
#############################

lgbm_model = LGBMClassifier(random_state=46)

lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [500, 1500],
               "colsample_bytree": [0.5, 0.7, 1]
             }

lgbm_gs_best = GridSearchCV(lgbm_model,
                            lgbm_params,
                            cv=3,
                            n_jobs=-1,
                            verbose=True).fit(X, y)

final_model = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X, y)

rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=5, scoring="neg_mean_squared_error")))

##########################
#Feature Importance
##########################

def plot_importance(model, features, num=len(X), save=False):

    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")

model = LGBMClassifier()
model.fit(X, y)

plot_importance(model, X)
