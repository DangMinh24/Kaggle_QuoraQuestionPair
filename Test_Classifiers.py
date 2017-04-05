import pandas as pd
import sys
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso,Ridge
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import numpy as np
import time

data_set=pd.read_csv("data/quora_features.csv")
# print(data_set.columns)

feature_set=data_set.iloc[:,3:]
target_set=pd.DataFrame(data_set.iloc[:,2].values,columns=["id_duplicate"])

##############Procedure of checking missing value
# categorical_feature_set=feature_set.select_dtypes(exclude=["float32","float64"])
# numeric_feature_set=feature_set.select_dtypes(include=["float32","float64"])
# for i in feature_set:
#     print(i + ": ", end=" ")
#     print(feature_set[i].count())
#
# for i in categorical_feature_set:
#     print(i+": ",end=" ")
#     print(categorical_feature_set[i].count())
#
# for i in numeric_feature_set:
#     print(i+": ",end=" ")
#     print(numeric_feature_set[i].count())
###Conclusion: Cosine_distance, jaccard_distance, canberra_distance have missing value
###Solution: Have many solution to deal with missing values. BUT I do the most common way: fill with median
missing_attribute=["cosine_distance","jaccard_distance","braycurtis_distance"]
for i in missing_attribute:
    feature_set[i].fillna(value=feature_set[i].median(),inplace=True)
    # print(i + ": ", end=" ")
    # print(feature_set[i].count())

##############Check numeric feature set, see that they use "inf" symbol in wmd and norm_wmd
###Solution: replace inf value with float_MAX
inf=sys.maxsize
inf_attribute=["wmd","norm_wmd"]
feature_set["wmd"].replace(np.inf,inf,inplace=True)
feature_set["norm_wmd"].replace(np.inf,inf,inplace=True)

categorical_feature_set=feature_set.select_dtypes(exclude=["float32","float64"])
numeric_feature_set=feature_set.select_dtypes(include=["float32","float64"])


clfs=[
    LogisticRegression(random_state=3,max_iter=300,penalty="l2"),
    RandomForestClassifier(n_estimators=50,max_depth=5,min_samples_split=2,random_state=3,n_jobs=-1),
    xgb.XGBClassifier(seed=4)
]


for iter,clf in enumerate(clfs):
    start=time.time()
    cvs=cross_val_score(estimator=clf,X=feature_set,y=target_set.values.ravel(),cv=5,n_jobs=-1,scoring="log_loss")
    stop=time.time()
    duration=stop-start
    print("Model %d"%iter)
    print("Run time %f"%duration)
    print("Result : "+str(cvs))
    print("Average score : %f"%np.mean(cvs))

    print()

