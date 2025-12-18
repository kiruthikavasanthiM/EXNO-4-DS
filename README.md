
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM: 
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
import pandas as pd

df1=pd.read_csv("/content/bmi.csv")

df1

<img width="500" height="530" alt="image" src="https://github.com/user-attachments/assets/204a986d-544d-4b93-a056-15f6f2bdfc3a" />

from sklearn.preprocessing import
StandardScaler,MinMaxScaler,MaxAbsScaler,Normalizer,RobustScaler

df2=df1.copy()

enc=StandardScaler()

df2[['new_heights','new_weight']]=enc.fit_transform(df2[['Height','Weight']])

df2

<img width="734" height="532" alt="image" src="https://github.com/user-attachments/assets/dd352fef-dfb9-447f-8ad9-688d895af1a7" />

df3=df1.copy()

enc=MinMaxScaler()

df3[['new_heights','new_weight']]=enc.fit_transform(df2[['Height','Weight']])

df3

<img width="709" height="525" alt="image" src="https://github.com/user-attachments/assets/a070d2cb-cdbd-4258-89f6-a21896866f89" />

df4=df1.copy()

enc=MaxAbsScaler()

df4[['new_heights','new_weight']]=enc.fit_transform(df2[['Height','Weight']])

df4

<img width="728" height="503" alt="image" src="https://github.com/user-attachments/assets/07a892a4-a02d-45b6-a906-01249e4fb2e0" />

df5=df1.copy()

enc=Normalizer()

df5[['new_heights','new_weight']]=enc.fit_transform(df2[['Height','Weight']])

df5

<img width="689" height="534" alt="image" src="https://github.com/user-attachments/assets/d2648973-2f06-4650-8f62-f3d59eb2f330" />

df6=df1.copy()

enc=RobustScaler()

df6[['new_heights','new_weight']]=enc.fit_transform(df2[['Height','Weight']])

df6

<img width="701" height="502" alt="image" src="https://github.com/user-attachments/assets/420944e5-191e-4f52-94c5-92d792802f19" />

import pandas as pd

df=pd.read_csv("/content/income(1) (1).csv")

df

<img width="1416" height="628" alt="image" src="https://github.com/user-attachments/assets/194bdb1a-8bfc-44e1-8b06-f96f0eeaf3b9" />

from sklearn.preprocessing import LabelEncoder

df_encoded=df.copy()

le=LabelEncoder()

for col in df_encoded.select_dtypes(include='object').columns:
        df_encoded[col] = le.fit_transform(df_encoded[col])
        
x=df_encoded.drop("SalStat",axis=1)

y=df_encoded["SalStat"]

x

<img width="1390" height="557" alt="image" src="https://github.com/user-attachments/assets/c4258763-ecd9-412a-914a-29f68426a3fa" />

y

<img width="290" height="577" alt="image" src="https://github.com/user-attachments/assets/ed788714-d058-4758-a111-0a9e3e21d4e5" />

import pandas as pd

from sklearn.feature_selection import SelectKBest, chi2

chi2_selector=SelectKBest(chi2,k=5)

chi2_selector.fit(x,y)

selected_features_chi2=x.columns[chi2_selector.get_support()]

print("Selected features(chi_square):",list(selected_features_chi2))

mi_scores=pd.Series(chi2_selector.scores_,index = x.columns)

print(mi_scores.sort_values(ascending=False))

<img width="1043" height="325" alt="image" src="https://github.com/user-attachments/assets/9e8861e4-5e47-4cae-916d-44d37bdc5d92" />

from sklearn.feature_selection import f_classif

anova_selector=SelectKBest(f_classif,k=5)

anova_selector.fit(x,y)

selected_features_anova=x.columns[anova_selector.get_support()]

print("Selected features(chi_square):",list(selected_features_anova))

mi_scores=pd.Series(anova_selector.scores_,index = x.columns)

print(mi_scores.sort_values(ascending=False))

<img width="1010" height="328" alt="image" src="https://github.com/user-attachments/assets/dbaaa904-55a2-45b8-9a01-10e9272f18dd" />

from sklearn.feature_selection import mutual_info_classif

mi_selector=SelectKBest(mutual_info_classif,k=5)

mi_selector.fit(x,y)

selected_features_mi=x.columns[mi_selector.get_support()]

print("Selected features(Mutual Info):",list(selected_features_mi))

mi_scores=pd.Series(anova_selector.scores_,index = x.columns)

print(mi_scores.sort_values(ascending=False))

<img width="1114" height="313" alt="image" src="https://github.com/user-attachments/assets/0d0f5ac6-504c-4d9a-91ee-0c5ca9ee4de5" />

from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import RFE

model = LogisticRegression(max_iter=100)

rfe= RFE(model, n_features_to_select=5)

rfe.fit(x,y)

selected_features_rfe = x.columns[rfe.support_]

print("Selected features (RFE):",list(selected_features_rfe))

<img width="1201" height="181" alt="image" src="https://github.com/user-attachments/assets/d232555f-9922-4b96-8536-88dc500b83c7" />

from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier()

rf.fit(x,y)

importances = pd.Series(rf.feature_importances_, index=x.columns)

selected_features_rf = importances.sort_values(ascending=False).head(5).index

print(importances)

print("Top 5 features (Random Forest Importance):",list(selected_features_rf))

<img width="1271" height="323" alt="image" src="https://github.com/user-attachments/assets/74574fdd-2f82-4d4e-a337-ec366a8575cd" />

from sklearn.linear_model import LassoCV

import numpy as np
 
    
lasso=LassoCV(cv=5).fit(x,y)

importance = np.abs(lasso.coef_)

selected_features_lasso=x.columns[importance>0]

print("Selected features (Lasso):",list(selected_features_lasso))

<img width="853" height="55" alt="image" src="https://github.com/user-attachments/assets/9b1d7bd0-c5c1-4a7d-ba97-fb343065881e" />

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("/content/income(1) (1).csv")

df_encoded = df.copy()

le = LabelEncoder()
for col in df_encoded.select_dtypes(include='object').columns:

    df_encoded[col] = le.fit_transform(df_encoded[col])

X = df_encoded.drop("SalStat", axis=1)

y = df_encoded["SalStat"]


X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.3, random_state=42
    
)


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


knn = KNeighborsClassifier(n_neighbors=3)  # you can tune k

knn.fit(X_train, y_train)


y_pred = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n", classification_report(y_test, y_pred))

<img width="724" height="369" alt="image" src="https://github.com/user-attachments/assets/0d3d8db9-d669-4b01-86c0-52dc4848e786" />









# RESULT:
Thus, Feature selection and Feature scaling has been used on thegiven dataset.
