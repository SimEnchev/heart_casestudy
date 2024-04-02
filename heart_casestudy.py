import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

heart_data = pd.read_csv("heart.csv")

print(heart_data.shape)
print(heart_data.head)
print(heart_data.info())

#No missing values as indicated by the non-null values. However, there are entries in Blood Pressure and Cholesterol that are listed as 0. 
#Removing these entries ensures better data quality, physiological plausibility, and reduces noise in the machine learning algorithm. 

dataset = heart_data[(heart_data['RestingBP'] > 0) & (heart_data['Cholesterol'] > 0)]
entries_removed = heart_data.shape[0] - dataset.shape[0]
cleaned_data_preview = dataset.head()
print(entries_removed)
print('Number of entries removed:', entries_removed)

print(dataset.describe().T)

# Summary of numerical variables
# Age: The cleaned dataset includes individuals aged between 28 and 77 years, with an average age of approximately 52.9 years.
# RestingBP (Resting Blood Pressure): After cleaning, the average resting blood pressure among the individuals is about 133.02 mm Hg, with the values ranging from 92 to 200 mm Hg. 
# Cholesterol: The average cholesterol level has been adjusted to approximately 244.64 mg/dl, with the lowest recorded level at 85 mg/dl and the highest at 603 mg/dl.
# FastingBS (Fasting Blood Sugar): Approximately 16.8% of the individuals have a fasting blood sugar level above 120 mg/dl
# MaxHR (Maximum Heart Rate): The average maximum heart rate achieved is around 140.23.

#Next, must separate numerical from categorical features. 
#Then we will perform some basic EDA. 

numerical_columns = dataset[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']]
categorical_columns = dataset.drop(numerical_columns.columns, axis=1)
categorical_columns = categorical_columns.drop('HeartDisease', axis=1)

#Bar plot of distrubitons:
figure, axis = plt.subplots(ncols=2, nrows=3, figsize=(7, 9))
axis = axis.flatten()

for i, column in enumerate(numerical_columns.columns):
    sns.kdeplot(data=dataset, x=column, hue='HeartDisease', palette='magma', ax=axis[i], fill=True)

figure.delaxes(axis[-1])
plt.suptitle('Distribution of Column Variables with HeartDisease')
plt.tight_layout(pad=2)
plt.show()

# Calculating the percentage of heart disease by gender.
sex_heartdisease = dataset.groupby('Sex')['HeartDisease'].mean() * 100

# Plotting the result
sex_heartdisease.plot(kind='bar', color=['pink', 'lightblue'])
plt.title('Percentage of Heart Disease by Gender')
plt.xlabel('Gender')
plt.ylabel('Percentage with Heart Disease')
plt.xticks(rotation=0)  
plt.ylim(0, 100)  
plt.show()

# We want to find the mutual information because it helps us understand how much one feature has an effect on another. This step is crucial for accuracy within our ML model. 
#https://stackoverflow.com/questions/75261514/mutual-information-for-continuous-variables-with-scikit-learn
# ^ extremely useful

# Must apply mappings to categorical columns to negate type errors
dataset['Sex'] = dataset['Sex'].replace({'M': 1, 'F': 0})
dataset['ChestPainType'] = dataset['ChestPainType'].replace({'TA': 0, 'ATA': 1, 'NAP': 2, 'ASY': 3})
dataset['RestingECG'] = dataset['RestingECG'].replace({'Normal': 0, 'ST': 1, 'LVH': 2})
dataset['ExerciseAngina'] = dataset['ExerciseAngina'].replace({'N': 0, 'Y': 1})
dataset['ST_Slope'] = dataset['ST_Slope'].replace({'Up': 0, 'Flat': 1, 'Down': 2})

mutual_information = mutual_info_classif(dataset.drop('HeartDisease', axis=1), dataset['HeartDisease'], discrete_features='auto', random_state=1)
mi_data = {'Scores': mutual_information}
mutual_info_dataset = pd.DataFrame(mi_data, index=dataset.drop('HeartDisease', axis=1).columns)
sns.barplot(y=mutual_info_dataset.index, x='Scores', data=mutual_info_dataset, palette='magma', orient='h')
plt.title('Mutual Information Scores')
plt.show()

print('Mutual information Scores in Descending Order:')
print(mutual_info_dataset['Scores'].sort_values(ascending=False))

# Now we can select the columns to feed into our ML algorithm. We observe that ST_Slope, ChestPainType, Oldpeak, ExerciseAngina and Sex are the main features that determine heart health. 
# First we will build the Pipelines.

# Given the nature/size of this dataset, I believe that Random Forrest, Logistical Regression, and Gradient Boosting are our best bets.

# Initializing the predicators
selected_columns = ['ST_Slope', 'ChestPainType', 'Oldpeak', 'ExerciseAngina', 'Sex']
X = dataset[selected_columns]
Y = dataset['HeartDisease']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Random Forest 
# https://www.analyticsvidhya.com/blog/2015/06/tuning-random-forest-model/

rf_pipeline = Pipeline([
    ('class', RandomForestClassifier())
])

rf_paramater_grid = {
    'class__n_estimators': [90, 180],
    'class__max_depth': [None, 5, 10],
    'class__min_samples_split': [2, 5],
}

rf_grid = GridSearchCV(rf_pipeline, param_grid=rf_paramater_grid, cv=5)
rf_grid.fit(X_train, Y_train)
rf_accuracy = rf_grid.score(X_test, Y_test)
print(f"The Accuracy Score achieved during Random Forest is: {rf_accuracy} %")


# Gradient Boosting 
# https://www.machinelearningplus.com/machine-learning/an-introduction-to-gradient-boosting-decision-trees/#:~:text=Using%20a%20low%20learning%20rate,0.3%20gives%20the%20best%20results.

gb_pipeline = Pipeline([
    ('class', GradientBoostingClassifier())
])

gb_param_grid = {
    'class__n_estimators': [90, 180],
    'class__learning_rate': [0.01, 0.1], #common starting point, could be better fine tuned with more training.
    'class__max_depth': [2, 5],
}

gb_grid = GridSearchCV(gb_pipeline, param_grid=gb_param_grid, cv=5)
gb_grid.fit(X_train, Y_train)
gb_accuracy = gb_grid.score(X_test, Y_test)
print(f"The Accuracy Score achieved during Gradient Boosting is: {gb_accuracy} %")

# Logistic Regression

lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('class', LogisticRegression())
])

lr_param_grid = {
    'class__C': [0.1, 1, 10],
}

lr_grid = GridSearchCV(lr_pipeline, param_grid=lr_param_grid, cv=5)
lr_grid.fit(X_train, Y_train)
lr_accuracy = lr_grid.score(X_test, Y_test)
print(f"The Accuracy Score achieved during Logistic Regression is: {lr_accuracy} %")

final_scores = [gb_accuracy, rf_accuracy, lr_accuracy]
algorithms = ["Gradient Boosting", "Random Forest", "Logistic Regression"]
sns.set_theme(rc={'figure.figsize': (15, 8)})

plt.xlabel("Algorithms")
plt.ylabel("Accuracy Score")
sns.barplot(x=algorithms, y=final_scores) 
plt.show()

# In conclusion, while the ML algorithms performed similarly in the end, Gradient Boosting shows the highest accuracy among the three. 