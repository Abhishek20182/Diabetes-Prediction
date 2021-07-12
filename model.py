#Python Libraries 
import numpy as np
import pandas as pd
import pickle


diabetes_df = pd.read_csv('archive1.zip')
#lowering #lowercasing all the column names
diabetes_df.columns = diabetes_df.columns.str.lower()

#renaming the column name 
diabetes_df = diabetes_df.rename(columns={'diabetespedigreefunction': 'diabetes_pedigree_function', \
	'bloodpressure': 'blood_pressure', 'skinthickness': 'skin_thickness'})

# Model Building
from sklearn.model_selection import train_test_split

# segregating the target variable
X = diabetes_df.drop(columns='outcome')
y = diabetes_df['outcome']
#spliting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

# Creating Random Forest Model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=20)
classifier.fit(X_train, y_train)


# Creating a pickle file for the classifier
filename = 'model.pkl'
pickle.dump(classifier, open(filename, 'wb'))