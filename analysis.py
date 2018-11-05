
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('train.csv')

#replace missing with zeros
data.fillna(0, inplace = True)

clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)

clf.fit()
