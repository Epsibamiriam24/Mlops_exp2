# %%
print("URK23CS1077")
import numpy as np
import pandas as pd
import sys
#import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
#from sklearn.preprocessing import

d = pd.read_csv("data/improved_disease_dataset.csv")
d.head()
d.tail()
# Check for empty data
if d.empty:
    print("ERROR: Dataset is empty!")
    sys.exit(1)  # Stop the script with error

# Check for duplicates
if d.duplicated().any():
    print("WARNING: Dataset contains duplicate rows")
#print(d.shape)
#sns.heatmap(d.corr(), annot=True)  # removed seaborn heatmap

# %%
from sklearn.model_selection import train_test_split
#print(d.isnull().sum())

# %%
#print(d.describe())

# %%
#print(d.info)

# %%
#dropping the target column and storing in another variable
x = d.drop('disease', axis=1)
y = d["disease"]

# %%
x_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# %%
from sklearn.ensemble import RandomForestClassifier
# Random Forest Algorithm
model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)
#print(model)

# %%
# Predict on training data
y_train_pred = model.predict(x_train)

# %%
#getting the feature importance
importances = pd.Series(model.feature_importances_, index=x.columns)
important_features = importances.sort_values(ascending=False)
#print("Top Important Features:\n")
#print(important_features)

# %%
train_comparison = pd.DataFrame({
    'Actual disease': y_train,
    'Predicted disease': y_train_pred
})
#print(train_comparison.head(10))  # Show first 10 rows

# %%
top_n = 10  # You can change this to show more or fewer
plt.figure(figsize=(10, 6))

# Replace seaborn barplot with matplotlib barh
plt.barh(important_features.index[:top_n][::-1], important_features[:top_n][::-1], color=plt.cm.viridis(np.linspace(0, 1, top_n)))

plt.title("Top Feature Importances")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# %%
accuracy = accuracy_score(y_train, y_train_pred)
print("accuracy", accuracy)
