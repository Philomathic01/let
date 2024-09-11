from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import accuracy_score
iris = load_iris()
X, y = iris.data, iris.target
print(X)
# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = RandomForestClassifier(n_estimators=100)
# train the classifier on the training data
clf.fit(X_train, y_train)
# predict on the test set
y_pred = clf.predict(X_test)
# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}") # Accuracy: 0.91

joblib.dump(clf,'rf_model.sav')