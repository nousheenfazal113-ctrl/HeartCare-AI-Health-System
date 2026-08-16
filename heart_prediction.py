import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataSet
df = pd.read_csv("data\\data.csv")

# Just check the shape and data
print(df.shape)
print(df.head())

print("\n===== Dataset information =====")
print(df.info())

print("\n===== Coulmns Names =====")
print(df.columns)

print("\n===== Ststistical Summary =====")
print(df.describe())

print("\n===== Check Missing value =====")
print(df.isnull().sum())

print("\n===== Check Duplicate Values =====")
print(df.duplicated().sum())  

# In resultant we hain 723 duplicates entries so will remove it.
df = df.drop_duplicates()
print("Shape after removing duplicates: ")
print(df.shape)

# kitne patients heart disease hain awr kinto ko nahi hain
print("\n===== Target Variables =====")
print(df["target"].value_counts())

# yeh percentge main find karte hain k kitno ko heart disease hain awr kitno ko nahi hain.
print("\n===== Target Percentage =====")
print(df["target"].value_counts(normalize=True) * 100)


plt.figure(figsize=(6,4))
sns.countplot(x = "target", data = df)
plt.title("Heart Disease Distribution")
plt.xlabel("Target")
plt.ylabel("Number Of Patients")
plt.show()

# Age destribution plot:
plt.figure(figsize=(8,5))
sns.histplot(df["age"], bins=10, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Patients")

plt.show()

# Male Vs Female Destribution plot:
plt.figure(figsize=(6,4))
sns.countplot(x="sex", data = df)
plt.title("Gender Destribution")
plt.xlabel("sex ( 0 = Female, 1 = Male)")
plt.ylabel("Number of Patients")
plt.show()

# Heart Disease By Gender plot:
plt.figure(figsize=(6,4))
sns.countplot(x = "sex", hue = "target", data = df)
plt.title("Heart Disease by Gender")
plt.xlabel("Sex (0 = Female, 1 = Male)")
plt.ylabel("Number of Patients")
plt.legend(["No Disease", "Heart Disease"])
plt.show()

# Correlation Heatmap plot:
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap = "coolwarm", fmt = ".2f")
plt.title("Correlation HeatMap")
plt.show()
# -------

# Correlation shows the relationship between two columns.
# Values range from -1 to +1.
# +1 = Positive Relation
#  0 = No Relation
# -1 = Negative Relation
correlation = df.corr()
print(correlation)

# Separate features(x) and labels(y):
x = df.drop("target", axis = 1)
y = df["target"]
print("Features shpae: ", x.shape)
print("Label shape: ", y.shape)
print("\nFeatures: ", x.head())
print("Labels: ", y.head())

# Train Test Split:
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size = 0.2,
    random_state = 42
)
# print("Training Features shape: ",x_train.shape,)------
# print("Testing Features shape: ", x_test.shape)----------
# print("Training Target Shape: ", y_train.shape)--------
# print("Testing Target shape: ", y_test.shape)-----------

# ab hum sab testing and traning features ko ek hi scale par lainge:
from sklearn.preprocessing import StandardScaler
# create Standard Scalar object:
scale = StandardScaler()

# learn from training data and scale it:
x_train = scale.fit_transform(x_train)

# Only scale testing data:
x_test = scale.transform(x_test)
print("\nTraining data after Scaling: ", x_train[:5])
print("Testing data after Scaling: ",x_test[:5])


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)

# Make predictions on testing data:
y_pred = model.predict(x_test)
print("Actual Values: ")
print(y_test[:10])
print("\nPredicted Values: ")
print(y_pred[:10])

# B check karte hain k model nain kitne data par sahii prediction kii hain.
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)  # accuracy = (correct pred / total pred)
print("Model Accuracy percentage: ", accuracy*100)

from sklearn.metrics import confusion_matrix
# create confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: ", cm)

# Day 5
from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred)
print("Classification Report: ")
print(report)

from sklearn.tree import DecisionTreeClassifier
# Create tree model
dt_model = DecisionTreeClassifier(random_state=42)
# Now train the model
dt_model.fit(x_train, y_train)
print("Decision tree model train Successfully!")

from sklearn.metrics import accuracy_score
# model prediction:
dt_pred = dt_model.predict(x_test)
print("Actual Values: ")
print(y_test[:10])
print("\nDecision Tree Prediction: ")
print(dt_pred[:10])
# Calculate accuracy:
dt_accuracy = accuracy_score(y_test, dt_pred)
print("Decision Tree Accuracy: ", dt_accuracy)

from sklearn.ensemble import RandomForestClassifier
# create random forest model:
rf_model = RandomForestClassifier(random_state=42)
# Train th modell:
rf_model.fit(x_train, y_train)
print("Random forest model train Sucessfully!")

from sklearn.metrics import accuracy_score

# Make predictions
rf_pred = rf_model.predict(x_test)

print("Actual Values:")
print(y_test[:10])

print("\nRandom Forest Predictions:")
print(rf_pred[:10])

# Calculate accuracy
rf_accuracy = accuracy_score(y_test, rf_pred)

print("\nRandom Forest Accuracy:", rf_accuracy)

print("\n===== Model Comparison =====")

print(f"Logistic Regression Accuracy : {accuracy:.2%}")
print(f"Decision Tree Accuracy       : {dt_accuracy:.2%}")
print(f"Random Forest Accuracy       : {rf_accuracy:.2%}")

results = {
    "Logistic Regression": accuracy,
    "Decision Tree": dt_accuracy,
    "Random Forest": rf_accuracy
}

best_model = max(results, key=results.get)

print("\nBest Model :", best_model)
print("Best Accuracy :", results[best_model])

models = list(results.keys())
scores = list(results.values())

plt.figure(figsize=(8,5))
plt.bar(models, scores)

plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")

plt.ylim(0,1)

plt.show()

print("\n===== Final Result =====")

if best_model == "Random Forest":
    print("Random Forest performed best.")
elif best_model == "Decision Tree":
    print("Decision Tree performed best.")
else:
    print("Logistic Regression performed best.")

import joblib
joblib.dump(rf_model, "heart_model.pkl")
joblib.dump(scale, "scaler.pkl")

print("\nModel saved successfully!")
print("Scaler saved successfully!")

print("\n===== Loading Saved Model =====")

loaded_model = joblib.load("heart_model.pkl")
loaded_scaler = joblib.load("scaler.pkl")

print("Model Loaded Successfully!")
print("Scaler Loaded Successfully!")

new_patient = [[
    52,   # age
    1,    # sex
    0,    # cp
    125,  # trestbps
    212,  # chol
    0,    # fbs
    1,    # restecg
    168,  # thalach
    0,    # exang
    1.0,  # oldpeak
    2,    # slope
    2,    # ca
    3     # thal
]]

new_patient = loaded_scaler.transform(new_patient)
prediction = loaded_model.predict(new_patient)

print("\nPrediction:", prediction)
if prediction[0] == 1:
    print("Patient is likely to have Heart Disease.")
else:
    print("Patient is not likely to have Heart Disease.")