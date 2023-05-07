from sklearn.datasets import load_iris
import pandas as pd

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('IRIS.csv')

# Print the first few rows of the DataFrame
print(df.head())

# Load the Iris dataset from the DataFrame
iris = load_iris(as_frame=True)

# Print the dataset description
print(iris.DESCR)

iris = load_iris()
X = iris.data
y = iris.target


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


import matplotlib.pyplot as plt

# Visualize Sepal Length vs Sepal Width
plt.scatter(X[:,0], X[:,1], c=y)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()

# Visualize Petal Length vs Petal Width
plt.scatter(X[:,2], X[:,3], c=y)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.show()

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)


from sklearn.metrics import accuracy_score

y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}%'.format(accuracy*100))


new_data = [[5.1, 3.5, 1.4, 0.2], [6.3, 3.3, 4.7, 1.6], [7.2, 3.0, 5.8, 1.6]]
predictions = knn.predict(new_data)
print(predictions)


new_sepal_length = 5.1
new_sepal_width = 3.5
new_petal_length = 1.4
new_petal_width = 0.2


new_data = [[new_sepal_length, new_sepal_width, new_petal_length, new_petal_width]]
prediction = knn.predict(new_data)

print('Predicted Species: {}'.format(iris.target_names[prediction[0]]))
