import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn import model_selection


class QuestionModel:

    def __init__(self):
        self.name = ''
        path = 'mexican_medical_students_mental_health_data.csv'
        self.df = pd.read_csv(path)
        self.df = self.df[['phq1', 'phq2', 'phq3', 'phq4',
                           'phq5', 'phq6', 'phq7', 'phq8', 'phq9']]

        # Handling Missing Data
        self.df['phq1'] = self.df['phq1'].fillna(self.df['phq1'].mode()[0])
        self.df['phq2'] = self.df['phq2'].fillna(self.df['phq2'].mode()[0])
        self.df['phq3'] = self.df['phq3'].fillna(self.df['phq3'].mode()[0])
        self.df['phq4'] = self.df['phq4'].fillna(self.df['phq4'].mode()[0])
        self.df['phq5'] = self.df['phq5'].fillna(self.df['phq5'].mode()[0])
        self.df['phq6'] = self.df['phq6'].fillna(self.df['phq6'].mode()[0])
        self.df['phq7'] = self.df['phq7'].fillna(self.df['phq7'].mode()[0])
        self.df['phq8'] = self.df['phq8'].fillna(self.df['phq8'].mode()[0])
        self.df['phq9'] = self.df['phq9'].fillna(self.df['phq9'].mode()[0])

    def split_data(self, df):
        x = df.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]].values
        y = df.iloc[:, 8].values
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.4, random_state=24)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        return x_train, y_train

    def svm_classifier(self):
        self.name = 'Svm Classifier'
        classifier = SVC()
        self.split_data(self.df)
        return classifier.fit(self.x_train, self.y_train)

    def decisionTree_classifier(self):
        self.name = 'Decision tree Classifier'
        classifier = DecisionTreeClassifier()
        return classifier.fit(self.x_train, self.y_train)

    def randomforest_classifier(self):
        self.name = 'Random Forest Classifier'
        classifier = RandomForestClassifier()
        return classifier.fit(self.x_train, self.y_train)

    def naiveBayes_classifier(self):
        self.name = 'Naive Bayes Classifier'
        classifier = GaussianNB()
        return classifier.fit(self.x_train, self.y_train)

    def knn_classifier(self):
        self.name = 'Knn Classifier'
        classifier = KNeighborsClassifier()
        return classifier.fit(self.x_train, self.y_train)

    def accuracy(self, model):
        predictions = model.predict(self.x_test)
        cm = confusion_matrix(self.y_test, predictions)
        cm_norm = cm.astype("float") / \
            cm.sum(axis=1)[:, np.newaxis]  # normalize it
        n_classes = cm.shape[0]
        accuracy = (cm[0][0] + cm[1][1]) / \
            (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
        print(f"{self.name} has accuracy of {accuracy *100} % ")
        print(cm)
        # Let's prettify it
        fig, ax = plt.subplots(figsize=(10, 10))
        # Create a matrix plot
        cax = ax.matshow(cm, cmap=plt.cm.Blues)
        fig.colorbar(cax)
        # Create classes
        classes = False

        if classes:
            labels = classes
        else:
            labels = np.arange(cm.shape[0])

        # Label the axes
        ax.set(title="Confusion Matrix",
               xlabel="Predicted label",
               ylabel="True label",
               xticks=np.arange(n_classes),
               yticks=np.arange(n_classes),
               xticklabels=labels,
               yticklabels=labels)

        # Set x-axis labels to bottom
        ax.xaxis.set_label_position("bottom")
        ax.xaxis.tick_bottom()

        # Adjust label size
        ax.xaxis.label.set_size(20)
        ax.yaxis.label.set_size(20)
        ax.title.set_size(20)

        # Set threshold for different colors
        threshold = (cm.max() + cm.min()) / 2.

        # Plot the text on each cell
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j] * 100:.1f}%)",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > threshold else "black",
                     size=15)
        # plt.show()


if __name__ == '__main__':
    model = QuestionModel()
    model.accuracy(model.svm_classifier())
    model.accuracy(model.decisionTree_classifier())
    model.accuracy(model.randomforest_classifier())
    model.accuracy(model.naiveBayes_classifier())
    model.accuracy(model.knn_classifier())
