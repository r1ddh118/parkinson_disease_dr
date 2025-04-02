import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset
from ucimlrepo import fetch_ucirepo 

class ParkinsonsML:
    def __init__(self):
        self.parkinsons = fetch_ucirepo(id=174)
        self.X = self.parkinsons.data.features
        self.y = self.parkinsons.data.targets.values.ravel()
        self.scaler = StandardScaler()
        self.models = {
            "Random Forest": RandomForestClassifier(),
            "SVM": SVC()
        }
        self.param_grids = {
            "Random Forest": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]},
            "SVM": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
        }

    def preprocess_data(self):
        self.X = self.X.dropna().drop_duplicates()
        self.X_scaled = self.scaler.fit_transform(self.X)

    def feature_importance(self):
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(self.X_scaled, self.y)
        feature_importances = pd.Series(rf.feature_importances_, index=self.X.columns)
        self.selected_features = feature_importances.nlargest(10).index.tolist()

    def exploratory_data_analysis(self):
        plt.figure(figsize=(10, 5))
        sns.countplot(x=self.y, palette='coolwarm')
        plt.title("Class Distribution")
        plt.show()

        plt.figure(figsize=(10, 5))
        sns.boxplot(data=self.X[self.selected_features])
        plt.xticks(rotation=90)
        plt.title("Feature Distributions")
        plt.show()

        plt.figure(figsize=(10, 5))
        sns.violinplot(data=self.X[self.selected_features])
        plt.xticks(rotation=90)
        plt.title("Feature Density Distributions")
        plt.show()

    def train_test_split(self):
        self.X_selected = self.X[self.selected_features].copy()
        self.X_scaled_selected = self.scaler.fit_transform(self.X_selected)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled_selected, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

    def dimensionality_reduction_visualization(self):
        self.dim_reduction_methods = {
            "PCA": PCA(n_components=2).fit_transform(self.X_scaled_selected),
            "LDA": LDA(n_components=1).fit_transform(self.X_scaled_selected, self.y),
            "SVD": TruncatedSVD(n_components=2).fit_transform(self.X_scaled_selected),
            "t-SNE": TSNE(n_components=2, random_state=42).fit_transform(self.X_scaled_selected)
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for ax, (method, X_transformed) in zip(axes.ravel(), self.dim_reduction_methods.items()):
            sns.scatterplot(x=X_transformed[:, 0], y=X_transformed[:, 1] if X_transformed.shape[1] > 1 else np.zeros_like(X_transformed[:, 0]),
                            hue=pd.Series(self.y), palette='coolwarm', ax=ax)
            ax.set_title(f"{method} Visualization")
        plt.tight_layout()
        plt.show()

    def compare_dimensionality_reduction(self):
        best_method = {}
        
        for method, X_transformed in self.dim_reduction_methods.items():
            X_train_dr, X_test_dr, y_train_dr, y_test_dr = train_test_split(
                X_transformed, self.y, test_size=0.2, random_state=42, stratify=self.y)
            
            for name, model in self.models.items():
                model.fit(X_train_dr, y_train_dr)
                y_pred_dr = model.predict(X_test_dr)
                accuracy_dr = accuracy_score(y_test_dr, y_pred_dr) * 100
                
                if name not in best_method or best_method[name][1] < accuracy_dr:
                    best_method[name] = (method, accuracy_dr)
        
        for name, (method, acc) in best_method.items():
            print(f"Best Dimensionality Reduction for {name}: {method} with Accuracy: {acc:.2f}%")

    def model_training_and_tuning(self):
        for name, model in self.models.items():
            grid_search = GridSearchCV(model, self.param_grids[name], cv=StratifiedKFold(n_splits=5), scoring='accuracy', n_jobs=-1)
            grid_search.fit(self.X_train, self.y_train)
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred) * 100
            print(f"Model: {name}")
            print("Best Parameters:", grid_search.best_params_)
            print(f"Accuracy: {accuracy:.2f}%")
            print("Classification Report:\n", classification_report(self.y_test, y_pred))
            print("Confusion Matrix:")
            plt.figure(figsize=(6, 4))
            sns.heatmap(confusion_matrix(self.y_test, y_pred), annot=True, fmt='d', cmap='Blues')
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title(f"Confusion Matrix - {name}")
            plt.show()

if __name__ == "__main__":
    parkinsons_ml = ParkinsonsML()
    parkinsons_ml.preprocess_data()
    parkinsons_ml.feature_importance()
    parkinsons_ml.exploratory_data_analysis()
    parkinsons_ml.train_test_split()
    parkinsons_ml.dimensionality_reduction_visualization()
    parkinsons_ml.compare_dimensionality_reduction()
    parkinsons_ml.model_training_and_tuning()