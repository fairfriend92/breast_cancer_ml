import os                                                        
from pathlib import Path  
import joblib 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score, roc_auc_score)

def main():
    print("Starting training...")

    # Load dataset
    data = load_breast_cancer(as_frame=True)
    ''' 
        X is a matrix (n_samples, n_features) 
        Rows are samples, i.e. patients
        Columns are features like "radius mean", "texture mean" etc.
    '''
    X = data.data   
    '''
        For each patient a target can be either 0 or 1
        0 = malignant tumor
        1 = benign
    '''
    y = data.target 

    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Classes distribution: {y.value_counts().to_dict()}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2,    # 20% test 80% train
        stratify=y,             # Same malignant/benign ratio in test and train
        random_state=42         # Fixed seed to reproduce results
    )

    # Dictionary containing model definitions inside pipelines
    pipelines = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "RandomForest": Pipeline([
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True, random_state=42))
        ])
    }
    
    # Save directories
    models_dir = Path("../models")
    models_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path("../outputs/confusion_matrices")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train and evaluate
    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)    

        y_pred = pipe.predict(X_test) 
        y_prob = pipe.predict_proba(X_test)[:, 1] # Get prediction probs only for class 1 (malignant)
        cm = confusion_matrix(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred) 
        
        '''
            F1-score = 2 * (TP_precision*TP_recall)/(TP_precision+TP_recall)
            TP_precision: TP/(TP+FP)  
            TP_recall: TP/(TP+FN)
            TP: true positive FP: false positive FN: false negative 
        '''        
        f1 = f1_score(y_test, y_pred)
        
        '''
            ROC: Receiver Operating Characteristic. Curve with Y=TP_rate
            and X=FP_rate. Each point is obtained for a different threshold. 
            The prediction probability must be compared to the threshold to obtain 
            a binary prediction, from which to compute TP, TN, FP, FN. 
            
            TP_rate: TP/(TP+FN)
            FP_rate: FP/(FP+TN)
            
            AUC: Area Under the Curve. The greater the area, the better the classifier.            
        '''
        auc = roc_auc_score(y_test, y_prob)
        
        print(f"{name} - Accuracy: {acc:.4f}, F1-score: {f1:.4f}, AUC: {auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=data.target_names))
        
        # Save models 
        model_path = models_dir / f"{name}.joblib"        
        joblib.dump(pipe, model_path)
        print(f"Saved trained model to: {model_path.resolve()}")
        
        # Plot and save confusion matrices 
        output_path = output_dir / f"{name}_confusion_matrix.png"
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(output_path)
        plt.close()

if __name__ == "__main__":
    main()
