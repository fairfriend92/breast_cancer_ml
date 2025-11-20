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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score, auc, roc_auc_score, roc_curve)

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
    
    # Base models
    lr = LogisticRegression(max_iter=1000)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    svc = SVC(probability=True, random_state=42)
    
    # Voting ensemble
    voting = VotingClassifier(
        estimators=[('lr', lr), ('rf', rf), ('svc', svc)],
        voting='soft'  # soft = use probabilities to choose among predictions
    )

    # Dictionary containing model definitions inside pipelines
    pipelines = {
        "LogisticRegression": Pipeline([("scaler", StandardScaler()), ("clf", lr)]),
        "RandomForest": Pipeline([("clf", rf)]), 
        "SVM": Pipeline([("scaler", StandardScaler()),("clf", svc)]), 
        "Voting": Pipeline([('scaler', StandardScaler()), ('clf', voting)])        
    }
    
    # Save directories
    models_dir = Path("../models")
    models_dir.mkdir(parents=True, exist_ok=True)
    cm_dir = Path("../outputs/confusion_matrices")
    cm_dir.mkdir(parents=True, exist_ok=True)
    roc_dir = Path("../outputs/roc_curves")
    roc_dir.mkdir(parents=True, exist_ok=True)
    
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
        auc_score = roc_auc_score(y_test, y_prob)
        
        print(f"\n{name} - Accuracy: {acc:.4f}, F1-score: {f1:.4f}, AUC: {auc_score:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=data.target_names))
        
        # Save models 
        model_path = models_dir / f"{name}.joblib"        
        joblib.dump(pipe, model_path)
        print(f"Saved trained model to: {model_path.resolve()}")
        
        # Plot and save confusion matrices 
        cm_path = cm_dir / f"{name}_cm.png"
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(cm_path)
        plt.close()
        print(f"Saved confusion matrix to: {cm_path.resolve()}")

    # Compare ROC curves 
    plt.figure(figsize=(6,5))
    for name, pipe in pipelines.items():
        y_prob = pipe.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.savefig(roc_dir / "roc_curve_all_models.png")
    plt.close()

if __name__ == "__main__":
    main()
