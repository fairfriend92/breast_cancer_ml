import os                                                        
from pathlib import Path  
import joblib 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

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

    # Pipeline with StandardScaler + LogisticRegression
    pipe = make_pipeline(
        StandardScaler(), # Scale features so that they all have mu=0 std=1
        LogisticRegression(max_iter=1000, random_state=42)
    )

    # Train
    pipe.fit(X_train, y_train)

    # Evaluate
    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, preds, target_names=data.target_names))

    # Save model
    models_dir = Path("../models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "logreg_breast_cancer.joblib"
    joblib.dump(pipe, model_path)
    print(f"Saved trained model to: {model_path.resolve()}")

if __name__ == "__main__":
    main()
