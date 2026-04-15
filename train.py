# Simple ML training script using scikit-learn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import sys

print("Loading dataset...")
data = load_iris()
X, y = data.data, data.target

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")
model = RandomForestClassifier()
model.fit(X_train, y_train)

print("Evaluating model...")
preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)

print(f"Model Accuracy: {accuracy:.2f}")

# Simulate failure if accuracy is too low (for testing pipeline)
if accuracy > 1:
    print("Model accuracy too low! Failing job...")
    sys.exit(1)

print("Training completed successfully!")
# trigger success run