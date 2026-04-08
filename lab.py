import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---------------------------
# 1. Generate dataset
# ---------------------------
SEED = 42
np.random.seed(SEED)

X, y = make_classification(
    n_samples=500,
    n_features=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=SEED
)

# Train/test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ---------------------------
# 2. Train baseline model
# ---------------------------
baseline_model = LogisticRegression()
baseline_model.fit(X_train, y_train)

y_pred_baseline = baseline_model.predict(X_test)
baseline_accuracy = accuracy_score(y_test, y_pred_baseline)

print(f"[+] Baseline Accuracy: {baseline_accuracy:.4f}")

# ---------------------------
# 3. Poison the data (label flipping)
# ---------------------------
poisoned_y_train = y_train.copy()

# flip 15% of labels

# Target points near center (hardest to classify)
indices = np.where((X_train[:, 0] > -0.5) & (X_train[:, 0] < 0.5))[0]

# flip only those
poisoned_y_train[indices] = 1 - poisoned_y_train[indices]

# ---------------------------
# 4. Train poisoned model
# ---------------------------
poisoned_model = LogisticRegression()
poisoned_model.fit(X_train, poisoned_y_train)

y_pred_poisoned = poisoned_model.predict(X_test)
poisoned_accuracy = accuracy_score(y_test, y_pred_poisoned)

print(f"[!] Poisoned Accuracy: {poisoned_accuracy:.4f}")

# ---------------------------
# 5. Plot decision boundaries
# ---------------------------
def plot_boundary(model, X, y, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.title(title)
    plt.savefig("results.png")
    plt.close()


plot_boundary(
    baseline_model,
    X_train,
    y_train,
    f"Baseline Model (Acc: {baseline_accuracy:.2f})"
)

plot_boundary(
    poisoned_model,
    X_train,
    poisoned_y_train,
    f"Poisoned Model (Acc: {poisoned_accuracy:.2f})"
)