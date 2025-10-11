# prescher_martin_hw4_prob4_simple.py
# DSCI-352, Fall 2025

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss, accuracy_score

# -------------------------
# 1) Data
# -------------------------
X, y = make_moons(n_samples=4000, noise=0.20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test  = scaler.transform(X_test)

# -------------------------
# 0) Plot the raw data (before training)
# -------------------------
plt.figure(figsize=(6,5))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor="k", s=12)
plt.title("make_moons dataset (n=4000, noise=0.20)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.tight_layout()
plt.savefig("prob4_simple_data.png", dpi=150)
plt.close()


# -------------------------
# 2) Two deep MLPs: logistic vs ReLU
#    (same depth/width, same optimizer & LR)
# -------------------------
hidden = (128, 128, 128, 128, 128, 128)  # 6 hidden layers
lr = 1e-2
epochs = 40
batch_size = 128

def make_mlp(activation):
    return MLPClassifier(
        hidden_layer_sizes=hidden,
        activation=activation,         # "logistic" (sigmoid) or "relu"
        solver="adam",
        alpha=0.0,
        batch_size=batch_size,
        learning_rate_init=lr,
        max_iter=1,                   # do 1 epoch per .fit call
        shuffle=True,
        random_state=123,
        warm_start=True,              #warm_start=True lets us fit for 1 epoch repeatedly
        verbose=False
    )

mlp_sig = make_mlp("logistic")
mlp_relu = make_mlp("relu")
# IMPORTANT: make_mlp implicitly initializes the weights already by drawing them from a uniform
# distribution that's dependent on the network architecture (called Glorot/Xavier initialization)

# -------------------------
# 3) Training loop
#    - Call .fit() for one epoch at a time
#    - Track: loss (log loss), accuracy, and weight-change norm per epoch
# -------------------------
def train_over_epochs(model, X_tr, y_tr, X_te, y_te, epochs):
    losses_tr, losses_te = [], []
    accs_tr, accs_te = [], []
    weight_change_norms = []
    weight_norms = []          # track total weight magnitude per epoch

    prev_coefs = None

    for ep in range(epochs):
        model.fit(X_tr, y_tr)  # 1 iteration because max_iter=1 and warm_start=True

        # Predictions
        proba_tr = model.predict_proba(X_tr)
        proba_te = model.predict_proba(X_te)
        yhat_tr  = np.argmax(proba_tr, axis=1)
        yhat_te  = np.argmax(proba_te, axis=1)

        # Metrics
        from sklearn.metrics import log_loss, accuracy_score
        losses_tr.append(log_loss(y_tr, proba_tr))
        losses_te.append(log_loss(y_te, proba_te))
        accs_tr.append(accuracy_score(y_tr, yhat_tr))
        accs_te.append(accuracy_score(y_te, yhat_te))

        # Weight-move proxy (sum of L2 norms of ΔW for all layers this epoch)
        if prev_coefs is None:
            weight_change_norms.append(0.0)  # first epoch baseline
        else:
            total = 0.0
            for W_prev, W_now in zip(prev_coefs, model.coefs_):
                diff = W_now - W_prev
                # linalg package gives computations like 'norm', or absolute value:
                total += np.linalg.norm(diff)
            weight_change_norms.append(total)

        # Total weight magnitude this epoch
        total_W_norm = sum(np.linalg.norm(W) for W in model.coefs_)
        weight_norms.append(total_W_norm)

        # store copy to compare with next epoch
        prev_coefs = [W.copy() for W in model.coefs_]

    return {
        "loss_tr": losses_tr,
        "loss_te": losses_te,
        "acc_tr": accs_tr,
        "acc_te": accs_te,
        "w_move": weight_change_norms,
        "w_norms": weight_norms,
        "model": model
    }


hist_sig  = train_over_epochs(mlp_sig,  X_train, y_train, X_test, y_test, epochs)
hist_relu = train_over_epochs(mlp_relu, X_train, y_train, X_test, y_test, epochs)

# -------------------------
# 4) Print final results
# -------------------------
print("\n=== Final Results ===")
print(f"Sigmoid  -> Train acc: {hist_sig['acc_tr'][-1]:.3f}, Test acc: {hist_sig['acc_te'][-1]:.3f}, Test loss: {hist_sig['loss_te'][-1]:.3f}")
print(f"ReLU     -> Train acc: {hist_relu['acc_tr'][-1]:.3f}, Test acc: {hist_relu['acc_te'][-1]:.3f}, Test loss: {hist_relu['loss_te'][-1]:.3f}")

# -------------------------
# 5) Plots (loss, accuracy, weight move)
# -------------------------
epochs_axis = np.arange(1, epochs + 1)

# Loss curves
plt.figure(figsize=(7,5))
plt.plot(epochs_axis, hist_sig["loss_tr"], label="Train (Sigmoid)")
plt.plot(epochs_axis, hist_sig["loss_te"], label="Test (Sigmoid)")
plt.plot(epochs_axis, hist_relu["loss_tr"], label="Train (ReLU)")
plt.plot(epochs_axis, hist_relu["loss_te"], label="Test (ReLU)")
plt.xlabel("Epoch")
plt.ylabel("Log loss")
plt.title("Loss vs Epochs (Sigmoid vs ReLU)")
plt.legend()
plt.tight_layout()
plt.savefig("prob4_simple_loss.png", dpi=150)
plt.close()

# Accuracy curves
plt.figure(figsize=(7,5))
plt.plot(epochs_axis, hist_sig["acc_tr"], label="Train (Sigmoid)")
plt.plot(epochs_axis, hist_sig["acc_te"], label="Test (Sigmoid)")
plt.plot(epochs_axis, hist_relu["acc_tr"], label="Train (ReLU)")
plt.plot(epochs_axis, hist_relu["acc_te"], label="Test (ReLU)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epochs (Sigmoid vs ReLU)")
plt.legend()
plt.tight_layout()
plt.savefig("prob4_simple_accuracy.png", dpi=150)
plt.close()

# Weight-change proxy (how much weights moved each epoch)
plt.figure(figsize=(7,5))
plt.plot(epochs_axis, hist_sig["w_move"], label="Weight move (Sigmoid)")
plt.plot(epochs_axis, hist_relu["w_move"], label="Weight move (ReLU)")
plt.xlabel("Epoch")
plt.ylabel("Sum of L2 norms of Δweights")
plt.title("Weight-Change Proxy per Epoch")
plt.legend()
plt.tight_layout()
plt.savefig("prob4_simple_weightmove.png", dpi=150)
plt.close()

# Total weight magnitude per epoch (good for spotting explosion)
plt.figure(figsize=(7,5))
plt.plot(epochs_axis, hist_sig["w_norms"], label="Total weight norm (Sigmoid)")
plt.plot(epochs_axis, hist_relu["w_norms"], label="Total weight norm (ReLU)")
plt.xlabel("Epoch"); plt.ylabel("Sum of L2 norms of weights")
plt.title("Total Weight Magnitude per Epoch")
plt.legend()
plt.tight_layout()
plt.savefig("prob4_simple_weightnorms.png", dpi=150)
plt.close()

print("\nSaved figures:")
print(" - prob4_simple_loss.png")
print(" - prob4_simple_accuracy.png")
print(" - prob4_simple_weightmove.png")
print("Take screenshots of these plots plus the printed final results.")
