import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from Layer import Layer
from NeuralNetwork import NeuralNetwork, binary_cross_entropy
from Optimizers import SGD, RMSprop, Adam

# Dataset
# Iris: 150 samples, 4 features, 3 classes.
# We simplify to binary: Setosa (0) vs Not-Setosa (1)
# Setosa is linearly separable — a good first real test.

iris          = load_iris()
X_all         = iris.data.astype(float)           # (150, 4)
y_all         = (iris.target > 0).astype(float).reshape(-1, 1)  # binary

scaler        = StandardScaler()
X_all         = scaler.fit_transform(X_all)

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

print(f"Dataset   : Iris (binary — Setosa vs Not-Setosa)")
print(f"Features  : {X_train.shape[1]}  |  Train: {len(X_train)}  |  Test: {len(X_test)}")
print(f"Class balance — train 1s: {y_train.mean():.0%}  test 1s: {y_test.mean():.0%}\n")

# Train all three optimisers

def make_layers():
    return [
        Layer(4, 8,  activation='relu'),
        Layer(8, 4,  activation='relu'),
        Layer(4, 1,  activation='sigmoid'),
    ]

EPOCHS     = 300
BATCH_SIZE = 16

configs = [
    ("SGD + Momentum", SGD(lr=0.05,  momentum=0.9)),
    ("RMSprop",        RMSprop(lr=0.001)),
    ("Adam",           Adam(lr=0.001)),
]

histories = {}
accuracies = {}

for name, opt in configs:
    print(f"{'─'*48}\n  {name}\n{'─'*48}")
    net     = NeuralNetwork(make_layers(), optimizer=opt)
    history = net.train(X_train, y_train,
                        epochs=EPOCHS, batch_size=BATCH_SIZE, print_every=100)
    preds   = (net.predict(X_test) > 0.5).astype(float)
    acc     = (preds == y_test).mean()
    histories[name]  = history
    accuracies[name] = acc
    print(f"  Test accuracy : {acc*100:.1f}%\n")

# Summary table

print(f"{'─'*48}")
print(f"  {'Optimiser':<18} {'Start':>8} {'End':>8} {'Acc':>8}")
print(f"{'─'*48}")
for name, hist in histories.items():
    print(f"  {name:<18} {hist[0]:>8.4f} {hist[-1]:>8.4f} "
          f"{accuracies[name]*100:>7.1f}%")
print(f"{'─'*48}\n")

# Plot

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor('#F1EFE8')
for ax in axes:
    ax.set_facecolor('#F1EFE8')

COLORS = {
    "SGD + Momentum": "#3B8BD4",
    "RMSprop":        "#1D9E75",
    "Adam":           "#D85A30",
}

# Left: loss curves
ax = axes[0]
for name, hist in histories.items():
    ax.plot(range(1, len(hist)+1), hist,
            label=name, color=COLORS[name], linewidth=2)

ax.set_title("Loss curve — Iris dataset", fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("BCE Loss", fontsize=11)
ax.legend(fontsize=10)
ax.spines[['top','right']].set_visible(False)
ax.grid(axis='y', linestyle='--', alpha=0.4)

# Right: final accuracy bar chart
ax = axes[1]
names = list(accuracies.keys())
vals  = [accuracies[n] * 100 for n in names]
bars  = ax.bar(names, vals,
               color=[COLORS[n] for n in names],
               width=0.5, zorder=3)

for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.5,
            f"{val:.1f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_title("Test accuracy by optimiser", fontsize=13, fontweight='bold', pad=12)
ax.set_ylabel("Accuracy (%)", fontsize=11)
ax.set_ylim(80, 105)
ax.spines[['top','right']].set_visible(False)
ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)

plt.suptitle("From-scratch neural net · Iris (binary)",
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/results.png', dpi=150, bbox_inches='tight')
print("Plot saved → results.png")