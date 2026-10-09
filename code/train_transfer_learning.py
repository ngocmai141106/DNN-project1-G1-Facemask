import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    precision_recall_curve
)

# ==========================================
# 1. Reproducibility & Environment Setup
# ==========================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print(f"TensorFlow Version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU Available: {gpus}")
else:
    print("Running on CPU.")

# ==========================================
# 2. Dataset Paths & Parameters
# ==========================================
# Matches 01_data_explore.ipynb and 02_preprocess.ipynb
if os.path.exists("../data"):
    BASE_DIR = "../data"
elif os.path.exists("./data"):
    BASE_DIR = "./data"
else:
    BASE_DIR = "../data"

TRAIN_DIR = os.path.join(BASE_DIR, "Train")
VAL_DIR = os.path.join(BASE_DIR, "Validation")
TEST_DIR = os.path.join(BASE_DIR, "Test")

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

print(f"Loading data from: {BASE_DIR}")

# ==========================================
# 3. Load Datasets via tf.data
# ==========================================
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    color_mode="rgb",
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    color_mode="rgb",
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    color_mode="rgb",
    shuffle=False
)

class_names = train_ds.class_names
print(f"Classes found ({len(class_names)}): {class_names}")

# Data Augmentation pipeline
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
], name="data_augmentation")

AUTOTUNE = tf.data.AUTOTUNE

# Optimize pipeline with prefetching & autotune
def prepare_train(ds):
    return (
        ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
          .cache()
          .prefetch(buffer_size=AUTOTUNE)
    )

def prepare_val_test(ds):
    return (
        ds.cache()
          .prefetch(buffer_size=AUTOTUNE)
    )

train_ds_ready = prepare_train(train_ds)
val_ds_ready = prepare_val_test(val_ds)
test_ds_ready = prepare_val_test(test_ds)

# ==========================================
# 4. Model Architecture (Transfer Learning)
# ==========================================
base_model = MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # Freeze backbone for Phase 1

inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="input_image")
# MobileNetV2 expects pixel values in [-1, 1]
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(1, activation="sigmoid", name="predictions")(x)

model = models.Model(inputs=inputs, outputs=outputs, name="FaceMask_MobileNetV2")
model.summary()

# ==========================================
# 5. Phase 1: Feature Extraction
# ==========================================
print("\n" + "=" * 50)
print("PHASE 1: Training Classification Head (Feature Extraction)")
print("=" * 50)

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

callbacks_p1 = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=4, restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6, verbose=1
    )
]

history_phase1 = model.fit(
    train_ds_ready,
    validation_data=val_ds_ready,
    epochs=10,
    callbacks=callbacks_p1
)

# ==========================================
# 6. Phase 2: Fine-Tuning
# ==========================================
print("\n" + "=" * 50)
print("PHASE 2: Fine-Tuning Top Layers of MobileNetV2")
print("=" * 50)

# Unfreeze the top layers of the backbone (last 30 layers)
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompile with low learning rate to avoid catastrophic forgetting
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

callbacks_p2 = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=4, restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=2, min_lr=1e-7, verbose=1
    )
]

history_finetune = model.fit(
    train_ds_ready,
    validation_data=val_ds_ready,
    epochs=10,
    callbacks=callbacks_p2
)

# ==========================================
# 7. Comprehensive Evaluation: F1, Precision, Recall, mAP50
# ==========================================
print("\n" + "=" * 50)
print("EVALUATION ON TEST SET")
print("=" * 50)

# Extract all true labels and predictions
y_true_list = []
for _, labels in test_ds:
    y_true_list.extend(labels.numpy().flatten())
y_true = np.array(y_true_list, dtype=int)

y_pred_probs = model.predict(test_ds).flatten()
# Decision threshold = 0.50
y_pred = (y_pred_probs >= 0.5).astype(int)

# Precision, Recall, F1 at threshold = 0.50
precision_weighted = precision_score(y_true, y_pred, average="weighted")
recall_weighted = recall_score(y_true, y_pred, average="weighted")
f1_weighted = f1_score(y_true, y_pred, average="weighted")

precision_binary = precision_score(y_true, y_pred, average="binary")
recall_binary = recall_score(y_true, y_pred, average="binary")
f1_binary = f1_score(y_true, y_pred, average="binary")

# Classification mAP: Mean Average Precision / Area Under PR-Curve
# AP for class 1 (WithoutMask)
ap_class1 = average_precision_score(y_true, y_pred_probs)
# AP for class 0 (WithMask): invert target and probabilities
ap_class0 = average_precision_score(1 - y_true, 1 - y_pred_probs)
# Mean AP across both classes
mAP = (ap_class0 + ap_class1) / 2.0

print(f"\nEvaluation Results on Test Set ({len(y_true)} images):")
print("-" * 50)
print(f"Decision Threshold: 0.50")
print(f"Weighted Precision: {precision_weighted:.4f}")
print(f"Weighted Recall:    {recall_weighted:.4f}")
print(f"Weighted F1-Score:  {f1_weighted:.4f}")
print("-" * 50)
print(f"Average Precision (AP) for '{class_names[0]}': {ap_class0:.4f}")
print(f"Average Precision (AP) for '{class_names[1]}': {ap_class1:.4f}")
print(f"mAP (Mean Average Precision / PR-AUC):       {mAP:.4f}")
print("-" * 50)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

print("Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
print(cm)

# Save model
os.makedirs("../models", exist_ok=True)
model_path = "../models/facemask_mobilenetv2_finetuned.keras"
model.save(model_path)
print(f"\nModel saved successfully at: {model_path}")

# ==========================================
# 8. Plot Precision-Recall Curves
# ==========================================
prec0, rec0, _ = precision_recall_curve(1 - y_true, 1 - y_pred_probs)
prec1, rec1, _ = precision_recall_curve(y_true, y_pred_probs)

plt.figure(figsize=(8, 6))
plt.plot(rec0, prec0, label=f"{class_names[0]} (AP = {ap_class0:.4f})", linewidth=2)
plt.plot(rec1, prec1, label=f"{class_names[1]} (AP = {ap_class1:.4f})", linewidth=2)
plt.title(f"Precision-Recall Curve (mAP = {mAP:.4f})", fontsize=14)
plt.xlabel("Recall", fontsize=12)
plt.ylabel("Precision", fontsize=12)
plt.legend(loc="lower left", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.7)

os.makedirs("../reports", exist_ok=True)
plot_path = "../reports/pr_curve_test.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
print(f"Precision-Recall curve saved at: {plot_path}")
