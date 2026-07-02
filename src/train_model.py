import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.models import Sequential
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

df = pd.read_csv("data/training_data.csv")
df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")

texts = df["text"].values
labels = df["relevant"].astype(int).values

indices = np.arange(len(texts))
np.random.seed(42)
np.random.shuffle(indices)
texts = texts[indices]
labels = labels[indices]

split = int(0.8 * len(texts))
train_texts, val_texts = texts[:split], texts[split:]
train_labels, val_labels = labels[:split], labels[split:]

print(f"Training examples: {len(train_texts)}")
print(f"Validation examples: {len(val_texts)}")

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights = {i: w for i, w in enumerate(class_weights_array)}
print(f"Class weights: {class_weights}")

max_tokens = 5000
sequence_length = 100

vectorize_layer = TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=sequence_length
)
vectorize_layer.adapt(train_texts)

model = Sequential([
    vectorize_layer,
    Embedding(max_tokens, 16),
    GlobalAveragePooling1D(),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    train_texts, train_labels,
    validation_data=(val_texts, val_labels),
    epochs=20,
    batch_size=16,
    class_weight=class_weights
)

model.save("data/event_relevance_model.keras")
print("\nModel saved to data/event_relevance_model.keras")

final_train_acc = history.history["accuracy"][-1]
final_val_acc = history.history["val_accuracy"][-1]
print(f"\nFinal training accuracy: {final_train_acc:.3f}")
print(f"Final validation accuracy: {final_val_acc:.3f}")

if final_train_acc - final_val_acc > 0.15:
    print("Warning: possible overfitting, training accuracy much higher than validation.")