import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from tf_keras.layers import Dense, Input
from tf_keras.models import Model
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
print("Loading Universal Sentence Encoder (this may take a minute the first time)...")

use_layer = hub.KerasLayer(
    "https://tfhub.dev/google/universal-sentence-encoder/4",
    input_shape=[],
    dtype=tf.string,
    trainable=False
)

inputs = Input(shape=[], dtype=tf.string)
x = use_layer(inputs)
x = Dense(16, activation="relu")(x)
outputs = Dense(1, activation="sigmoid")(x)

model = Model(inputs, outputs)

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
    batch_size=16
)

model.save("data/event_relevance_model_use.keras")
print("\nModel saved to data/event_relevance_model_use.keras")

final_train_acc = history.history["accuracy"][-1]
final_val_acc = history.history["val_accuracy"][-1]
print(f"\nFinal training accuracy: {final_train_acc:.3f}")
print(f"Final validation accuracy: {final_val_acc:.3f}")

best_val_acc = max(history.history["val_accuracy"])
print(f"Best validation accuracy across all epochs: {best_val_acc:.3f}")