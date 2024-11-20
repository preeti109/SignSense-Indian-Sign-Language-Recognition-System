import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam

# Define the number of classes
num_classes = 37

# Directory paths for training and validation data
train_dir = r"splitedDataSet\train"
val_dir = r"splitedDataSet\val"

# Automatically determine the number of classes based on dataset structure
num_classes = len(os.listdir(train_dir))  # Counts subdirectories in train_dir
print(f"Number of classes detected: {num_classes}")


# Model architecture
def create_cnn_model(input_shape=(64, 64, 3), num_classes=37, dropout_rate=0.5):
    print("\nCreating CNN model...")
    model = Sequential(
        [
            Conv2D(
                32, (3, 3), activation="relu", input_shape=input_shape, name="Conv2D_1"
            ),
            MaxPooling2D(2, 2, name="MaxPooling2D_1"),
            Conv2D(64, (3, 3), activation="relu", name="Conv2D_2"),
            MaxPooling2D(2, 2, name="MaxPooling2D_2"),
            Conv2D(128, (3, 3), activation="relu", name="Conv2D_3"),
            MaxPooling2D(2, 2, name="MaxPooling2D_3"),
            Flatten(name="Flatten"),
            Dense(128, activation="relu", name="Dense_1"),
            Dropout(dropout_rate, name="Dropout"),
            Dense(num_classes, activation="softmax", name="Output"),
        ]
    )
    return model


# Compile the model
def compile_model(model, learning_rate=0.001):
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    print(
        "\nModel compiled with Adam optimizer, categorical crossentropy loss, and accuracy metric."
    )


# Set up ImageDataGenerators
def create_data_generators():
    train_datagen = ImageDataGenerator(rescale=1.0 / 255)
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(64, 64), batch_size=32, class_mode="categorical"
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir, target_size=(64, 64), batch_size=32, class_mode="categorical"
    )

    print("Class indices:", train_generator.class_indices)
    return train_generator, val_generator


# Set up Callbacks (Early Stopping, Model Checkpoints, TensorBoard)
def setup_callbacks():
    early_stop = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    # Save the model with .keras extension as required by ModelCheckpoint
    checkpoint = ModelCheckpoint(
        "model.keras", monitor="val_loss", save_best_only=True, mode="min"
    )
    tensorboard = TensorBoard(log_dir="./logs")

    return [early_stop, checkpoint, tensorboard]


# Training the model
def train_model(model, train_generator, val_generator, epochs=10):
    callbacks = setup_callbacks()
    print(f"\nTraining model for {epochs} epochs...")

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
    )

    return history


# Save the trained model
def save_model(model, filepath="trainedModel.keras"):
    print(f"\nSaving model to {filepath}...")
    model.save(filepath)


# Main function to execute the workflow
def main():
    model = create_cnn_model(
        input_shape=(64, 64, 3), num_classes=num_classes, dropout_rate=0.5
    )

    compile_model(model, learning_rate=0.001)

    train_generator, val_generator = create_data_generators()

    history = train_model(model, train_generator, val_generator, epochs=10)

    save_model(model, filepath="trainedModel.keras")


if __name__ == "__main__":
    main()
