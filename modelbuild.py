from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import json

num_classes = 37


def create_cnn_model(input_shape=(64, 64, 3), num_classes=37, dropout_rate=0.5):
    print(
        f"\nCreating CNN model with input shape {input_shape} and {num_classes} classes..."
    )
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


def compile_model(
    model, optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"]
):
    print("\nCompiling the model with the following configurations:")
    print(f"Optimizer: {optimizer}")
    print(f"Loss function: {loss}")
    print(f"Metrics: {metrics}")
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


def save_model_summary(model, filepath="modelSummary.txt"):
    print(f"\nSaving model summary to {filepath}...")
    with open(filepath, "w", encoding="utf-8") as file:
        model.summary(print_fn=lambda x: file.write(x + "\n"))


def save_model_config(model, filepath="modelConfig.json"):
    print(f"\nSaving model configuration to {filepath}...")
    config = model.to_json()
    with open(filepath, "w") as json_file:
        json.dump(json.loads(config), json_file, indent=4)


def main():
    input_shape = (64, 64, 3)
    dropout_rate = 0.5
    learning_rate = 0.001
    optimizer = Adam(learning_rate=learning_rate)

    print("\n--- Starting CNN Model Creation Process ---")
    print(
        f"Input Shape: {input_shape}, Dropout Rate: {dropout_rate}, Learning Rate: {learning_rate}"
    )

    model = create_cnn_model(
        input_shape=input_shape, num_classes=num_classes, dropout_rate=dropout_rate
    )

    compile_model(
        model,
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("\nModel Summary:")
    model.summary()

    save_model_summary(model)
    save_model_config(model)

    print("\n--- Model Creation Process Completed ---")
    return model


if __name__ == "__main__":
    cnn_model = main()
