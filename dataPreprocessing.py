from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os


def validate_directories(train_dir, val_dir):
    if not os.path.exists(train_dir):
        raise ValueError(f"Training directory does not exist: {train_dir}")
    if not os.path.exists(val_dir):
        raise ValueError(f"Validation directory does not exist: {val_dir}")
    print(f"Training directory: {train_dir}")
    print(f"Validation directory: {val_dir}")


def create_datagen(
    rescale=1.0 / 255.0,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
):
    return ImageDataGenerator(
        rescale=rescale,
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        validation_split=validation_split,
    )


def create_data_generator(
    datagen, data_dir, target_size=(64, 64), batch_size=32, subset="training"
):
    return datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset=subset,
    )


def display_generator_info(generator, name="Generator"):
    print(f"\n--- {name} Information ---")
    print(f"Number of batches: {len(generator)}")
    print(f"Batch size: {generator.batch_size}")
    print(f"Image shape: {generator.image_shape}")
    print(f"Classes: {generator.class_indices}\n")


def main():
    train_dir = r"splitedDataSet\train"
    val_dir = r"splitedDataSet\val"
    target_size = (64, 64)
    batch_size = 32
    validation_split = 0.2

    validate_directories(train_dir, val_dir)

    print("\nCreating ImageDataGenerators with augmentation...")
    datagen = create_datagen(validation_split=validation_split)

    print("\nCreating Training Data Generator...")
    train_generator = create_data_generator(
        datagen,
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        subset="training",
    )
    display_generator_info(train_generator, "Training Data Generator")

    print("\nCreating Validation Data Generator...")
    val_generator = create_data_generator(
        datagen,
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        subset="validation",
    )
    display_generator_info(val_generator, "Validation Data Generator")

    print("\nGenerators created successfully.")
    return train_generator, val_generator


if __name__ == "__main__":
    train_gen, val_gen = main()
