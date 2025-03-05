import tensorflow as tf
from tensorflow.python.keras import layers, models, optimizers
import matplotlib.pyplot as plt


CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def create_dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    return x_train, y_train, x_test, y_test

def data_aug():
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

def cnn(input_shape=(32, 32, 3), num_classes=10):
    model = models.Sequential([
        data_aug(),
        
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def train_model(model, x_train, y_train, x_test, y_test):
    lr_schedule = optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=10000,
        decay_rate=0.9
    )
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr_schedule),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=3, 
            min_lr=1e-5
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', 
            patience=5, 
            restore_best_weights=True
        )
    ]
    
    stats = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        batch_size=64,
        epochs=20,
        callbacks=callbacks
    )
    
    return stats

def eval(model, x_test, y_test):
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
    
    predictions = model.predict(x_test)
    predicted_classes = tf.argmax(predictions, axis=1)
    true_classes = tf.argmax(y_test, axis=1)
    
    print("\nPer-class Accuracy:")
    for i in range(10):
        class_mask = true_classes == i
        class_accuracy = tf.reduce_mean(
            tf.cast(predicted_classes[class_mask] == true_classes[class_mask], tf.float32)
        )
        print(f"{CLASSES[i]}: {class_accuracy.numpy() * 100:.2f}%")

def vis_predict(model, x_test, y_test, num_images=5):
    predictions = model.predict(x_test)
    predicted_classes = tf.argmax(predictions, axis=1)
    true_classes = tf.argmax(y_test, axis=1)
    
    plt.figure(figsize=(15, 3))
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        plt.imshow(x_test[i])
        plt.title(f'Pred: {CLASSES[predicted_classes[i]]}\nTrue: {CLASSES[true_classes[i]]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_history(stats):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(stats.history['accuracy'])
    plt.plot(stats.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(stats.history['loss'])
    plt.plot(stats.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.tight_layout()
    plt.show()

def main():
    x_train, y_train, x_test, y_test = create_dataset()
    model = cnn()
    
    model.summary()
    
    print("\nStarting Training:")
    stats = train_model(model, x_train, y_train, x_test, y_test)
    
    print("\nModel Evaluation:")
    eval(model, x_test, y_test)
    
    plot_history(stats)
    
    print("\nVisualizing Predictions:")
    vis_predict(model, x_test, y_test)

if __name__ == '__main__':
    main()
