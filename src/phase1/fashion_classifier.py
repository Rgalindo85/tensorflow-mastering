import numpy as np
import tensorflow as tf

def main():
    datamanager = DataManager()
    (x_train, y_train), (x_test, y_test) = datamanager.load_data()

    classifier = FashionClassifier()
    classifier.train(x_train, y_train, x_test, y_test, epochs=5, batch_size=32)


class FashionClassifier:
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential([
            # Flatten the 28x28 images into a 784-dimensional vector
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            
            # Hidden layer: 128 neurons with ReLU activation
            # ReLU activation introduces non-linearity, allowing the model to learn complex patterns
            tf.keras.layers.Dense(128, activation='relu'),
            
            # Output layer: 10 neurons (one for each class) with softmax activation
            # Softmax activation converts logits to probabilities for multi-class classification
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        return model
    
    def train(self, x_train, y_train, x_test, y_test, epochs=5, batch_size=32):
        # Compile the model with Adam optimizer and sparse categorical crossentropy loss
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy']
                           )
        
        # Train the model on the training data
        self.model.fit(x_train, y_train, 
                       epochs=epochs, 
                       batch_size=batch_size,
                       validation_data=(x_test, y_test)
                       )


class DataManager():
    def load_data(self):
        fashion_mnist = tf.keras.datasets.fashion_mnist
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        
        # Normalize the images to [0, 1] range
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

        return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    main()