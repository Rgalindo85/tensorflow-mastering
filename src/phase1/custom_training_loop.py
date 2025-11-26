import tensorflow as tf

def main():
    x = tf.random.normal((500, 10))
    y = tf.random.normal((500, 1))

    trainer = CustomTrainingLoop()
    trainer.train(x, y, epochs=5, batch_size=16)

class CustomTrainingLoop:
    def __init__(self):
        # Define a simple linear regression model
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
            tf.keras.layers.Dense(1)
        ])
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

    def train(self, x, y, epochs: int=5, batch_size: int=16):
        # Create a TensorFlow Dataset
        dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(buffer_size=1024).batch(batch_size)

        # Custom training loop
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")

            # Iterate over batches
            for step, (batch_x, batch_y) in enumerate(dataset):
                with tf.GradientTape() as tape:
                    predictions = self.model(batch_x, training=True)
                    loss = self.loss_fn(batch_y, predictions)
                
                # Compute and apply gradients
                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

                if step % 10 == 0:
                    print(f"Step {step}, Loss: {loss.numpy():.4f}")

if __name__ == "__main__":
    main()