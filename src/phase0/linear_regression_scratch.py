import logging

import tensorflow as tf

class LinearRegressionScratch:
    def __init__(self):
        self.W = tf.Variable(
            tf.random.normal([1]),
            trainable=True
        )
        self.b = tf.Variable(
            tf.random.normal([1]),
            trainable=True
        )
    
    def predict(self, x):
        return self.W * x + self.b
    
    def loss(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))

    def train(self, x, y, epochs=300, lr=0.01):
        opt = tf.keras.optimizers.SGD(learning_rate=lr)

        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                y_pred = self.predict(x)
                loss_value = self.loss(y, y_pred)

            grads = tape.gradient(loss_value, [self.W, self.b])
            opt.apply_gradients(zip(grads, [self.W, self.b]))
        
            if epoch % 50 == 0:
                logging.info('Epoch %d: Loss: %.4f, W: %.4f, b: %.4f',
                             epoch, loss_value.numpy(), 
                             self.W.numpy()[0], 
                             self.b.numpy()[0]
                             )


def main():
    
    x = tf.constant([1., 2., 3., 4.])
    y = tf.constant([3., 5., 7., 9.])

    model = LinearRegressionScratch()
    model.train(x, y, epochs=300, lr=0.01)

    x_test = tf.constant([5., 6.])
    y_pred = model.predict(x_test)
    logging.info('Predictions for x_test [5., 6.]: %s', y_pred.numpy())


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()