import logging

import tensorflow as tf

def main():

    # define matrices
    a = tf.constant([[1., 2.], [3., 4.]])
    b = tf.constant([[2., 0.], [1., 2.]])

    tb = TensorBasics(a, b)
    tb.run()


class TensorBasics:
    def __init__(self, a=None, b=None):
        self.a = a
        self.b = b

    def run(self):
        self.matrices_info()

        # matrix operations
        addition_result = self.matrices_addition()
        subtraction_result = self.matrices_subtraction()
        multiplication_result = self.matrices_multiplication() 
        logging.info('Addition Result:\n%s', addition_result.numpy())
        logging.info('Subtraction Result:\n%s', subtraction_result.numpy())
        logging.info('Multiplication Result:\n%s', multiplication_result.numpy())

        # broadcasting example
        self.broadcast_example()
        # reshaping example
        self.reshape_example()
        # gradient example
        self.gradient_example()

    def gradient_example(self):
        x = tf.Variable(3.0)

        with tf.GradientTape() as tape:
            y = x ** 2 + 2 * x + 1

        dy_dx = tape.gradient(y, x)
        logging.info('Gradient dy/dx at x=3: %s', dy_dx.numpy())

    def reshape_example(self):
        reshaped = tf.reshape(self.a, [4, 1])
        logging.info('Reshaped Matrix A:\n%s', reshaped.numpy())
    

    def broadcast_example(self):

        result = self.a + tf.constant([1., 2.])
        logging.info('Broadcasting Result:\n%s', result.numpy())

    def matrices_addition(self):
        return tf.add(self.a, self.b)
    
    def matrices_subtraction(self):
        return tf.subtract(self.a, self.b)

    def matrices_multiplication(self):
        return tf.matmul(self.a, self.b)

    def matrices_info(self):
        print('Matrix A:')
        print(self.a)
        print('Matrix B:')
        print(self.b)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()