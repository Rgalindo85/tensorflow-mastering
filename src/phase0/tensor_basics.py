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
        print('Addition Result:\n', addition_result.numpy())
        print('Subtraction Result:\n', subtraction_result.numpy())
        print('Multiplication Result:\n', multiplication_result.numpy())

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
        print('Gradient dy/dx at x=3:', dy_dx.numpy())

    def reshape_example(self):
        reshaped = tf.reshape(self.a, [4, 1])
        print('Reshaped Matrix A:\n', reshaped.numpy())
    

    def broadcast_example(self):

        result = self.a + tf.constant([1., 2.])
        print('Broadcasting Result:\n', result.numpy())

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