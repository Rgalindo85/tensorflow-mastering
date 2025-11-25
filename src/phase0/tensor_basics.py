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