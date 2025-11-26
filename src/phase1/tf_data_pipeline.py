import logging

import numpy as np
import tensorflow as tf

def main():
    x = np.random.rand(1000, 20)
    y = np.random.randint(0, 2, size=(1000,))

    pipeline = TFDataPipeline(x, y)
    ds = pipeline.load_numpy()

    for batch in ds.take(1):
        logging.info(f"Batch shape: {batch[0].shape}, {batch[1].shape}")
        # print("Batch data:", batch)


class TFDataPipeline:
    def __init__(self, x: np.ndarray, y: np.ndarray): 
        self.x = x
        self.y = y

    def load_numpy(self):
        # Create a TensorFlow Dataset from numpy arrays
        ds = tf.data.Dataset.from_tensor_slices((self.x, self.y))
        # Apply transformations: shuffle, batch, prefetch
        ds = ds.shuffle(buffer_size=1024).batch(32).prefetch(tf.data.AUTOTUNE)

        return ds


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()