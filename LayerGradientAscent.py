import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class LayerGradientAscent:
    def __init__(self, model, margin=5, size=64):
        self.model = model
        self.margin = margin
        self.size=  size

    @staticmethod
    def post_process_image(img):
        img -= tf.reduce_mean(img)
        img /= (tf.math.reduce_std(img) + 1e-5)
        img *= 0.1

        img += 0.5
        img = tf.clip_by_value(img, 0, 1)

        img *= 255
        img = tf.clip_by_value(img, 0, 255)
        img = tf.dtypes.cast(img, tf.uint8)
        return img

    @staticmethod
    def cal_grad(model, inp, filter_index=0):
        # Calculate gradient of model's output according to the input tensor
        with tf.GradientTape() as grad_tap:
            grad_tap.watch(inp)

            loss = tf.reduce_mean(model(inp)[:, :, :, filter_index])
            grad = grad_tap.gradient(loss, inp)
        grad = grad[0]
        # Normalization gradient
        grad /= (tf.sqrt(tf.reduce_mean(tf.square(grad))) + 1e-5)
        return grad

    def get_interested_feature(self, layer_name, filter_index=0, size=150, epochs=40):
        # Create temporary model
        temp_model = tf.keras.Model(inputs=self.model.input, outputs=self.model.get_layer(layer_name).output)
        # Create random input image
        inp_image = tf.random.uniform([1, size, size, 3])
        # Train input_image
        for _ in range(epochs):
            grad = self.cal_grad(temp_model, inp_image, filter_index=filter_index)
            inp_image += grad
        inp_image = inp_image[0]
        return self.post_process_image(inp_image)

    def visualize(self, layer_name, filter_nums=8):
        # Get real number of filters
        real_filter_nums = self.model.get_layer(layer_name).output.shape[-1]
        assert filter_nums <= real_filter_nums, \
            "filter_nums should less or equal to real filter_nums: {} <= {}".format(filter_nums, real_filter_nums)

        # Initial result image
        col = 8
        row = math.ceil(filter_nums / 8)
        results = np.zeros([row * self.size + (row - 1) * self.margin, col * self.size + (col - 1) * self.margin, 3]).astype("uint8")
        # Get pattern from each filter
        for i in range(row):
            for j in range(col):
                filter_pattern = self.get_interested_feature(layer_name, i + (j * row), size=self.size).numpy()

                horizontal_start = i * self.size + i * self.margin
                horizontal_end = horizontal_start + self.size
                vertical_start = j * self.size + j * self.margin
                vertical_end = vertical_start + self.size

                results[horizontal_start:horizontal_end, vertical_start:vertical_end, :] = filter_pattern
        plt.figure(figsize=(int(2.5 * row), int(2.5 * col)))
        plt.imshow(results)
        plt.show()
