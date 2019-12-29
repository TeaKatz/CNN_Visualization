import scipy
import numpy as np
import tensorflow as tf


class DeepDream:
    def __init__(self, model, layer_contributions, lr=0.01, num_octave=3, octave_scale=1.4, max_loss=40):
        self.model = model
        self.layer_contributions = layer_contributions
        self.lr = lr
        self.num_octave = num_octave
        self.octave_scale = octave_scale
        self.max_loss = max_loss

    @staticmethod
    def resize_image(image, size):
        image = np.copy(image)
        factors = (1,
                   float(size[0]) / image.shape[1],
                   float(size[1]) / image.shape[2],
                   1)
        return scipy.ndimage.zoom(image, factors, order=1)

    def cal_loss_and_grads(self, inp):
        # Get temporary models, each of them output specific target layer output
        temp_models = dict([(layer_name, tf.keras.Model(inputs=self.model.input, outputs=self.model.get_layer(layer_name).output))
                            for layer_name in self.layer_contributions])

        # Initial loss value
        loss = 0
        with tf.GradientTape() as grad_tape:
            grad_tape.watch(inp)
            for layer_name in self.layer_contributions:
                # Get weight for each specific layer
                coeff = self.layer_contributions[layer_name]
                # Get layer output (activation)
                activation = temp_models[layer_name](inp)

                # Calculate L2 norm of loss (out target is to maximize this loss value)
                scaling = tf.reduce_prod(tf.cast(activation.shape, tf.float32))
                loss += coeff * tf.reduce_sum(tf.square(activation[:, 2:-2, 2:-2, :])) / scaling
            # Calculate gradients
            grads = grad_tape.gradient(loss, inp)
            # Normalization gradient by dividing by its L1 norm
            grads /= tf.maximum(tf.reduce_mean(tf.abs(grads)), 1e-7)
        return loss, grads

    def gradient_ascent(self, inp, epochs):
        # Convert type to tensor
        inp = tf.convert_to_tensor(inp)
        for i in range(epochs):
            loss_value, grad_values = self.cal_loss_and_grads(inp)
            if self.max_loss is not None and loss_value > self.max_loss:
                break
            # Modify input image by gradient value from differnt layers according to the input image
            inp += self.lr * grad_values
        return inp

    def visualize(self, inp, epochs=20):
        # Get original image size
        original_shape = inp.shape[1:3]

        # Get smaller image size
        successive_shapes = [original_shape]
        for i in range(1, self.num_octave):
            shape = tuple([int(dim / (self.octave_scale ** i)) for dim in original_shape])
            successive_shapes.append(shape)
        # Revert image shape to smallest to bigest
        successive_shapes = successive_shapes[::-1]

        original_image = np.copy(inp)
        # Get smallest image
        shrunk_original_image = self.resize_image(original_image, successive_shapes[0])

        dream = np.copy(inp)
        for shape in successive_shapes:
            # Get dream image from each image size
            dream = self.resize_image(dream, shape)
            dream = self.gradient_ascent(dream, epochs=epochs)

            # Get lost detail from difference between upscaled image from current size image and downscaled image from original size image
            upscaled_shrunk_original_image = self.resize_image(shrunk_original_image, shape)
            downscaled_original_image = self.resize_image(original_image, shape)
            lost_detail = downscaled_original_image - upscaled_shrunk_original_image

            # Add lost detail to dream image
            dream += lost_detail

            # Update shruck_original_image
            shrunk_original_image = self.resize_image(original_image, shape)
        return dream
