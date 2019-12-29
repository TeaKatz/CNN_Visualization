import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class ClassActivationMap:
    def __init__(self, model, last_conv_layer_name, target_class_index=None):
        self.model = model
        self.last_conv_layer_name = last_conv_layer_name
        self.target_class_index = target_class_index

    def visualize(self, inp, original_img=None):
        # Convert input to tensor
        tf_inp = tf.convert_to_tensor(inp)

        # Get last conv layer
        conv_layer = tf.keras.Model(inputs=self.model.input, outputs=self.model.get_layer(self.last_conv_layer_name).output)
        # Create temporary model to get conv_layer output and predtiction
        temp_model = tf.keras.Model(inputs=self.model.input, outputs=[self.model.output, conv_layer.output])

        # Calculate gradient
        with tf.GradientTape() as grad_tap:
            grad_tap.watch(tf_inp)
            pred, conv_layer_output = temp_model(tf_inp)
            if self.target_class_index is None:
                # Use maximum class
                grad = grad_tap.gradient(tf.reduce_max(pred[0]), conv_layer_output)
            else:
                # Use target class
                grad = grad_tap.gradient(pred[0, self.target_class_index], conv_layer_output)
        grad = grad[0]
        conv_layer_output = conv_layer_output[0]

        # Convert tensor to numpy
        conv_layer_output = conv_layer_output.numpy()

        # Weight conv output with gradient
        for i in range(conv_layer_output.shape[-1]):
            conv_layer_output[:, :, i] *= grad[:, :, i]

        # Calculate heatmap by average last dim of conv output
        heatmap = np.mean(conv_layer_output, axis=-1)
        # Truncate negative values and scale to [0, 1]
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        # Resize heatmap to target image
        if original_img is None:
            target_img = inp[0]
        else:
            target_img = original_img
        heatmap = cv2.resize(heatmap, (target_img.shape[1], target_img.shape[0]))
        # Scale to [0, 255]
        heatmap = np.uint8(255 * heatmap)
        # Convert from 1 chanel to 3 chanels
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        # Combine image and heatmap
        combined_img = np.uint8(np.minimum(heatmap * 0.2 + target_img, 255))
        plt.imshow(combined_img)
        plt.show()
