import io

import matplotlib.pyplot as plt
from tensorflow.python.keras import backend as K
# from keras import backend as K
# from keras.callbacks import Callback
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.summary import summary as tf_summary


class TensorBoard(Callback):
    ''' Tensorboard basic visualizations.

    This callback writes a log for TensorBoard, which allows
    you to visualize dynamic graphs of your training and test
    metrics, as well as activation histograms for the different
    layers in your model.

    TensorBoard is a visualization tool provided with TensorFlow.

    If you have installed TensorFlow with pip, you should be able
    to launch TensorBoard from the command line:
    ```
    tensorboard --logdir=/full_path_to_your_logs
    ```
    You can find more information about TensorBoard
    [here](https://www.tensorflow.org/versions/master/how_tos/summaries_and_tensorboard/index.html).

    # Arguments
        log_dir: the path of the directory where to save the log
            files to be parsed by Tensorboard
        histogram_freq: frequency (in epochs) at which to compute activation
            histograms for the layers of the model. If set to 0,
            histograms won't be computed.
        write_graph: whether to visualize the graph in Tensorboard.
            The log file can become quite large when
            write_graph is set to True.
    '''
    def __init__(self, log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False, validation_data=None, batch_size=16):
        super(TensorBoard, self).__init__()
        # if K._BACKEND != 'tensorflow':
        #     raise RuntimeError('TensorBoard callback only works '
        #                        'with the TensorFlow backend.')
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.merged = None
        self.write_graph = write_graph
        self.write_images = write_images
        self.validation_data = validation_data
        self.batch_size = batch_size


    def set_model(self, model):
        import tensorflow as tf
        # import keras.backend.tensorflow_backend as KTF
        # import tensorflow.python.keras.backend as KTF

        self.model = model
        # self.sess = KTF.get_session()
        self.sess = tf.keras.backend.get_session()
        if self.histogram_freq and self.merged is None:
            for layer in self.model.layers:

                for weight in layer.weights:
                    tf.summary.histogram(weight.name, weight)

                    if self.write_images:
                        w_img = tf.squeeze(weight)

                        shape = w_img.get_shape()
                        if len(shape) > 1 and shape[0] > shape[1]:
                            w_img = tf.transpose(w_img)

                        if len(shape) == 1:
                            w_img = tf.expand_dims(w_img, 0)

                        w_img = tf.expand_dims(tf.expand_dims(w_img, 0), -1)

                        tf.summary.image(weight.name, w_img)

                if hasattr(layer, 'output'):
                    tf.summary.histogram('{}_out'.format(layer.name),
                                         layer.output)
        self.merged = tf.summary.merge_all()
        if self.write_graph:
            self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        else:
            self.writer = tf.summary.FileWriter(self.log_dir)

    def gen_plot(self, epoch, logs):
        """Create a pyplot plot and save to buffer."""
        self._valdata = next(self.validation_data)
        x, y = self._valdata
        decoded_imgs = self.model.predict(x)

        n = 10
        fig, ax = plt.subplots(3, n, figsize=(20, 4))
        for i in range(n):
            ax[0][i].imshow(y[i])
            ax[1][i].imshow(x[i])
            ax[2][i].imshow(decoded_imgs[i])
            ax[0][i].get_xaxis().set_visible(False)
            ax[0][i].get_yaxis().set_visible(False)
            ax[1][i].get_xaxis().set_visible(False)
            ax[1][i].get_yaxis().set_visible(False)
            ax[2][i].get_xaxis().set_visible(False)
            ax[2][i].get_yaxis().set_visible(False)


        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        return buf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        import tensorflow as tf
        import numpy as np

        # Prepare the plot
        plot_buf = self.gen_plot(epoch, logs)

        # Convert PNG buffer to TF image
        image = tf.image.decode_png(plot_buf.getvalue(), channels=4)

        # Add the batch dimension
        image = tf.expand_dims(image, 0)

        # Add image summary
        plot_summary_op = tf.summary.image('Epoch {epoch:02d}'.format(epoch=epoch, **logs), image)
        plot_summary = self.sess.run(plot_summary_op)
        self.writer.add_summary(plot_summary, global_step=epoch)


        if not self.validation_data and self.histogram_freq:
            raise ValueError('If printing histograms, validation_data must be '
                             'provided, and cannot be a generator.')
        if self.validation_data and self.histogram_freq:
            if epoch % self.histogram_freq == 0:

                val_data = list(self._valdata)
                _sample_weights = np.ones(shape=self.batch_size)
                val_data.append(_sample_weights)
                tensors = (
                        self.model.inputs + self.model.targets + self.model.sample_weights)

                if self.model.uses_learning_phase:
                    tensors += [K.learning_phase()]

                assert len(val_data) == len(tensors)
                val_size = val_data[0].shape[0]
                i = 0
                while i < val_size:
                    step = min(self.batch_size, val_size - i)
                    batch_val = []
                    batch_val.append(val_data[0][i:i + step]
                                     if val_data[0] is not None else None)
                    batch_val.append(val_data[1][i:i + step]
                                     if val_data[1] is not None else None)
                    batch_val.append(val_data[2][i:i + step]
                                     if val_data[2] is not None else None)
                    if self.model.uses_learning_phase:
                        # do not slice the learning phase
                        batch_val = [x[i:i + step] if x is not None else None
                                     for x in val_data[:-1]]
                        batch_val.append(val_data[-1])
                    else:
                        batch_val = [x[i:i + step] if x is not None else None
                                     for x in val_data]
                    feed_dict = {}
                    for key, val in zip(tensors, batch_val):
                        if val is not None:
                            feed_dict[key] = val
                    result = self.sess.run([self.merged], feed_dict=feed_dict)
                    summary_str = result[0]
                    self.writer.add_summary(summary_str, epoch)
                    i += self.batch_size

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf_summary.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, epoch)
        self.writer.flush()

    def __on_epoch_end(self, epoch, logs={}):
        import tensorflow as tf

        # Prepare the plot
        plot_buf = self.gen_plot(epoch, logs)

        # Convert PNG buffer to TF image
        image = tf.image.decode_png(plot_buf.getvalue(), channels=4)

        # Add the batch dimension
        image = tf.expand_dims(image, 0)

        # Add image summary
        plot_summary_op = tf.summary.image('Epoch {epoch:02d}'.format(epoch=epoch, **logs), image)
        plot_summary = self.sess.run(plot_summary_op)
        self.writer.add_summary(plot_summary, global_step=epoch)

        if self.validation_data and self.histogram_freq:
            if epoch % self.histogram_freq == 0:
                # TODO: implement batched calls to sess.run
                # (current call will likely go OOM on GPU)
                if self.model.uses_learning_phase:
                    cut_v_data = len(self.model.inputs)
                    val_data = self.model.validation_data[:cut_v_data] + [0]
                    tensors = self.model.inputs + [K.learning_phase()]
                else:
                    val_data = self.validation_data
                    tensors = self.model.inputs
                feed_dict = dict(zip(tensors, val_data))
                result = self.sess.run([self.merged], feed_dict=feed_dict)
                summary_str = result[0]
                self.writer.add_summary(summary_str, epoch)

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, epoch)
        self.writer.flush()

    def on_train_end(self, _):
        self.writer.close()
