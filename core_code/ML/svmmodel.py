import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops
import os
from datetime import datetime
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector


class SVMModel:
    # Create graph
    sess = ''
    features_size = 2 * 480
    x_vals: np.ndarray
    y_vals: np.ndarray
    model_path = ''
    production_path = ''
    logs_path = ''

    def __init__(self, features, labels):
        ops.reset_default_graph()
        self.y_vals = np.array(labels)
        self.x_vals = np.array(
            [self.normalize_data(np.concatenate((x['ASE'].reshape(-1, 1), x['Ti'].reshape(-1, 1)), axis=0))
             for i, x in enumerate(features)])
        current_file_path = os.path.abspath(__file__)
        parent_path = os.path.abspath(current_file_path + '/../')
        self.model_path = parent_path + '/models/svm/svm_model.cpkt'
        self.production_path = parent_path + '/models/production/svm/'
        now = datetime.now()
        self.logs_path = parent_path + '/log/svm/' + now.strftime("%Y%m%d-%H%M%S") + '/'

    def train_the_system(self):

        # Declare batch size
        batch_size = int(134)

        # Initialize placeholders
        with tf.variable_scope('input_data'):
            x_data = tf.placeholder(shape=[None, self.features_size], dtype=tf.float32, name='features')

        with tf.variable_scope('targets'):
            y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='targets')

        embedding_var = tf.Variable(batch_size, name="embedding_graph")

        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name

        # Create variables for svm
        with tf.variable_scope('svm_param'):
            b = tf.Variable(tf.random_normal(shape=[1, batch_size]), dtype=tf.float32, name='biases')
            learning_rate_placeholder = tf.placeholder(tf.float32, [], name='learning_rate')
            prediction_grid = tf.placeholder(shape=[None, self.features_size], dtype=tf.float32,
                                             name="system_predictions")
            saver = tf.train.Saver([b])
            self.sess = tf.Session()

        # Apply kernel
        # Linear Kernel
        # my_kernel = tf.matmul(x_data, tf.transpose(x_data))

        with tf.variable_scope('kernel_function'):
            # Gaussian (RBF) kernel
            gamma = tf.constant(-50.0, name='gamma_constant')
            dist = tf.reduce_sum(tf.square(x_data), 1)
            dist = tf.reshape(dist, [-1, 1])
            sq_dists = tf.add(tf.subtract(dist, tf.multiply(2., tf.matmul(x_data, tf.transpose(x_data)))),
                              tf.transpose(dist))
            my_kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)), name="Gaussian_RBF_kernel")

        # Compute SVM Model
        with tf.variable_scope('Cost'):
            first_term = tf.reduce_sum(b, name="sum_biases")
            b_vec_cross = tf.matmul(tf.transpose(b), b)
            y_target_cross = tf.matmul(y_target, tf.transpose(y_target))
            second_term = tf.reduce_sum(tf.multiply(my_kernel, tf.multiply(b_vec_cross, y_target_cross)))
            loss = tf.negative(tf.subtract(first_term, second_term), name="loss")

        # Create Prediction Kernel
        # Linear prediction kernel
        # my_kernel = tf.matmul(x_data, tf.transpose(prediction_grid))

        # Gaussian (RBF) prediction kernel
        with tf.variable_scope('Train'):
            with tf.variable_scope('kernel_prediction'):
                rA = tf.reshape(tf.reduce_sum(tf.square(x_data), 1), [-1, 1],
                                name='inputs_kernel')  # Convert it to column vector
                rB = tf.reshape(tf.reduce_sum(tf.square(prediction_grid), 1), [-1, 1],
                                name='Predictions')  # Convert it to column vector
                pred_sq_dist = tf.add(
                    tf.subtract(rA, tf.multiply(2., tf.matmul(x_data, tf.transpose(prediction_grid)))),
                    tf.transpose(rB))  #
                pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))
                prediction_output = tf.matmul(tf.multiply(tf.transpose(y_target), b), pred_kernel)
                prediction = tf.sign(prediction_output - tf.reduce_mean(prediction_output), name="predictor")
                accuracy = tf.reduce_mean(
                    tf.cast(tf.equal(self.hardLim(tf.squeeze(prediction)), tf.squeeze(y_target)), tf.float32))
            with tf.variable_scope('Optimizer'):
                # Declare optimizer
                my_opt = tf.train.GradientDescentOptimizer(learning_rate_placeholder)
                train_step = my_opt.minimize(loss, name='gradient_optimizer')

        with tf.variable_scope('system_svm_predictions'):
            feature_size = 960
            x_data_pred = tf.Variable(self.x_vals, dtype=tf.float32, name='biases')
            y_val_transpose = np.transpose(self.y_vals).reshape(-1, 1)
            y_val_pred = tf.Variable(y_val_transpose, dtype=tf.float32, name='biases')
            prediction_grid_input = tf.placeholder(shape=[2, feature_size], dtype=tf.float32,
                                                   name="prediction_input")
            rA_pre = tf.reshape(tf.reduce_sum(tf.square(x_data_pred), 1), [-1, 1])
            rB_pre = tf.reshape(tf.reduce_sum(tf.square(prediction_grid_input), 1), [-1, 1])
            pred_sq_dist_pre = tf.add(
                tf.subtract(rA_pre, tf.multiply(2., tf.matmul(x_data_pred, tf.transpose(prediction_grid_input)))),
                tf.transpose(rB_pre))
            pred_kernel_pre = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist_pre)))
            prediction_output_pre = tf.matmul(tf.multiply(tf.transpose(y_val_pred), b), pred_kernel_pre)
            prediction_pre = tf.sign(prediction_output_pre - tf.reduce_mean(prediction_output_pre))

        # Create a summary operation to log the progress of the network
        with tf.variable_scope('logging'):
            tf.summary.scalar('current_cost', loss)
            tf.summary.scalar('current_accuracy', accuracy)
            tf.summary.histogram('Predictions', prediction_output)
            tf.summary.histogram('biases', b)
            summary = tf.summary.merge_all()

        training_writer = tf.summary.FileWriter(self.logs_path, self.sess.graph)

        # Initialize/Load  variables
        init = tf.global_variables_initializer()
        # saver.restore(self.sess, self.model_path)
        self.sess.run(init)

        with tf.variable_scope('Training_loop'):
            # Training loop
            loss_vec = []
            batch_accuracy = []
            epoch = 1800
            for i in range(epoch):
                rand_index = np.random.choice(len(self.x_vals), size=batch_size)
                rand_x = self.x_vals[rand_index]
                rand_y = np.transpose([self.y_vals[rand_index]])
                learning_rate = 0.000430
                # if len(loss_vec) > 1 and 0 < loss_vec[-1] < 0.1:
                #     print('Changing Learning Rate to ' + str(learning_rate) + ' Iteration =  ' + str(i))

                self.sess.run(train_step,
                              feed_dict={x_data: rand_x, y_target: rand_y, learning_rate_placeholder: learning_rate})
                [temp_loss, log_summary] = self.sess.run([loss, summary], feed_dict={x_data: rand_x, y_target: rand_y,
                                                                                     learning_rate_placeholder: learning_rate,
                                                                                     prediction_grid: rand_x,
                                                                                     })
                loss_vec.append(temp_loss)
                training_writer.add_summary(log_summary, i)
                acc_temp = self.sess.run(accuracy, feed_dict={x_data: rand_x,
                                                              y_target: rand_y,
                                                              prediction_grid: rand_x})
                batch_accuracy.append(acc_temp)

                if (i + 1) % 10 == 0:
                    print('Step #' + str(i + 1))
                    print('Loss = ' + str(temp_loss))
                    print('accuracy = ' + str(acc_temp))
        # Plot batch accuracy
        saver.save(self.sess, self.model_path)
        plt.plot(batch_accuracy, 'k-', label='Accuracy')
        plt.title('Batch Accuracy')
        plt.xlabel('Generation')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.show()

        # Plot loss over time
        plt.plot(loss_vec, 'k-')
        plt.title('Loss per Generation')
        plt.xlabel('Generation')
        plt.ylabel('Loss')
        plt.show()
        # data_val = np.concatenate([self.x_vals[0].reshape(1, -1),np.random.uniform(low=-1.0, high=1, size=(1, 960))],axis=0)
        # [evaluations] = self.sess.run(prediction_pre, feed_dict={x_data: self.x_vals,
        #                                                          y_target: np.transpose([self.y_vals]),
        #                                                          prediction_grid_input: data_val})

        model_builder = tf.saved_model.builder.SavedModelBuilder(self.production_path + "exported_model")

        inputs = {
            'input': tf.saved_model.utils.build_tensor_info(prediction_grid_input)
        }
        outputs = {
            'earnings': tf.saved_model.utils.build_tensor_info(prediction_pre)
        }

        signature_def = tf.saved_model.signature_def_utils.build_signature_def(
            inputs=inputs,
            outputs=outputs,
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
        )

        model_builder.add_meta_graph_and_variables(
            self.sess,
            tags=[tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_def
            }
        )

        model_builder.save()

    def normalize_data(self, data):
        min = data.min()
        max = data.max()
        normalized_data = np.array([2 * (x - min) / (max - min) - 1 for i, x in enumerate(data)]).reshape(-1, )
        return normalized_data

    def hardLim(self, data):
        def negative():
            return -1.0

        def positive():
            return 1.0

        hard_lim = lambda x: tf.cond(x < 0, lambda: negative(), lambda: positive())

        return tf.map_fn(hard_lim, data)

        return tf.cond(data < 0, lambda: negative(), lambda: positive())
