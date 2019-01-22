import tensorflow as tf

import datetime

from utils import *


class DeepLabelizer():
    """
    Model class
    """
    def __init__(self, config):
        """
        Initializes the model from the provided config file
        Params
        ------
        config: `dictionary`
        """
        self.config = config
        self.n_epochs = config["n_epochs"]
        self.initial_lr = config["initial_lr"]
        self.batch_size = config["batch_size"]
        self.use_weighted_loss = config["use_weighted_loss"]
        self.pos_weight = config["pos_weight"]
        self.n_gru_cells = config["n_gru_cells"]
        self.gru_units = config["gru_units"]
        self.n_filters = config["n_filters"]
        self.kernel_size = config["kernel_size"]
        self.stride = config["stride"]
        self.train_keep_prob = config["keep_prob"]

    def build_model(self):
        """
        Builds computation graph

        """
        tf.reset_default_graph()
        # Build graph
        self.X_placeholder = tf.placeholder(tf.float32, shape=(None, 2149, 129), name='input')

        net = tf.contrib.layers.conv1d(self.X_placeholder, num_outputs=self.n_filters, kernel_size=self.kernel_size,
                                       stride=self.stride, normalizer_fn=tf.contrib.layers.batch_norm)
        y_output_size = net.shape.as_list()[1]

        self.y_placeholder = tf.placeholder(tf.float32, shape=(None, y_output_size, 1), name='label')
        self.keep_prob = tf.placeholder(tf.float32, shape=())


        # Dropout
        net = tf.contrib.layers.dropout(net, keep_prob=self.keep_prob)


        # GRU bloc
        gru_cells = []
        for _ in range(self.n_gru_cells):
            cell = tf.contrib.rnn.GRUCell(num_units=self.gru_units)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
            gru_cells.append(cell)
        cell = tf.contrib.rnn.MultiRNNCell(gru_cells)
        output, state = tf.nn.dynamic_rnn(cell, net, dtype=tf.float32)


        # Dense layer
        out_size = self.y_placeholder.get_shape()[2].value
        logit = tf.contrib.layers.fully_connected(output, out_size, activation_fn=None)
        self.prediction = tf.nn.sigmoid(logit)

        flat_target = tf.reshape(self.y_placeholder, [-1] + self.y_placeholder.shape.as_list()[2:])
        flat_logit = tf.reshape(logit, [-1] + logit.shape.as_list()[2:])


        # Losses
        if self.use_weighted_loss:
            # Training loss
            loss = tf.nn.weighted_cross_entropy_with_logits(targets=flat_target, logits=flat_logit,
                                                            pos_weight=self.pos_weight)
            self.loss = tf.reduce_mean(loss)
            # Validation loss
            valid_loss = tf.nn.weighted_cross_entropy_with_logits(targets=flat_target, logits=flat_logit,
                                                                  pos_weight=self.pos_weight)
            self.valid_loss = tf.reduce_mean(valid_loss)
        else:
            # Training loss
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=flat_target, logits=flat_logit)
            self.loss = tf.reduce_mean(loss)
            # Validation loss
            valid_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=flat_target, logits=flat_logit)
            self.valid_loss = tf.reduce_mean(valid_loss)

            # Adam optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.initial_lr).minimize(self.loss)

        # Train metrics
        correct_pred = tf.equal(tf.cast(tf.round(self.prediction), tf.float32), self.y_placeholder)
        self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        train_recall = tf.metrics.recall(predictions=tf.round(self.prediction), labels=self.y_placeholder)
        self.train_recall = tf.reduce_mean(train_recall)

        train_precision = tf.metrics.precision(labels=self.y_placeholder, predictions=tf.round(self.prediction))
        self.train_precision = tf.reduce_mean(train_precision)

        train_f1 = tf.contrib.metrics.f1_score(labels=self.y_placeholder, predictions=tf.round(self.prediction))
        self.train_f1 = tf.reduce_mean(train_f1)

        # Valid metrics
        valid_correct_pred = tf.equal(tf.cast(tf.round(self.prediction), tf.float32), self.y_placeholder)
        self.valid_acc = tf.reduce_mean(tf.cast(valid_correct_pred, tf.float32))

        valid_recall = tf.metrics.recall(predictions=tf.round(self.prediction), labels=self.y_placeholder)
        self.valid_recall = tf.reduce_mean(valid_recall)

        valid_precision = tf.metrics.precision(labels=self.y_placeholder, predictions=tf.round(self.prediction))
        self.valid_precision = tf.reduce_mean(valid_precision)

        valid_f1 = tf.contrib.metrics.f1_score(labels=self.y_placeholder, predictions=tf.round(self.prediction))
        self.valid_f1 = tf.reduce_mean(valid_f1)

        # Add ops to save and restore all the variables.

        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("accuracy", self.acc)
        tf.summary.scalar("valid_loss", self.valid_loss)
        tf.summary.scalar("valid_accuracy", self.valid_acc)
        tf.summary.scalar("train_precision", self.train_precision)
        tf.summary.scalar("train_recall", self.train_recall)
        tf.summary.scalar("train_f1", self.train_f1)
        tf.summary.scalar("valid_precision", self.valid_precision)
        tf.summary.scalar("valid_recall", self.valid_recall)
        tf.summary.scalar("valid_f1", self.valid_f1)

        self.merged_summary_op = tf.summary.merge_all()

        self.init_global = tf.global_variables_initializer()
        self.init_local = tf.local_variables_initializer()

    def train_labelizer(self, X, y, sess, path_model, t, verbose=None):
        """
        Params
        ------
        X:
        y:
        sess:
        verbose: `int`
            if verbose == 1, training information will be displayed

        """

        X_train, y_train = build_datasets(X, y, mode='train')
        X_valid, y_valid = build_datasets(X, y, mode='valid')
        n_train = len(X_train)
        n_valid = len(X_valid)
        n_iter_train = n_train // self.batch_size
        n_iter_valid = n_valid // self.batch_size


        # Save the model and the logs to keep track of the training
        logs_path = path_model + "/logs_{}".format(t)

        # Create paths for logs and model if they don't already exist
        if not os.path.exists(path_model):
            os.makedirs(path_model)
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)

        # Save config file in the model folder to keep track of the parameters used
        with open(path_model + '/config.json', 'w') as fp:
            json.dump(self.config, fp)

        saver = tf.train.Saver()

        # Variables initialization
        sess.run(self.init_global)  # Initialize global variables
        sess.run(self.init_local)  # Initialize local variables (e.g: metrics)

        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        # Training
        for epoch in range(self.n_epochs):
            indices_train = np.arange(n_train)
            np.random.shuffle(indices_train)

            indices_valid = np.arange(n_valid)
            np.random.shuffle(indices_valid)

            train_losses = []
            valid_losses = []
            train_accuracies = []
            valid_accuracies = []
            train_recalls = []
            valid_recalls = []
            train_precisions = []
            valid_precisions = []
            train_f1s = []
            valid_f1s = []

            # Train computations
            for i in range(n_iter_train):
                idx = indices_train[i * self.batch_size: (i + 1) * self.batch_size]
                X_batch = np.stack(X_train[idx])
                y_batch = np.stack(y_train[idx])
                train_loss, train_acc, _, train_precision_, train_recall_, train_f1 = sess.run(
                    [self.loss, self.acc, self.optimizer, self.train_precision, self.train_recall, self.train_f1],
                    feed_dict={self.X_placeholder: X_batch, self.y_placeholder: y_batch,
                               self.keep_prob: self.train_keep_prob})

                train_losses.append(train_loss)
                train_accuracies.append(train_acc)
                train_precisions.append(train_precision_)
                train_recalls.append(train_recall_)
                train_f1s.append(train_f1)

            train_loss__ = np.mean(train_losses)
            train_acc__ = np.mean(train_accuracies)
            train_precision__ = np.mean(train_precisions)
            train_recall__ = np.mean(train_recalls)
            train_f1__ = np.mean(train_f1s)

            # Valid computations
            for i in range(n_iter_valid):
                idx = indices_valid[i * self.batch_size: (i + 1) * self.batch_size]
                X_batch_valid = np.stack(X_valid[idx])
                y_batch_valid = np.stack(y_valid[idx])
                valid_loss_, valid_acc_, valid_precision_, valid_recall_, valid_f1 = sess.run(
                    [self.valid_loss, self.valid_acc, self.valid_precision, self.valid_recall, self.valid_f1],
                    feed_dict={self.X_placeholder: X_batch_valid, self.y_placeholder: y_batch_valid,
                               self.keep_prob: 1.0})
                valid_losses.append(valid_loss_)
                valid_accuracies.append(valid_acc_)
                valid_precisions.append(valid_precision_)
                valid_recalls.append(valid_recall_)
                valid_f1s.append(valid_f1)

            valid_loss__ = np.mean(valid_losses)
            valid_acc__ = np.mean(valid_accuracies)
            valid_precision__ = np.mean(valid_precisions)
            valid_recall__ = np.mean(valid_recalls)
            valid_f1__ = np.mean(valid_f1s)

            if verbose == 2:
                print(
                    "Epoch {}: Train:  loss: {}, acc: {}, precision: {}, recall: {}, f1: {}".format(epoch, train_loss__,
                                                                                                    train_acc__,
                                                                                                    train_precision__,
                                                                                                    train_recall__,
                                                                                                    train_f1__))
                print(
                    "Epoch {}: Valid: loss: {}, acc: {}, precision: {}, recall: {}, f1: {}".format(epoch, valid_loss__,
                                                                                                   valid_acc__,
                                                                                                   valid_precision__,
                                                                                                   valid_recall__,
                                                                                                   valid_f1__))
            elif verbose == 1:
                if (epoch + 1) % 10 == 0:
                    print("Epoch {}: Train:  loss: {}, acc: {}, precision: {}, recall: {}, f1: {}".format(epoch,
                                                                                                          train_loss__,
                                                                                                          train_acc__,
                                                                                                          train_precision__,
                                                                                                          train_recall__,
                                                                                                          train_f1__))
                    print("Epoch {}: Valid: loss: {}, acc: {}, precision: {}, recall: {}, f1: {}".format(epoch,
                                                                                                         valid_loss__,
                                                                                                         valid_acc__,
                                                                                                         valid_precision__,
                                                                                                         valid_recall__,
                                                                                                         valid_f1__))

            summary = sess.run(self.merged_summary_op,
                               feed_dict={self.loss: train_loss__, self.acc: train_acc__,
                                          self.train_precision: train_precision__,
                                          self.train_recall: train_recall__, self.train_f1: train_f1__,
                                          self.valid_loss: valid_loss__, self.valid_acc: valid_acc__,
                                          self.valid_precision: valid_precision__, self.valid_recall: valid_recall__,
                                          self.valid_f1: valid_f1__})
            summary_writer.add_summary(summary, epoch)

            if epoch % 5 == 0:
                save_path = saver.save(sess, path_model + "/model.ckpt")
                print("Model saved in path: %s" % save_path)

            if epoch + 1 == self.n_epochs:
                save_path = saver.save(sess, path_model + "/last_model.ckpt")
                print("Last model saved in path: %s" % save_path)

    def evaluate_labelizer(self, X, y, session):
        y_pred = session.run(self.prediction, feed_dict={self.X_placeholder: X, self.y_placeholder: y,
                                                         self.keep_prob: 1.0})
        return y_pred
