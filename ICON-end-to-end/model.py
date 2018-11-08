import tensorflow as tf

class ICON:

    def __init__(self, config, embeddingMatrix, session = None):
        
        # CNN related
        self._embedding_matrix = embeddingMatrix
        self._sequence_length = config["max_sequence_length"]
        self._embedding_size = config["embedding_dim"]
        self._filter_sizes = config["filter_sizes"]
        self._num_filters = config["num_filters"]
        self._num_filters_total = self._num_filters * len(self._filter_sizes)

        # Memory Network hops
        self._timesteps = config["timesteps"]
        self._hops = config["hops"]
        
        # Training related
        self._lr = config["learning_rate"]
        self._batch_size = config["batch_size"]
        self._class_size = config["num_classes"]
        self._max_grad_norm = config["max_grad_norm"]
        self._init = tf.random_normal_initializer(stddev=0.01, seed=1227)
        self._name = "ICON"


        ## inputs to receive from the dataset
        self._build_inputs()

        ## tensor variables of the tensorflow graph
        self._build_vars()

        ## optimizer choices for training
        # self._opt = tf.train.GradientDescentOptimizer(learning_rate=self._lr)
        self._opt = tf.train.AdamOptimizer(learning_rate=self._lr)

        ## cross entropy loss
        logits = self._inference() # (batch_size, class size)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.cast(self._labels, tf.float32), name="cross_entropy")
        cross_entropy_sum = tf.reduce_mean(cross_entropy, keepdims=False, name="cross_entropy_sum")

        print('\n ---- TRAINABLE VARIABLES ---- \n')
        tvars = tf.trainable_variables()
        reg_loss=[]
        for tvar in tvars:
            print(tvar.name)
            if "bias" not in tvar.name:
                reg_loss.append(tf.nn.l2_loss(tvar))
        print('----------- \n')

        # loss op
        self.regularization_loss = tf.reduce_mean(reg_loss)
        self.loss_op = loss_op = tf.reduce_mean(cross_entropy_sum + 0.001*self.regularization_loss)

        # gradient pipeline
        grads_and_vars = self._opt.compute_gradients(loss_op)
        grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v) for g,v in grads_and_vars]
        grads_and_vars = [(g, v) for g,v in grads_and_vars]
        self.train_op = train_op = self._opt.apply_gradients(grads_and_vars, name="train_op")

        # predict ops
        self.predict_op = predict_op = tf.argmax(logits, 1, name="predict_op")

        self._sess = session
        self._sess.run(tf.global_variables_initializer())


    def _build_inputs(self):

        # input queries
        self._input_queries = tf.placeholder(tf.int32, [self._batch_size, self._sequence_length], name="input_queries")

        # histories
        self._own_histories = tf.placeholder(tf.int32, [self._batch_size, self._timesteps, self._sequence_length], name="own_histories")
        self._other_histories = tf.placeholder(tf.int32, [self._batch_size, self._timesteps, self._sequence_length], name="other_histories")

        # True Labels
        self._labels = tf.placeholder(tf.int32, [self._batch_size, self._class_size], name="labels")


    def _build_vars(self):
        with tf.variable_scope(self._name):

            with tf.variable_scope("CNN"):
                self.embedding_matrix = tf.Variable(self._embedding_matrix, name="embedding_matrix", dtype=tf.float32)
                self.conv_final_W = tf.get_variable(
                        "conv_final_W", 
                        shape=[self._num_filters_total, self._embedding_size],
                        initializer=tf.contrib.layers.xavier_initializer())
                self.conv_final_b = tf.Variable(tf.constant(0.1, shape=[self._embedding_size]), name="conv_final_b")

            with tf.variable_scope("output"):
                self.output_W = tf.get_variable(
                    "output_W",
                    shape=[self._embedding_size, self._class_size],
                    initializer=tf.contrib.layers.xavier_initializer())
                self.output_b = tf.Variable(tf.constant(0.1, shape=[self._class_size]), name="output_b")

            with tf.variable_scope("SIM"):
                self.rnn_own_history= tf.contrib.rnn.GRUCell(num_units=self._embedding_size, reuse = tf.AUTO_REUSE, name='rnn_own_history')
                self.rnn_other_history= tf.contrib.rnn.GRUCell(num_units=self._embedding_size, reuse = tf.AUTO_REUSE, name='rnn_other_history')

            with tf.variable_scope("DGIM"):
                self.rnn_dgim= tf.contrib.rnn.GRUCell(num_units=self._embedding_size, reuse = tf.AUTO_REUSE, name='rnn_dgim')

            with tf.variable_scope("MemoryNet"):
                self.rnn_memory= tf.contrib.rnn.GRUCell(num_units=self._embedding_size, reuse = tf.AUTO_REUSE, name='rnn_memory')

    def _convolution(self, input_to_conv):

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for idx, filter_size in enumerate(self._filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size):
                
                # Convolution Layer
                filter_shape = [filter_size, self._embedding_size, 1, self._num_filters]
                W = tf.get_variable("W", initializer=tf.truncated_normal(filter_shape, stddev=0.1))
                b = tf.get_variable("b", initializer=tf.constant(0.1, shape=[self._num_filters]))
                conv = tf.nn.conv2d(
                    input_to_conv,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")

                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self._sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, self._num_filters_total], name="h_pool_flat")

        return tf.nn.xw_plus_b(self.h_pool_flat, self.conv_final_W, self.conv_final_b, name="conv_dense")


    def _inference(self):

        with tf.variable_scope(self._name, reuse=tf.AUTO_REUSE):

            with tf.variable_scope("CNN", reuse=tf.AUTO_REUSE):

                # feature extraction for queries
                embedded_words_queries = tf.expand_dims(tf.nn.embedding_lookup(self.embedding_matrix, self._input_queries), -1) # (batch, sequence_length, embedding_dim, 1)
                queries = queries_conv_output = self._convolution(embedded_words_queries) # (batch, _num_filters_total)

                # feature extraction for ownHistory
                own_history_conv_output=[]
                for i in range(self._timesteps):
                    local_history = tf.squeeze(self._own_histories[:,tf.constant(i)]) # (batch, sequence_length)
                    embedded_local_history = tf.expand_dims(tf.nn.embedding_lookup(self.embedding_matrix, local_history), -1) # (batch, sequence_length, embedding_dim, 1)
                    own_history_conv_output.append(self._convolution(embedded_local_history)[:,tf.newaxis,:]) # (batch, 1, _num_filters_total)
                own_history_conv_output = tf.concat(own_history_conv_output, axis=1) # (batch, timesteps, _num_filters_total)

                # feature extraction for otherHistory
                other_history_conv_output=[]
                for i in range(self._timesteps):
                    local_history = tf.squeeze(self._other_histories[:,tf.constant(i)]) # (batch, sequence_length)
                    embedded_local_history = tf.expand_dims(tf.nn.embedding_lookup(self.embedding_matrix, local_history), -1) # (batch, sequence_length, embedding_dim, 1)
                    other_history_conv_output.append(self._convolution(embedded_local_history)[:,tf.newaxis,:]) # (batch, 1, _num_filters_total)
                other_history_conv_output = tf.concat(other_history_conv_output, axis=1) # (batch, timesteps, _num_filters_total)


                # SIM on histories
                rnn_own_history, _ = tf.nn.dynamic_rnn(self.rnn_own_history, own_history_conv_output, dtype=tf.float32)
                rnn_other_history, _ = tf.nn.dynamic_rnn(self.rnn_other_history, other_history_conv_output, dtype=tf.float32)

                print(rnn_own_history.get_shape())

                # DGIM on histories
                dgim_input = (rnn_own_history + rnn_other_history)


            with tf.variable_scope("DGIM"):

                for hop in range(self._hops):

                    # Memory Update
                    if hop == 0:
                        rnn_input = dgim_input
                        rnn_cell = self.rnn_dgim
                    else:
                        rnn_input = rnn_outputs
                        rnn_cell = self.rnn_memory

                    # Memory write of previous hop == memory input of current hop
                    rnn_outputs, _ = tf.nn.dynamic_rnn(rnn_cell, rnn_input, dtype=tf.float32)


                    # Attentional Read operation from rnn_output memories
                    attScore = tf.nn.tanh(tf.squeeze(tf.matmul(queries[:,tf.newaxis,:], tf.transpose(rnn_outputs,[0,2,1]))))  # (batch, 1, _num_filters_total)  X (batch, _num_filters_total, timesteps) == (batch, 1, timesteps) -> (batch, time)
                    attScore = tf.nn.softmax(attScore) # (batch, time)
                    weighted = tf.squeeze(tf.matmul(attScore[:,tf.newaxis,:], rnn_outputs)) # (batch, 1, timesteps)  X (batch, timesteps, _num_filters_total) == (batch, _num_filters_total)
                    queries = tf.nn.tanh(queries + weighted)

                

            with tf.variable_scope("output"):
                
                return tf.nn.xw_plus_b(queries, self.output_W, self.output_b, name="output_scores")


    def batch_fit(self, queries, ownHistory, otherHistory, labels):

        feed_dict = {self._input_queries: queries, self._own_histories: ownHistory, self._other_histories: otherHistory, self._labels: labels}
        loss, _ = self._sess.run([self.loss_op, self.train_op], feed_dict=feed_dict)
        return loss

    def predict(self, queries, ownHistory, otherHistory,):

        feed_dict = {self._input_queries: queries, self._own_histories: ownHistory, self._other_histories: otherHistory}
        return self._sess.run(self.predict_op, feed_dict=feed_dict)

