import tensorflow as tf
import numpy as np
import sys
import tensorflow.contrib.rnn as rnn_cell
from sklearn.model_selection import train_test_split




class ICON:

    def __init__(self, CONFIG, session = None):
        
        self._batch_size = CONFIG.batch_size
        self._input_dim = CONFIG.input_dims
        self._timesteps = CONFIG.timesteps
        self._class_size = CONFIG.class_size
        self._embedding_size = CONFIG.embedding_size
        self._hops = CONFIG.hops
        self._max_grad_norm = CONFIG.max_grad_norm
        self._nonlin = CONFIG.nonlin
        self._nonlin_func = CONFIG.nonlin_func
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
        logits = self._inference(self._histories_own, self._histories_other, self._queries) # (batch_size, class size)
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

        self._queries = tf.placeholder(tf.float32, [self._batch_size, self._input_dim], name="queries") # utterances to be classified

        # Histories of the utterances
        self._histories_own = tf.placeholder(tf.float32, [self._batch_size, self._timesteps, self._input_dim], name="histories_own") 
        self._histories_other = tf.placeholder(tf.float32, [self._batch_size, self._timesteps, self._input_dim], name="histories_other")
        self._histories_own_mask = tf.cast(tf.placeholder(tf.float32, [self._batch_size, self._timesteps], name="histories_own_mask"), dtype=tf.bool)
        self._histories_other_mask = tf.cast(tf.placeholder(tf.float32, [self._batch_size, self._timesteps], name="histories_other_mask"), dtype=tf.bool)
        self._mask = tf.cast(tf.placeholder(tf.float32, [self._batch_size, self._timesteps], name="global_mask"), dtype=tf.bool)

        # True Labels
        self._labels = tf.placeholder(tf.int32, [self._batch_size, self._class_size], name="labels")
        

        # Learning Rate
        self._lr = tf.placeholder(tf.float32, [], name="learning_rate")

        # Dropout Probability
        self._dropout = tf.placeholder(tf.float32, [], name="dropout_keep_rate")

        # Training mode
        self._training = tf.placeholder(tf.bool, [], name="training_testing_mode")

    def _build_vars(self):

        with tf.variable_scope(self._name):

            with tf.variable_scope("localGRU"):
                # GRUs for per-person local input modeling
                self.localGRUOwn = tf.contrib.rnn.GRUCell(num_units=self._embedding_size, reuse = tf.AUTO_REUSE, name='localGRUOwn')
                self.localGRUOther = tf.contrib.rnn.GRUCell(num_units=self._embedding_size, reuse = tf.AUTO_REUSE, name='localGRUOther')

            with tf.variable_scope("globalGRU"):
                self.globalGRU = tf.contrib.rnn.GRUCell(num_units=self._embedding_size, reuse = tf.AUTO_REUSE, name='globalGRU')

            with tf.variable_scope("memoryGRU"):
                self.memoryGRU = tf.contrib.rnn.GRUCell(num_units=self._embedding_size, reuse = tf.AUTO_REUSE, name='memoryGRU')

            with tf.variable_scope("output"):

                # Output Projection Matrix
                self.outputProj = tf.get_variable("outProj", shape=([self._embedding_size, self._class_size]), trainable=True, initializer=self._init)
                self.outputProjBias = tf.get_variable("outputProjBias", shape=([1, self._class_size]), trainable=True, initializer=self._init)
            

    def _inference(self, histories_own, histories_other, queries):

        with tf.variable_scope(self._name):

            with tf.variable_scope("input"):
                
                q = tf.contrib.layers.fully_connected(
                    queries,
                    self._embedding_size,
                    activation_fn=tf.nn.tanh,
                    normalizer_fn=None,
                    normalizer_params=None,
                    weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=1227),
                    weights_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                    biases_initializer=tf.zeros_initializer(),
                    trainable=True,
                    scope="input"
                )

            # SIM Module
            with tf.variable_scope("localGRU"):

                
                ## Input GRU Own
                hidden_vector = self.localGRUOwn.zero_state(self._batch_size, tf.float32)
                ownRNNOutput=[]
                for i in range(self._timesteps):

                    localMask = tf.squeeze(self._histories_own_mask[:,tf.constant(i)]) # batch_size
                    localInput = tf.squeeze(histories_own[:,tf.constant(i),:]) # batch_size * dim
                    prev_hidden_vector = hidden_vector
                    hidden_vector,_= self.localGRUOwn(localInput, hidden_vector) # batch_size * dim
                    # mask hidden vector (mask 0 places should not apply GRU)
                    hidden_vector = tf.where(localMask, hidden_vector, prev_hidden_vector) # batch_size * dim

                    # masked output_vector
                    output_vector = tf.where(localMask, hidden_vector, tf.zeros( (self._batch_size,self._embedding_size), dtype=np.float32)) # batch_size * dim
                    ownRNNOutput.append(output_vector[:,tf.newaxis,:])
                    
                ownHistoryRNNOutput = tf.concat(ownRNNOutput, axis=1) # batch_size * timesteps * dim
                ownHistoryRNNOutput = tf.nn.dropout(ownHistoryRNNOutput, keep_prob = self._dropout, name = "own_rnn_dropout")

                ## Input GRU Other
                hidden_vector = self.localGRUOther.zero_state(self._batch_size, tf.float32)
                otherRNNOutput=[]
                for i in range(self._timesteps):

                    localMask = tf.squeeze(self._histories_other_mask[:,tf.constant(i)]) # batch_size
                    localInput = tf.squeeze(histories_other[:,tf.constant(i),:]) # batch_size * dim
                    prev_hidden_vector = hidden_vector
                    hidden_vector,_= self.localGRUOther(localInput, hidden_vector) # batch_size * dim
                    # mask hidden vector (mask 0 places should not apply GRU)
                    hidden_vector = tf.where(localMask, hidden_vector, prev_hidden_vector) # batch_size * dim

                    # masked output_vector
                    output_vector = tf.where(localMask, hidden_vector, tf.zeros( (self._batch_size,self._embedding_size), dtype=np.float32)) # batch_size * dim
                    otherRNNOutput.append(output_vector[:,tf.newaxis,:])
                    
                otherHistoryRNNOutput = tf.concat(otherRNNOutput, axis=1)
                otherHistoryRNNOutput = tf.nn.dropout(otherHistoryRNNOutput, keep_prob = self._dropout, name = "other_rnn_dropout")


            # DGIM Module
            globalGRUInput = ownHistoryRNNOutput + otherHistoryRNNOutput
            globalGRUInput = tf.nn.tanh(globalGRUInput)

            with tf.variable_scope("globalGRU"):                 

                for hop in range(self._hops):

                    # Memory Update
                    if hop == 0:
                        rnn_input = globalGRUInput
                        rnn_cell = self.globalGRU
                    else:
                        rnn_input = rnn_outputs
                        rnn_cell = self.memoryGRU

                    rnn_outputs, final_state = tf.nn.dynamic_rnn(rnn_cell, rnn_input, dtype=tf.float32)

                    # looping for masking as tf.where doesnt support 2d masking
                    rnnOutputs = []
                    for j in range(self._timesteps):
                        localMask = tf.squeeze(self._mask[:,tf.constant(j)]) # batch_size
                        localOutput = tf.squeeze(rnn_outputs[:,tf.constant(j),:]) # batch_size * dim
                        outputVector = tf.where(localMask, localOutput, tf.zeros((self._batch_size,self._embedding_size), dtype=np.float32))
                        rnnOutputs.append(outputVector[:,tf.newaxis,:])
                    rnn_outputs = tf.concat(rnnOutputs, axis=1)

                    
                    # Attentional Read operation from rnn_output memories
                    attScore = tf.nn.tanh(tf.squeeze(tf.matmul(q[:,tf.newaxis,:], tf.transpose(rnn_outputs,[0,2,1]))))  # (batch, 1, dim)  X (batch, dim, time) == (batch, 1, time) -> (batch, time)
                    attScore = tf.where( self._mask, attScore, tf.constant( -10000 , shape= [self._batch_size, self._timesteps], dtype= tf.float32))
                    softmax_output = attScore = tf.nn.softmax(attScore) # (batch, time)
                    attScore = tf.nn.dropout(attScore, keep_prob = self._dropout, name='ttScore_dropout')
                    attScore = tf.where( self._mask, attScore, tf.zeros(tf.shape(attScore), dtype= tf.float32))
                    weighted = tf.squeeze(tf.matmul(attScore[:,tf.newaxis,:], rnn_outputs)) # (batch, 1, time)  X (batch, time, dim) == (batch, dim)
                    q = tf.nn.tanh(q + weighted)


            with tf.variable_scope("output"):

                return tf.add(tf.matmul(q, self.outputProj), self.outputProjBias)


    def batch_fit(self, histories_own, histories_other, histories_own_mask, histories_other_mask, global_mask, queries, labels, learning_rate, dropout_keep_rate, training_mode=None):
        '''
        Runs the training algorithm over the passed batch
        Returns:
            loss: floating-point number, the loss computed for the batch
        '''
        feed_dict = {self._histories_own: histories_own, self._histories_other: histories_other, self._queries: queries, self._labels: labels,\
         self._lr: learning_rate, self._histories_own_mask: histories_own_mask, self._histories_other_mask: histories_other_mask, self._dropout: dropout_keep_rate, self._mask: global_mask, self._training:training_mode}
        loss, _ = self._sess.run([self.loss_op, self.train_op], feed_dict=feed_dict)
        return loss

        
    def predict(self, histories_own, histories_other, histories_own_mask, histories_other_mask, global_mask, queries, dropout_keep_rate, labels, training_mode=None):
        '''
        Predicts answers as one-hot encoding.
        Returns:
            loss: floating-point number, the loss computed for the batch
            answers: Tensor (None, class size)
        '''
        feed_dict = {self._histories_own: histories_own, self._histories_other: histories_other, self._queries: queries, self._labels: labels,\
        self._histories_own_mask: histories_own_mask, self._histories_other_mask: histories_other_mask, self._dropout: dropout_keep_rate, self._mask: global_mask, self._training:training_mode}
        return self._sess.run([self.loss_op, self.predict_op], feed_dict=feed_dict)