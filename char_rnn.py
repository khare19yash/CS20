import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import random
import sys
sys.path.append('..')
import time

import tensorflow as tf

import utils

def vocab_encode(text, vocab):
    #returning index of character from vocabulary
    return [vocab.index(x) + 1 for x in text if x in vocab]

def vocab_decode(array,vocab):
    #returning character from given index array
    return ''.join([vocab[x - 1] for x in array])

def read_data(filename, vocab, window, overlap):
    lines = [line.strip() for line in open(filename,'r').readlines()]
    while True:
        random.shuffle(lines)
        
        for text in lines:
            text = vocab_encode(text,vocab)
            for start in range(0,len(text) - window, overlap):
                chunk = text[start: start + window]
                chunk += [0] * (window - len(chunk))
                yield chunk

#yields a list of length batch_size in which every element is also a list of length num_time_steps
def read_batch(stream,batch_size):
    batch = []
    for element in stream:
        batch.append(element)
        if len(batch) == batch_size:
            yield batch
            batch = []
    yield batch
    
class CharRNN(object):
    
    def __init__(self):
        self.path = 'data/trump_tweets.txt'
        #vocabulary consisting of all characters used in tweets
        self.vocab = ("$%'()+,-./0123456789:;=?ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                      " '\"_abcdefghijklmnopqrstuvwxyz{|}@#➡📈")
        self.seq = tf.placeholder(tf.int32,[None,None])
        self.temp = tf.constant(1.5)
        self.hidden_sizes = [128,256]
        self.batch_size = 64
        self.lr = 0.0003
        self.skip_step = 1
        self.num_time_steps = 50
        self.len_generated = 200
        self.gstep = tf.Variable(0,dtype=tf.int32,trainable=False,
                                 name='global_step')
        
    def create_rnn(self,seq):
        
        layers = [tf.nn.rnn_cell.GRUCell(size) 
                      for size in self.hidden_sizes]
        cells = tf.nn.rnn_cell.MultiRNNCell(layers)
        batch = tf.shape(seq)[0]
        zero_states = cells.zero_state(batch,dtype=tf.float32)
        self.initial_state = tuple([tf.placeholder_with_default(state,
                                       [None,state.shape[1]])
                                       for state in zero_states])
            
            #calculate real length of sequence
        length = tf.reduce_sum(tf.reduce_max(tf.sign(seq),2),1)
        self.output,self.out_state = tf.nn.dynamic_rnn(cells,seq,
                                                    length,self.initial_state)
            
    def create_model(self):
        seq = tf.one_hot(self.seq,len(self.vocab))
        print(type(seq))
        print(seq.shape)
        self.create_rnn(seq)
        self.logits = tf.layers.dense(self.output,len(self.vocab),None)

        loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits[:,:-1],labels=seq[:,1:])

        self.loss = tf.reduce_sum(loss)

        #sample next character with maxwell-boltzman distribution
        self.sample = tf.multinomial(tf.exp(self.logits[:,-1]/self.temp),
                                    1)[:,0]

        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss,global_step=self.gstep)
    
    def train(self):

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        start = time.time()

        min_loss = None 
        with tf.Session() as sess:
            writer = tf.summary.FileWriter('./graphs/gist',sess.graph)
            sess.run(init)

            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/trump_tweets/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)

            iteration = self.gstep.eval()

            stream = read_data(self.path,self.vocab,self.num_time_steps,overlap = self.num_time_steps // 2)
            data = read_batch(stream,self.batch_size)

            while True:
                batch = next(data)

                batch_loss,_ = sess.run([self.loss,self.optimizer],{self.seq:batch})

                if (iteration + 1) % self.skip_step == 0:
                    print('Iter {}. \n Loss {}. Time {}'.format(iteration,batch_loss,time.time() - start))
                    self.online_infer(sess)
                    start = time.time()
                    checkpoint_name = 'checkpoints/trump_tweets/char-rnn'
                    if min_loss is None:
                        saver.save(sess,checkpoint_name,iteration)
                    elif batch_loss < min_loss:
                        saver.save(sess,checkpoint_name,iteration)
                        min_loss = batch_loss
                iteration += 1
        writer.close()


    def online_infer(self,sess):

        for seed in ['Hillary','I','R','T','@','N','M','.','G','A','W']:
            sentence = seed
            state = None
            for _ in range(self.len_generated):
                batch = [vocab_encode(sentence[-1],self.vocab)]
                feed = {self.seq:batch}
                if state is not None:
                    for i in range(len(state)):
                        feed.update({self.initial_state[i]:state[i]})

                index,state = sess.run([self.sample,self.out_state],feed)
                char = vocab_decode(index,self.vocab)
                sentence += vocab_decode(index,self.vocab)

            #print('\t' + sentence)
            
def main():
    utils.safe_mkdir('checkpoints')
    utils.safe_mkdir('checkpoints/trump_tweets')
    
    lang_model = CharRNN()
    lang_model.create_model()
    lang_model.train()
    
if __name__ == '__main__':
    main()
    
    


                
            
        
        