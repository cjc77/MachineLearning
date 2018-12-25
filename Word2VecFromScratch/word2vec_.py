import numpy as np
from collections import defaultdict

class Word2Vec():
    def __init__(self, N):
        self.N = N
        self.V = 0
        self.C = 0
        self.vocab = []
        self.word_index = {}
        self.index_word = {}
        self.W1 = []
        self.W2 = []
        self.h = []
        self.u = []
        self.y = []
        self.loss = 0
    
    def __forward_pass(self, x):
        self.h = np.dot(x, self.W1)
        self.u = np.dot(self.W2.T, self.h)
        self.y = self.softmax(self.u)
    
    def __backprop(self, EI, x, eta):
        dl_dW2 = np.outer(self.h, EI)
        dl_dW1 = np.outer(x, np.dot(self.W2, EI.T))
                
        self.W1 -= eta * dl_dW1
        self.W2 -= eta * dl_dW2
    
    def train(self, X, eta=0.1, epochs=50, verbose=False):
        for i in range(epochs):
            self.loss = 0.0
            
            for x, Yc in X:
                # forward pass
                self.__forward_pass(x)
                
                # calculate error
                EI = np.sum([np.subtract(self.y, yc) for yc in Yc], axis=0)
                
                # backprop
                self.__backprop(EI, x, eta)
                
                # calculate loss
                self.loss += (
                    -np.sum([self.u[np.where(yc == 1)] for yc in Yc]) +
                    Yc.shape[0] * np.log(np.sum(np.exp(self.u)))
                )
                self.loss += (
                    -2 * np.log(Yc.shape[0]) - np.sum([self.u[np.where(yc == 1)] for yc in Yc]) +
                    (Yc.shape[0] * np.log(np.sum(np.exp(self.u))))
                )
            if verbose:
                print("Epoch: {}, Loss: {}".format(i, self.loss))
    
    def context_probs(self, x):
        self.__forward_pass(x)
        return self.y

    def gen_training_data(self, corp, C):
        self.C = C
        wd_counts = defaultdict(int)
        for row in corp:
            for word in row:
                wd_counts[word] += 1
        
        self.V = len(wd_counts.keys())
        
        # Look-up tables
        self.vocab = np.array(sorted(list(wd_counts.keys())))
        self.word_index = dict((word, i) for i, word in enumerate(self.vocab))
        self.index_word = dict((i, word) for i, word in enumerate(self.vocab))
        
        training_data = []
        
        for sentence in corp:
            sentence_len = len(sentence)   
            for i, word in enumerate(sentence):
                w_target = self.to_one_hot(word)
                w_context = []
                
                for j in range(i - C, i + C + 1):
                    if j >= 0 and j != i and j < sentence_len:
                        w_context.append(self.to_one_hot(sentence[j]))
                training_data.append([w_target, np.array(w_context)])
        
        # Prep weight matrices
        self.W1 = np.random.uniform(low=-0.5, high=0.5, size=(self.V, self.N))
        self.W2 = np.random.uniform(low=-0.5, high=0.5, size=(self.N, self.V))
        return np.array(training_data)

    def vec_sim(self, vec, top_n=1):
        word_sim = {}

        for i in range(self.V):
            vec2 = self.W1[i]
            theta_num = np.dot(vec, vec2)
            theta_den = np.linalg.norm(vec) * np.linalg.norm(vec2)
            theta = theta_num / theta_den

            word = self.index_word[i]
            word_sim[word] = theta

        return sorted(word_sim.items(), key=lambda kv: kv[1], reverse=True)[:top_n]

    def word_vec(self, word):
        return self.W1[self.word_index[word]]
    
    def get_one_hot(self, word):
        return (self.vocab == word).astype(int)

    def to_one_hot(self, word):
        word_vec = np.zeros(shape=(self.V,))
        word_vec[self.word_index[word]] = 1
        return word_vec
    
    def softmax(self, z):
        e_z = np.exp(z - np.max(z))
        return e_z / np.sum(e_z, axis=0)