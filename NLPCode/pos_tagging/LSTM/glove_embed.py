import numpy as np

class GloveEmbedding():
    def __init__(self, EP):
        self.EP = EP
        

    def create_embedding(self, text, tags, UNKNOWN_TAG, word2idx=None): # its none if it's not the train set
           
        # in tags you can find all the gold labeld tags in sents you can find an array of tokens of type string
        self.sents = text        
        self.tags_li = tags

        # if train set init vocab for embedding layer
        word2idx_g = {}
        if not word2idx == None:
            # if it is not the train set
            self.word2idx = word2idx
            # if it is the train set
        else:
            # source: https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
            # create a word to index using glove
            words = []
            idx = 0
            vectors = []

            # create a dic that given an word outputs the index of the glove
            with open(f'{self.EP.glove_dir}/glove.6B.100d.txt', 'rb') as f:
                for l in f:
                    line = l.decode().split()
                    word = line[0]
                    words.append(word)
                    word2idx_g[word] = idx
                    idx += 1
                    vect = np.array(line[1:]).astype(np.float)
                    vectors.append(vect)

            # dict that given a word outputs a 100 dim vecor according to glove
            glove = {w: vectors[word2idx_g[w]] for w in words}

            # get the vocab size
            # save each token in a word set to get the length
            word_set = set() 
            for sent in self.sents:
                for tok in sent:
                    word_set.add(tok)

            # now create the pretrained embedding weights from glove and a word to index vector (vocab)
            matrix_len = len(word_set)
            weights_matrix = np.zeros((matrix_len + 1, 100))
            words_found = 0
            self.word2idx = {}
            for i, word in enumerate(word_set):
                try: 
                    weights_matrix[i] = glove[word.lower()]
                    words_found += 1
                except KeyError:
                    weights_matrix[i] = np.random.normal(scale=0.6, size=(100, ))
                self.word2idx[word.lower()] = i
            # add an unknown token for future training sets
            self.word2idx[UNKNOWN_TAG] = len(self.word2idx)
            # init the matrix random for this unknown tag
            weights_matrix[-1] = np.random.normal(scale=0.6, size=(100, ))
            self.weights_matrix = weights_matrix
            print(f'found {words_found} out of {len(word_set)} tokens. Thats {(words_found/len(word_set))*100:.2f}%')   
            
        return self.weights_matrix, self.word2idx