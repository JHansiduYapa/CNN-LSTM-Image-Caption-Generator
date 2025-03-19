class Vocabulary:
    def __init__(self, freq_threshold=1):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.stoi = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}

    def __len__(self):
        return len(self.itos)

    # this will generate the dictionary of words given list of sentences
    def build_vocabulary(self, sentence_list):
        frequencies = {}
        # starting index for new words
        idx = 4
        for sentence in sentence_list:
            # tokenize the sentence using nltk
            tokens = nltk.tokenize.word_tokenize(sentence.lower())
            for token in tokens:
                # if the token is not in the dictionary already make it using 0+1 frequency
                frequencies[token] = frequencies.get(token, 0) + 1
                # if the token is new token that means it frquency is 1 add to the tok-idx and idx-tok
                if frequencies[token] == self.freq_threshold:
                    self.stoi[token] = idx
                    self.itos[idx] = token
                    idx += 1

    # a sententence pass then it convert to numerical representation pass as string 
    def encode(self, text):
        tokenized_text = nltk.tokenize.word_tokenize(text.lower())
        return [self.stoi.get(token, self.stoi["<unk>"]) for token in tokenized_text]

    # convert back to words pass as normal python list
    def decode(self, tokens):
        return [self.itos.get(token, "<unk>") for token in tokens]
        

    # give the token and return the index of that token
    def __call__(self, token):
        return self.stoi.get(token, self.stoi["<unk>"])