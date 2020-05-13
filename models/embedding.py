import nltk
import numpy as np
from nltk.corpus import brown, stopwords
from utils.bigram import decode_bigrams, build_bigram_freq, sorted_bigrams, build_bigram_probs
from utils.data import to_json
from utils.sentence import sentence_inverse, sentence_score, next_pred, nearest_neighbours, FakeIdx




class BiGramEmbedding:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.start_idx = ''
        self.end_idx = ''
        self.D = 0
        self.embedding = []

    def build(self, smoothing):
        self._encode_tokens()
        self._encode_sentences()

        self.fake_idx = FakeIdx(self.D, self.start_idx, self.end_idx)

        self._build_bigrams(smoothing)
        self._decode_sort_bigrams()

    def _encode_tokens(self):
        nltk.download('brown')

        idx2word = {
            0: 'START',
            1: 'END',
        }

        rest_idx2word = {idx + 2: token for idx, token in
                         ((idx, token.lower()) for idx, token in enumerate(brown.words()))
                         if token not in stopwords.words('english')
                         }

        self.idx2word = {**idx2word, **rest_idx2word}

        del idx2word
        del rest_idx2word

        self.word2idx = {word: idx for idx, word in self.idx2word.items()}

        self.start_idx, self.end_idx = self.word2idx['START'], self.word2idx['END']

        self.D = len(self.word2idx)

    def _encode_sentences(self):
        self.embedding = [
            [self.word2idx[token.lower()] for token in sentence if token.lower() in self.word2idx]
            for sentence in brown.sents()
        ]

    def _build_bigrams(self, smoothing):
        self.bigram_probs = build_bigram_probs(
            build_bigram_freq(self.embedding, self.start_idx, self.end_idx, smoothing))

    def _decode_sort_bigrams(self):
        self.decoded_bigrams = sorted_bigrams(
            decode_bigrams(self.bigram_probs, self.idx2word))

    def dump_bigrams(self, name):
        to_json(self.bigram_probs, name)

    def dump_decoded_sorted_bigrams(self, name):
        to_json(self.decoded_bigrams, name)

    def sentence_cross_entropy(self, encoded_sents):
        return sentence_score(self.bigram_probs, encoded_sents, self.start_idx, self.end_idx)

    def sentence_decode(self, encoded_sents):
        return sentence_inverse(self.idx2word, encoded_sents)

    def word_pred(self, last_word):
        return next_pred(self.bigram_probs, self.word2idx, self.idx2word, last_word)

    def neighbours(self, last_word, top_n=5):
        return nearest_neighbours(self.bigram_probs, self.word2idx, self.idx2word, last_word, top_n)

    def real_fake(self):
        corpus_encoded_sentence = self.embedding[np.random.choice(len(self.embedding))]
        random_encoded_sentence = self.fake_idx(len(corpus_encoded_sentence))

        real_sentence = self.sentence_decode(corpus_encoded_sentence)
        real_ce = self.sentence_cross_entropy(corpus_encoded_sentence)

        fake_sentence = self.sentence_decode(random_encoded_sentence)
        fake_ce = self.sentence_cross_entropy(random_encoded_sentence)

        return {
            'real': {
                'sentence': real_sentence,
                'ce': real_ce
            },
            'fake': {
                'sentence': fake_sentence,
                'ce': fake_ce
            }
        }
