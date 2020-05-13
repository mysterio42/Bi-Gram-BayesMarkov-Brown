import argparse

from models.config import dump_embedding, load_embedding
from models.embedding import BiGramEmbedding
from utils.data import to_csv

def parse_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected')

    parser = argparse.ArgumentParser()

    parser.add_argument('--load', type=str2bool, default=False, required=True,
                        help='True: Load Bi-Gram Embedding model False: Build Bi-Gram Embedding model default: False ')

    parser.add_argument('--word', type=str,
                        help='Bayes Inference  max -> p(current_word|last_word)')

    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Bi-Gram model Smoothing parameter')

    parser.print_help()

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    if args.load:
        bigram_embedding = load_embedding()
    else:
        bigram_embedding = BiGramEmbedding()
        bigram_embedding.build(args.smoothing)
        dump_embedding(bigram_embedding)

    last_word = args.word

    current_word, proba = bigram_embedding.word_pred(last_word)
    ret_inference = {
        'last_word': last_word,
        'current_word': current_word,
        'proba': proba
    }
    print(ret_inference)

    word_proba = bigram_embedding.neighbours(last_word, top_n=224)
    ret_neighs = {word: proba for word, proba in word_proba}
    print(f'{last_word} neighbours {ret_neighs}')

    to_csv('preds',last_word,ret_neighs)

    ret_real_fake = bigram_embedding.real_fake()
    print(ret_real_fake)
