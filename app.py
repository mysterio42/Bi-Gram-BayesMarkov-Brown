from flask import Flask, jsonify, request
from flask_restful import Resource, Api

from models.config import load_embedding
app = Flask(__name__)
api = Api(app)

bigram = load_embedding()


class BayesInference(Resource):

    def post(self):
        posted_json = request.get_json()
        assert 'token' in posted_json
        last_word = posted_json['token']
        preds = bigram.word_pred(last_word)
        if isinstance(preds, tuple):
            current_word, proba = preds
            return jsonify({
                'last_word': last_word,
                'current_word': current_word,
                'proba': proba
            })
        else:
            return jsonify({
                'last_word': last_word,
                'msg': "Word doesn't exist in corpus"
            })


class NearestNeighbours(Resource):

    def post(self):
        posted_json = request.get_json()
        assert 'token' in posted_json
        assert 'top_n' in posted_json

        last_word = posted_json['token']
        top_n = posted_json['top_n']

        word_proba = bigram.neighbours(last_word, top_n)
        if word_proba:
            posted_json['nearest_neighbours'] = {word: proba for word, proba in
                                                 word_proba}
        else:
            posted_json['msg'] = f"{last_word} doesn't exist in corpus"
        return jsonify(posted_json)


class BrownBattle(Resource):

    def get(self):
        return jsonify(
            bigram.real_fake())


api.add_resource(BayesInference, '/inference')
api.add_resource(BrownBattle, '/battle')
api.add_resource(NearestNeighbours, '/neighs')
if __name__ == '__main__':
    app.run(host='0.0.0.0')
