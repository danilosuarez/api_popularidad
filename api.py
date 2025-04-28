from flask import Flask
from flask_restx import Api, Resource, fields
import joblib
import numpy as np

app = Flask(__name__)
api = Api(
    app,
    version='1.0',
    title='Popularidad de Canciones API',
    description='API para predecir la popularidad de una canción basada en características de audio.'
)

ns = api.namespace('predict', description='Predicción de popularidad')

try:
    model = joblib.load('modelo_popularidad.pkl')
except:
    model = None
    print("⚠️ Modelo no cargado.")

parser = ns.parser()
parser.add_argument('danceability', type=float, required=True, help='Danceability', location='args')
parser.add_argument('energy', type=float, required=True, help='Energy', location='args')
parser.add_argument('loudness', type=float, required=True, help='Loudness', location='args')
parser.add_argument('speechiness', type=float, required=True, help='Speechiness', location='args')
parser.add_argument('acousticness', type=float, required=True, help='Acousticness', location='args')
parser.add_argument('instrumentalness', type=float, required=True, help='Instrumentalness', location='args')
parser.add_argument('liveness', type=float, required=True, help='Liveness', location='args')
parser.add_argument('valence', type=float, required=True, help='Valence', location='args')
parser.add_argument('tempo', type=float, required=True, help='Tempo', location='args')

resource_fields = api.model('Prediction', {
    'popularity_prediction': fields.Float,
})

@ns.route('/')
class SongPopularityPredictor(Resource):

    @ns.doc(parser=parser)
    @ns.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        features = [
            args['danceability'],
            args['energy'],
            args['loudness'],
            args['speechiness'],
            args['acousticness'],
            args['instrumentalness'],
            args['liveness'],
            args['valence'],
            args['tempo'],
        ]
        
        if model:
            prediction = model.predict([features])[0]
        else:
            prediction = -1
        
        return {
            'popularity_prediction': prediction
        }, 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
