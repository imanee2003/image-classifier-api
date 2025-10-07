from flask import Flask
from flask_restful import Resource, Api, reqparse
from werkzeug.datastructures import FileStorage
from predict_resnet50 import predict
import tempfile
import os

app = Flask(__name__)
app.logger.setLevel('INFO')

api = Api(app)

# Parser pour valider les entrées (obligatoire : un fichier image)
parser = reqparse.RequestParser()
parser.add_argument('file',
                    type=FileStorage,
                    location='files',
                    required=True,
                    help='Fournissez un fichier image (JPEG/PNG)')

class Image(Resource):
    def post(self):
        args = parser.parse_args()
        the_file = args['file']
        
        # Sauvegarder temporairement le fichier (Flask ne garde pas en mémoire)
        ofile, ofname = tempfile.mkstemp(suffix='.jpg')  # Suffix pour image
        try:
            the_file.save(ofname)
            
            # Prédire avec notre module
            results = predict(ofname)
            
            # Formater en JSON (decode_predictions retourne des tuples avec float32 ; on convertit)
            output = {'top_categories': []}
            for _, category, score in results:
                output['top_categories'].append({'category': category, 'score': float(score)})
            
            # Nettoyer le fichier temp
            os.close(ofile)
            os.unlink(ofname)
            
            return output
        
        except Exception as e:
            # Nettoyer en cas d'erreur
            if 'ofname' in locals():
                os.close(ofile)
                os.unlink(ofname)
            return {'error': str(e)}, 400  # Erreur 400 : Bad Request

api.add_resource(Image, '/image')

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)