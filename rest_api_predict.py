from flask import Flask
from flask_restful import Resource, Api, reqparse
from werkzeug.datastructures import FileStorage
from predict_resnet50 import predict  # Importe ton module (doit être dans le même dossier)
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

# Route santé pour Render/AWS (optionnel, pour vérifier si l'app vit)
@app.route('/health')
def health():
    return {'status': 'OK', 'message': 'API Image Classifier prête !'}, 200

if __name__ == '__main__':
    # Adaptation pour Render : utilise PORT env var (set par Render), fallback 5000 local
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)  # host='0.0.0.0' pour cloud ; debug=False en prod