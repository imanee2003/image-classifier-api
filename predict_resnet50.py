import tensorflow.keras.applications.resnet50 as resnet50
from tensorflow.keras.preprocessing import image
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Pour éviter les warnings sur Mac


def predict(fname):
    """
    Retourne les top 5 catégories pour une image.
    :param fname: Chemin vers le fichier image
    """
    input_shape = (224, 224, 3)
    img = image.load_img(fname, target_size=input_shape[:2])
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    print(f"Forme de l'image après preprocessing: {x.shape}")
    img_array = resnet50.preprocess_input(x)

    model = resnet50.ResNet50(weights='imagenet', input_shape=input_shape)
    preds = model.predict(img_array)
    return resnet50.decode_predictions(preds, top=5)[0]


if __name__ == '__main__':
    import pprint
    import sys

    if len(sys.argv) != 2:
        print("Usage: python predict_resnet50.py <chemin_image>")
        sys.exit(1)

    file_name = sys.argv[1]
    results = predict(file_name)
    pprint.pprint(results)
