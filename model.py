from keras.models import load_model
from PIL import image, ImageOps
import numpy as np

def get_class(model_path, labels_path, image_path ):
    np.set_printoptions(suppress=True)
    model = load_model(model_path, compile=Flase)
    class_names = open(labels_path, "r", encoding="utf-8").readlines()
    data = np.ndarray(shape=(1,224,224,3), dtype=np.float32)
    image = image.open(image_path).convert("RBG")
    size = (224,224)
    image = ImageOps.fit(image, size, image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normallized_image_array = (image_array.astype(np.float32)/127.)-1
    data[0] = normallized_image_array
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names(index)
    confidence_score = prediction[0][index]
    return(class_name[2:], confidence_score)
