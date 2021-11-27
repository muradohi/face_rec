import os
import pickle

path = r'C:\Users\murad\Downloads\celebsface\Bollywood'

filename = []
for folder in os.listdir(path):
    for img in os.listdir(os.path.join(path,folder)):
        filename.append(os.path.join(path,folder,img))

pickle.dump(filename,open('filename1.pkl','wb'))

from tensorflow.keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import pickle



filename = pickle.load(open('filename1.pkl','rb'))
model = VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')

def feature_ext(image_path,model):
    img = image.load_img(image_path,target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)
    img = preprocess_input(img)

    result = model.predict(img).flatten()

    return result

ext_feat = []
for i in filename:
    ext_feat.append(feature_ext(i,model))

pickle.dump(ext_feat,open('ext_feat1.pkl','wb'))
