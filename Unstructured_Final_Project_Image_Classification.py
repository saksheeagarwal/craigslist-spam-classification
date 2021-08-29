from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image 
import matplotlib.pyplot as plt 
from PIL import Image 
import seaborn as sns 
import pandas as pd 
import numpy as np 
import os 

img1 = "C:/MSBA/mgmtAUD/pictures/00w0w_j4FddYdlakT_600x450.jpg" 
img2 = "C:/MSBA/mgmtAUD/pictures/00404_qEiaobIz3Y_600x450.jpg" 
img3 = "C:/MSBA/mgmtAUD/pictures/WeChat Image_20191203202155.jpg"
img4 = "C:/MSBA/mgmtAUD/pictures/WeChat Image_20191203202311.jpg"
imgs = [img1, img2, img3, img4]

def _load_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img 

def _get_predictions(_model):
    f, ax = plt.subplots(1, 4)
    f.set_size_inches(80, 40)
    for i in range(4):
        ax[i].imshow(Image.open(imgs[i]).resize((200, 200), Image.ANTIALIAS))
    plt.show()
    
    f, axes = plt.subplots(1, 4)
    f.set_size_inches(80, 20)
    for i,img_path in enumerate(imgs):
        img = _load_image(img_path)
        preds  = decode_predictions(_model.predict(img), top=3)[0]
        b = sns.barplot(y=[c[1] for c in preds], x=[c[2] for c in preds], color="gray", ax=axes[i])
        b.tick_params(labelsize=55)
        f.tight_layout()

from keras.applications.vgg16 import VGG16
vgg16_weights = 'C:/MSBA/mgmtAUD/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
vgg16_model = VGG16(weights=vgg16_weights)
car = _get_predictions(vgg16_model)
car


from keras.applications.vgg19 import VGG19

vgg19_weights = 'C:/MSBA/mgmtAUD/vgg19_weights_tf_dim_ordering_tf_kernels.h5'
vgg19_model = VGG19(weights=vgg19_weights)
_get_predictions(vgg19_model)


from keras.applications.resnet50 import ResNet50
#resnet_weights = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
resnet_model = ResNet50(weights='imagenet')
_get_predictions(resnet_model)


#Prepare Dataset
basepath = "C:/MSBA/mgmtAUD/pictures/"
class1 = os.listdir(basepath + "BIKEEE/")
class2 = os.listdir(basepath + "Motor/")

data = {'Bike': class1[:10], 
        'Motorcycles': class2[:10], 
        'test': [class1[4], class2[4]]}

from keras.applications.vgg19 import VGG19
vgg19_weights = 'C:/MSBA/mgmtAUD/vgg19_weights_tf_dim_ordering_tf_kernels.h5'
vgg19_model = VGG19(weights=vgg19_weights)


#Feature extraction using VGG19
def _get_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    vgg19_features = vgg19_model.predict(img_data)
    return vgg19_features

# Give labels to the data

features = {"Motorcycles" : [], "Bike" : [], "test" : []}
testimgs = []
for label, val in data.items():
    for k, each in enumerate(val):        
        if label == "test" and k == 0:
            img_path = basepath + "/BIKEEE/" + each
            testimgs.append(img_path)
        elif label == "test" and k == 1:
            img_path = basepath + "/Motor/" + each
            testimgs.append(img_path)
        else: 
            img_path = basepath + label.title() + "/" + each
        feats = _get_features(img_path)
        features[label].append(feats.flatten())   

# #convert the above dictionary format to dataframe which will show labels of each image at the end

dataset = pd.DataFrame()
for label, feats in features.items():
    temp_df = pd.DataFrame(feats)
    temp_df['label'] = label
    dataset = dataset.append(temp_df, ignore_index=True)

#check first few rows of the dataset 
dataset.head()

#prepare predictors and target 
y = dataset[dataset.label != 'test'].label
X = dataset[dataset.label != 'test'].drop('label', axis=1)

from sklearn.feature_selection import VarianceThreshold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

model = MLPClassifier(hidden_layer_sizes=(100, 10))
pipeline = Pipeline([('low_variance_filter', VarianceThreshold()), ('model', model)]) #variance filter to reduce the dimentionality. Other methods : perform PCA / SVD to obtain the dense features.
pipeline.fit(X, y)

print ("Model Trained on pre-trained features")

preds = pipeline.predict(features['test'])

f, ax = plt.subplots(1, 2)
for i in range(2):
    ax[i].imshow(Image.open(testimgs[i]).resize((200, 200), Image.ANTIALIAS))
    ax[i].text(10, 180, 'Predicted: %s' % preds[i], color='k', backgroundcolor='red', alpha=0.8)
plt.show()



