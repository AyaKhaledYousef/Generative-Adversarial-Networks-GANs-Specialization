import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from keras.preprocessing.sequence import pad_sequences
from pickle import load
from keras.models import load_model
from tensorflow.keras.applications.xception import Xception


ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help="Image Path")
args = vars(ap.parse_args())
img_path = args['image']
#convert dictionary to clear list of descriptions
def dict_to_list(descriptions):
   all_desc = []
for key in descriptions.keys():
       [all_desc.append(d) for d in descriptions[key]]
return all_desc
#creating tokenizer class
#this will vectorise text corpus
#each integer will represent token in dictionary
from keras.preprocessing.text import Tokenizer
def create_tokenizer(descriptions):
   desc_list = dict_to_list(descriptions)
   tokenizer = Tokenizer()
   tokenizer.fit_on_texts(desc_list)
return tokenizer
#convert dictionary to clear list of descriptions
# give each word an index, and store that into tokenizer.p pickle file
tokenizer = create_tokenizer(train_descriptions)
dump(tokenizer, open('tokenizer.p', 'wb'))
vocab_size = len(tokenizer.word_index) + 1
print(Vocab_size) #The size of our vocabulary is 7577 words.
#calculate maximum length of descriptions to decide the model structure parameters.
def max_length(descriptions):
   desc_list = dict_to_list(descriptions)
return max(len(d.split()) for d in desc_list)
max_length = max_length(descriptions)
print(Max_length) #Max_length of description is 32

def extract_features(filename, model):
    try:
        image = Image.open(filename)
        print("YYYYY")
        print(image)
        print(image.size)
    except:
        print("ERROR: Can't open image! Ensure that image path and extension is correct")
        image = image.resize((299,299))
        image = np.array(image)
        print(image)
        # for 4 channels images, we need to convert them into 3 channels
    image = np.expand_dims(image, axis=0)
    print(image.shape)
    if image.shape[3] == 4:
        image = image[..., :3]
        image = image/127.5
        image = image - 1.0
        print(image.shape)
        feature = model.predict(image)

    elif image.shape[3]==3:
        image = image/127.5
        image = image - 1.0
        print(image.shape)
        feature = model.predict(image)
    return feature
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None
def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text
max_length = 34
tokenizer = load(open("tokenizer.pkl","rb"))
model = load_model('model_9.h5')
xception_model = Xception(include_top=False, pooling="avg")
photo = extract_features(img_path, xception_model)
img = Image.open(img_path)
description = generate_desc(model, tokenizer, photo, max_length)
print("nn")
print(description)
plt.imshow(img)
