a# %%
from IPython.display import clear_output
clear_output()

# %%


# %%
import numpy as np 

# %%
import pandas as pd

# %%
from tqdm import tqdm

# %%
import cv2

# %%
import os

# %%
import glob

# %%
import time

# %%
import shutil

# %%
import itertools

# %%
import imutils

# %%
import matplotlib.pyplot as plt

# %%
import seaborn as sns

# %%
from sklearn.preprocessing import LabelBinarizer

# %%
from sklearn.model_selection import train_test_split

# %%
from sklearn.metrics import accuracy_score, confusion_matrix

# %%
import plotly.graph_objs as go

# %%
from plotly.offline import init_notebook_mode, iplot

# %%
from plotly import tools

# %%
import tensorflow as tf

# %%
from tensorflow.keras.models import Sequential, Model, load_model

# %%
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense,BatchNormalization,Conv2D, MaxPooling2D

# %%
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# %%
from tensorflow.keras import backend as K

# %%
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

# %%
from tensorflow.keras.applications import EfficientNetB2

# %%
from tensorflow.keras.regularizers import l1, l2

# %%
from tensorflow.keras.optimizers import Adamax

# %%
from tensorflow.keras import layers

# %%
from tensorflow.keras.models import Model, Sequential

# %%
from tensorflow.keras.optimizers import Adam, RMSprop

# %%
from tensorflow.keras.callbacks import EarlyStopping

# %%
from sklearn.utils import Bunch

# %%
from sklearn import svm, metrics, datasets

# %%
from sklearn.metrics import confusion_matrix, classification_report

# %%
from sklearn.model_selection import GridSearchCV, train_test_split

# %%
!pip install scikit-image


# %%
from skimage.io import imread

# %%
from skimage.transform import resize

# %%
from pathlib import Path

# %%
from sklearn import tree

# %%
from sklearn.neighbors import KNeighborsClassifier

# %%
from math import *

# %%
init_notebook_mode(connected=True)
RANDOM_SEED = 123

# %%
IMG_PATH = 'Datasets/'

# %%
image = cv2.imread('Datasets/Abnormal/1.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(20, 20))
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(image)


kernel_sharpening = np.array([[-1,-1,-1], 
                              [-1,9,-1], 
                              [-1,-1,-1]])

sharpened = cv2.filter2D(image, -1, kernel_sharpening)


plt.subplot(1, 2, 2)
plt.title("Image Sharpening")
plt.imshow(sharpened)

plt.show()

# %%
image = cv2.imread('Datasets/Normal/2.jpg', 0)

plt.figure(figsize=(30, 30))
plt.subplot(3, 2, 1)
plt.title("Original")
plt.imshow(image)

ret,thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

plt.subplot(3, 2, 2)
plt.title("Threshold Binary")
plt.imshow(thresh1)


image = cv2.GaussianBlur(image, (3, 3), 0)

thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 5) 

plt.subplot(3, 2, 3)
plt.title("Adaptive Mean Thresholding")
plt.imshow(thresh)


_, th2 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

plt.subplot(3, 2, 4)
plt.title("Otsu's Thresholding")
plt.imshow(th2)


plt.subplot(3, 2, 5)
blur = cv2.GaussianBlur(image, (5,5), 0)
_, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
plt.title("Guassian Otsu's Thresholding")
plt.imshow(th3)
plt.show()


# %%
image = cv2.imread('Datasets/Abnormal/3.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(20, 20))
plt.subplot(3, 2, 1)
plt.title("Original")
plt.imshow(image)


kernel = np.ones((5,5), np.uint8)

erosion = cv2.erode(image, kernel, iterations = 1)

plt.subplot(3, 2, 2)
plt.title("Erosion")
plt.imshow(erosion)

# 
dilation = cv2.dilate(image, kernel, iterations = 1)
plt.subplot(3, 2, 3)
plt.title("Dilation")
plt.imshow(dilation)


opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
plt.subplot(3, 2, 4)
plt.title("Opening")
plt.imshow(opening)


closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
plt.subplot(3, 2, 5)
plt.title("Closing")
plt.imshow(closing)

# %%
image = cv2.imread('Datasets/Abnormal/7.jpg', 0)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

height, width,_ = image.shape

sobel_x = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)

plt.figure(figsize=(20, 20))

plt.subplot(3, 2, 1)
plt.title("Original")
plt.imshow(image)

plt.subplot(3, 2, 2)
plt.title("Sobel X")
plt.imshow(sobel_x)


plt.subplot(3, 2, 3)
plt.title("Sobel Y")
plt.imshow(sobel_y)

sobel_OR = cv2.bitwise_or(sobel_x, sobel_y)

plt.subplot(3, 2, 4)
plt.title("sobel_OR")
plt.imshow(sobel_OR)

laplacian = cv2.Laplacian(image, cv2.CV_64F)

plt.subplot(3, 2, 5)
plt.title("Laplacian")
plt.imshow(laplacian)

canny = cv2.Canny(image, 50, 120)

plt.subplot(3, 2, 6)
plt.title("Canny")
plt.imshow(canny)

# %%
image = cv2.imread('Datasets/Abnormal/6.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(20, 20))

plt.subplot(2, 2, 1)
plt.title("Original")
plt.imshow(image)

image_scaled = cv2.resize(image, None, fx=0.75, fy=0.75)

plt.subplot(2, 2, 2)
plt.title("Scaling - Linear Interpolation")
plt.imshow(image_scaled)

img_scaled = cv2.resize(image, None, fx=2, fy=2, interpolation = cv2.INTER_CUBIC)

plt.subplot(2, 2, 3)
plt.title("Scaling - Cubic Interpolation")
plt.imshow(img_scaled)

img_scaled = cv2.resize(image, (900, 400), interpolation = cv2.INTER_AREA)

plt.subplot(2, 2, 4)
plt.title("Scaling - Skewed Size")
plt.imshow(img_scaled)

# %%
image = cv2.imread('Datasets/Abnormal/48.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(20, 20))

plt.subplot(2, 2, 1)
plt.title("Original")
plt.imshow(image)

kalman_3x3 = np.ones((3, 3), np.float32) / 9

blurred = cv2.filter2D(image, -1, kalman_3x3)

plt.subplot(2, 2, 2)
plt.title("Kalman Filter")
plt.imshow(blurred)

kernel_7x7 = np.ones((7, 7), np.float32) / 49

blurred2 = cv2.medianBlur(image, 3)

plt.subplot(2, 2, 3)
plt.title("Median Filter")
plt.imshow(blurred2)

blurred3 = cv2.GaussianBlur(image, (3,3),1)

plt.subplot(2, 2, 4)
plt.title("Gaussian Filter")
plt.imshow(blurred3)

# %%
image = cv2.imread('Datasets/Normal/9.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(20, 20))

plt.subplot(2, 2, 1)
plt.title("Original")
plt.imshow(image)



gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)


edged = cv2.Canny(gray, 30, 200)

plt.subplot(2, 2, 2)
plt.title("Canny Edges")
plt.imshow(edged)



contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

plt.subplot(2, 2, 3)
plt.title("Canny Edges After Contouring")
plt.imshow(edged)

print("Number of Contours found = " + str(len(contours)))


cv2.drawContours(image, contours, -1, (0,255,0), 3)

plt.subplot(2, 2, 4)
plt.title("Contours")
plt.imshow(image)

# %%
image = cv2.imread('Datasets/Abnormal/16.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(20, 20))

plt.subplot(2, 2, 1)
plt.title("Original")
plt.imshow(image)

orig_image = image.copy()



gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)


contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)


for c in contours:
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(orig_image,(x,y),(x+w,y+h),(0,0,255),2)    
    plt.subplot(2, 2, 2)
    plt.title("Bounding Rectangle")
    plt.imshow(orig_image)

    

for c in contours:
    
    accuracy = 0.03 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, accuracy, True)
    cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
    
    plt.subplot(2, 2, 3)
    plt.title("Approx Poly DP")
    plt.imshow(image)

plt.show()
    

# %%
def load_data(dir_path, img_size=(100,100)):
    
    X = []
    y = []
    i = 0
    labels = dict()
    for path in tqdm(sorted(os.listdir(dir_path))):
        if not path.startswith('.'):
            labels[i] = path
            for file in os.listdir(dir_path + path):
                if not file.startswith('.'):
                    img = cv2.imread(dir_path + path + '/' + file)
                    X.append(img)
                    y.append(i)
            i += 1
    X = np.array(X)
    y = np.array(y)
    print(f'{len(X)} images loaded from {dir_path} directory.')
    return X, y, labels





# %%
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    plt.figure(figsize = (6,6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    cm = np.round(cm,2)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# %%
def print_in_color(txt_msg,fore_tupple,back_tupple,):
    rf,gf,bf=fore_tupple
    rb,gb,bb=back_tupple
    msg='{0}' + txt_msg
    mat='\33[38;2;' + str(rf) +';' + str(gf) + ';' + str(bf) + ';48;2;' + str(rb) + ';' +str(gb) + ';' + str(bb) +'m' 
    print(msg .format(mat), flush=True)
    print('\33[0m', flush=True) 
    return

# %%
class LRA(tf.keras.callbacks.Callback):
    def __init__(self,model, base_model, patience,stop_patience, threshold, factor, dwell, batches, initial_epoch,epochs, ask_epoch):
        super(LRA, self).__init__()
        self.my_model = model
        self.base_model=base_model
        self.patience=patience 
        self.stop_patience=stop_patience 
        self.threshold=threshold 
        self.factor=factor 
        self.dwell=dwell
        self.batches=batches 
        self.initial_epoch=initial_epoch
        self.epochs=epochs
        self.ask_epoch=ask_epoch
        self.ask_epoch_initial=ask_epoch 
        self.count=0 
        self.stop_count=0        
        self.best_epoch=1         

        # ✅ FIX 1: learning_rate instead of lr
        self.initial_lr=float(tf.keras.backend.get_value(model.optimizer.learning_rate))          

        self.highest_tracc=0.0 
        self.lowest_vloss=np.inf 

        # ✅ FIX 2: use my_model instead of model
        self.best_weights=self.my_model.get_weights() 
        self.initial_weights=self.my_model.get_weights()   

    def on_train_begin(self, logs=None):        
        if self.base_model != None:
            status=self.base_model.trainable   # ✅ FIX 3
            if status:
                msg=' initializing callback starting train with base_model trainable'
            else:
                msg='initializing callback starting training with base_model not trainable'
        else:
            msg='initialing callback and starting training'                        
        print_in_color (msg, (244, 252, 3), (55,65,80)) 
        msg='{0:^8s}{1:^10s}{2:^9s}{3:^9s}{4:^9s}{5:^9s}{6:^9s}{7:^10s}{8:10s}{9:^8s}'.format('Epoch', 'Loss', 'Accuracy',
                                                                                              'V_loss','V_acc', 'LR', 'Next LR', 'Monitor','% Improv', 'Duration')
        print_in_color(msg, (244,252,3), (55,65,80)) 
        self.start_time= time.time()
        
    def on_train_end(self, logs=None):
        stop_time=time.time()
        tr_duration= stop_time- self.start_time            

        # ✅ FIX 4
        self.my_model.set_weights(self.best_weights) 

        msg=f'Training is completed - model is set with weights from epoch {self.best_epoch} '
        print_in_color(msg, (0,255,0), (55,65,80))

    def on_epoch_begin(self,epoch, logs=None):
        self.now= time.time()
        
    def on_epoch_end(self, epoch, logs=None):  
        later=time.time()
        duration=later-self.now 

        # ✅ FIX 5: learning_rate
        lr=float(tf.keras.backend.get_value(self.my_model.optimizer.learning_rate)) 
        current_lr=lr

        v_loss=logs.get('val_loss')  
        acc=logs.get('accuracy')  
        v_acc=logs.get('val_accuracy')
        loss=logs.get('loss')        

        if acc < self.threshold: 
            monitor='accuracy'
            if acc>self.highest_tracc:                
                self.highest_tracc=acc 
                self.best_weights=self.my_model.get_weights() 
                self.count=0 
                self.stop_count=0
                color= (0,255,0)
                self.best_epoch=epoch + 1               
            else: 
                if self.count>=self.patience -1:
                    color=(245, 170, 66)
                    lr= lr* self.factor 

                    # ✅ FIX 6
                    tf.keras.backend.set_value(self.my_model.optimizer.learning_rate, lr) 

                    self.count=0 
                    self.stop_count=self.stop_count + 1 
                    if self.dwell:
                        self.my_model.set_weights(self.best_weights)                        
                else:
                    self.count=self.count +1                  
        else: 
            monitor='val_loss'
            if v_loss< self.lowest_vloss: 
                self.lowest_vloss=v_loss               
                self.best_weights=self.my_model.get_weights() 
                self.count=0 
                self.stop_count=0  
                color=(0,255,0)                
                self.best_epoch=epoch + 1 
            else: 
                if self.count>=self.patience-1: 
                    color=(245, 170, 66)
                    lr=lr * self.factor                    

                    # ✅ FIX 7
                    tf.keras.backend.set_value(self.my_model.optimizer.learning_rate, lr) 

                    self.stop_count=self.stop_count + 1 
                    self.count=0 
                    if self.dwell:
                        self.my_model.set_weights(self.best_weights) 
                else: 
                    self.count =self.count +1                   

        msg=f'{str(epoch+1):^3s}/{str(self.epochs):4s} {loss:^9.3f}{acc*100:^9.3f}{v_loss:^9.5f}{v_acc*100:^9.3f}{current_lr:^9.5f}{lr:^9.5f}{monitor:^11s}{0:^10.2f}{duration:^8.2f}'
        print_in_color (msg,color, (55,65,80))

        if self.stop_count> self.stop_patience - 1: 
            msg=f' training has been halted at epoch {epoch + 1}'
            print_in_color(msg, (0,255,255), (55,65,80))

            # ✅ FIX 8
            self.my_model.stop_training = True 

# %%
def tr_plot(tr_data, start_epoch):
    tacc=tr_data.history['accuracy']
    tloss=tr_data.history['loss']
    vacc=tr_data.history['val_accuracy']
    vloss=tr_data.history['val_loss']
    Epoch_count=len(tacc)+ start_epoch
    Epochs=[]
    for i in range (start_epoch ,Epoch_count):
        Epochs.append(i+1)   
    index_loss=np.argmin(vloss)
    val_lowest=vloss[index_loss]
    index_acc=np.argmax(vacc)
    acc_highest=vacc[index_acc]
    plt.style.use('fivethirtyeight')
    sc_label='best epoch= '+ str(index_loss+1 +start_epoch)
    vc_label='best epoch= '+ str(index_acc + 1+ start_epoch)
    fig,axes=plt.subplots(nrows=1, ncols=2, figsize=(20,8))
    axes[0].plot(Epochs,tloss, 'r', label='Training loss')
    axes[0].plot(Epochs,vloss,'g',label='Validation loss' )
    axes[0].scatter(index_loss+1 +start_epoch,val_lowest, s=150, c= 'blue', label=sc_label)
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[1].plot (Epochs,tacc,'r',label= 'Training Accuracy')
    axes[1].plot (Epochs,vacc,'g',label= 'Validation Accuracy')
    axes[1].scatter(index_acc+1 +start_epoch,acc_highest, s=150, c= 'blue', label=vc_label)
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    plt.tight_layout
    #plt.style.use('fivethirtyeight')
    plt.show()

# %%
def print_info( test_gen, preds, print_code, save_dir, subject ):
    class_dict=test_gen.class_indices
    labels= test_gen.labels
    file_names= test_gen.filenames 
    error_list=[]
    true_class=[]
    pred_class=[]
    prob_list=[]
    new_dict={}
    error_indices=[]
    y_pred=[]
    for key,value in class_dict.items():
        new_dict[value]=key             # dictionary {integer of class number: string of class name}
    # store new_dict as a text fine in the save_dir
    classes=list(new_dict.values())     # list of string of class names     
    errors=0      
    for i, p in enumerate(preds):
        pred_index=np.argmax(p)         
        true_index=labels[i]  # labels are integer values
        if pred_index != true_index: # a misclassification has occurred
            error_list.append(file_names[i])
            true_class.append(new_dict[true_index])
            pred_class.append(new_dict[pred_index])
            prob_list.append(p[pred_index])
            error_indices.append(true_index)            
            errors=errors + 1
        y_pred.append(pred_index)    
    if print_code !=0:
        if errors>0:
            if print_code>errors:
                r=errors
            else:
                r=print_code           
            msg='{0:^28s}{1:^28s}{2:^28s}{3:^16s}'.format('Filename', 'Predicted Class' , 'True Class', 'Probability')
            print_in_color(msg, (0,255,0),(55,65,80))
            for i in range(r):                
                split1=os.path.split(error_list[i])                
                split2=os.path.split(split1[0])                
                fname=split2[1] + '/' + split1[1]
                msg='{0:^28s}{1:^28s}{2:^28s}{3:4s}{4:^6.4f}'.format(fname, pred_class[i],true_class[i], ' ', prob_list[i])
                print_in_color(msg, (255,255,255), (55,65,60))
                #print(error_list[i]  , pred_class[i], true_class[i], prob_list[i])               
        else:
            msg='With accuracy of 100 % there are no errors to print'
            print_in_color(msg, (0,255,0),(55,65,80))
    if errors>0:
        plot_bar=[]
        plot_class=[]
        for  key, value in new_dict.items():        
            count=error_indices.count(key) 
            if count!=0:
                plot_bar.append(count) # list containg how many times a class c had an error
                plot_class.append(value)   # stores the class 
        fig=plt.figure()
        fig.set_figheight(len(plot_class)/3)
        fig.set_figwidth(10)
        plt.style.use('fivethirtyeight')
        for i in range(0, len(plot_class)):
            c=plot_class[i]
            x=plot_bar[i]
            plt.barh(c, x, )
            plt.title( ' Errors by Class on Test Set')
    y_true= np.array(labels)        
    y_pred=np.array(y_pred)
    if len(classes)<= 30:
        cm = confusion_matrix(y_true, y_pred )        
        length=len(classes)
        if length<8:
            fig_width=8
            fig_height=8
        else:
            fig_width= int(length * .5)
            fig_height= int(length * .5)
        plt.figure(figsize=(fig_width, fig_height))
        sns.heatmap(cm, annot=True, vmin=0, fmt='g', cmap='Blues', cbar=False)       
        plt.xticks(np.arange(length)+.5, classes, rotation= 90)
        plt.yticks(np.arange(length)+.5, classes, rotation=0)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()
    clr = classification_report(y_true, y_pred, target_names=classes)
    print("Classification Report:\n----------------------\n", clr)

# %%
def saver(save_path, model, model_name, subject, accuracy,img_size, scalar, generator):
    save_id=str (model_name +  '-' + subject +'-'+ str(acc)[:str(acc).rfind('.')+3] + '.h5')
    model_save_loc=os.path.join(save_path, save_id)
    model.save(model_save_loc)
    print_in_color ('model was saved as ' + model_save_loc, (0,255,0),(55,65,80))     
    class_dict=generator.class_indices 
    height=[]
    width=[]
    scale=[]
    for i in range(len(class_dict)):
        height.append(img_size[0])
        width.append(img_size[1])
        scale.append(scalar)
    Index_series=pd.Series(list(class_dict.values()), name='class_index')
    Class_series=pd.Series(list(class_dict.keys()), name='class') 
    Height_series=pd.Series(height, name='height')
    Width_series=pd.Series(width, name='width')
    Scale_series=pd.Series(scale, name='scale by')
    class_df=pd.concat([Index_series, Class_series, Height_series, Width_series, Scale_series], axis=1)    
    csv_name='class_dict.csv'
    csv_save_loc=os.path.join(save_path, csv_name)
    class_df.to_csv(csv_save_loc, index=False) 
    print_in_color ('class csv file was saved as ' + csv_save_loc, (0,255,0),(55,65,80)) 
    return model_save_loc, csv_save_loc

# %%
def predictor(sdir, csv_path,  model_path, crop_image = False):    
    class_df=pd.read_csv(csv_path)    
    img_height=int(class_df['height'].iloc[0])
    img_width =int(class_df['width'].iloc[0])
    img_size=(img_width, img_height)
    scale=class_df['scale by'].iloc[0] 
    try: 
        s=int(scale)
        s2=1
        s1=0
    except:
        split=scale.split('-')
        s1=float(split[1])
        s2=float(split[0].split('*')[1]) 
        print (s1,s2)
    path_list=[]
    paths=os.listdir(sdir)
    for f in paths:
        path_list.append(os.path.join(sdir,f))
    print (' Model is being loaded- this will take about 10 seconds')
    model=load_model(model_path)
    image_count=len(path_list)    
    index_list=[] 
    prob_list=[]
    cropped_image_list=[]
    good_image_count=0
    for i in range (image_count):       
        img=cv2.imread(path_list[i])
        if crop_image == True:
            status, img=crop(img)
        else:
            status=True
        if status== True:
            good_image_count +=1
            img=cv2.resize(img, img_size)            
            cropped_image_list.append(img)
            img=img*s2 - s1
            img=np.expand_dims(img, axis=0)
            p= np.squeeze (model.predict(img))           
            index=np.argmax(p)            
            prob=p[index]
            index_list.append(index)
            prob_list.append(prob)
    if good_image_count==1:
        class_name= class_df['class'].iloc[index_list[0]]
        probability= prob_list[0]
        img=cropped_image_list [0] 
        plt.title(class_name, color='blue', fontsize=16)
        plt.axis('off')
        plt.imshow(img)
        return class_name, probability
    elif good_image_count == 0:
        return None, None
    most=0
    for i in range (len(index_list)-1):
        key= index_list[i]
        keycount=0
        for j in range (i+1, len(index_list)):
            nkey= index_list[j]            
            if nkey == key:
                keycount +=1                
        if keycount> most:
            most=keycount
            isave=i             
    best_index=index_list[isave]    
    psum=0
    bestsum=0
    for i in range (len(index_list)):
        psum += prob_list[i]
        if index_list[i]==best_index:
            bestsum += prob_list[i]  
    img= cropped_image_list[isave]/255    
    class_name=class_df['class'].iloc[best_index]
    plt.title(class_name, color='blue', fontsize=16)
    plt.axis('off')
    plt.imshow(img)
    return class_name, bestsum/image_count

# %%
def trim (df, max_size, min_size, column):
    df=df.copy()
    sample_list=[] 
    groups=df.groupby(column)
    for label in df[column].unique():        
        group=groups.get_group(label)
        sample_count=len(group)         
        if sample_count> max_size :
            samples=group.sample(max_size, replace=False, weights=None, random_state=123, axis=0).reset_index(drop=True)
            sample_list.append(samples)
        elif sample_count>= min_size:
            sample_list.append(group)
    df=pd.concat(sample_list, axis=0).reset_index(drop=True)
    balance=list(df[column].value_counts())
    print (balance)
    return df

# %%
def preprocess (sdir, trsplit, vsplit):
    filepaths=[]
    labels=[]    
    classlist=os.listdir(sdir)
    for klass in classlist:
        classpath=os.path.join(sdir,klass)
        flist=os.listdir(classpath)
        for f in flist:
            fpath=os.path.join(classpath,f)
            filepaths.append(fpath)
            labels.append(klass)
    Fseries=pd.Series(filepaths, name='filepaths')
    Lseries=pd.Series(labels, name='labels')
    df=pd.concat([Fseries, Lseries], axis=1)        
    dsplit=vsplit/(1-trsplit)
    strat=df['labels']    
    train_df, dummy_df=train_test_split(df, train_size=trsplit, shuffle=True, random_state=123, stratify=strat)
    strat=dummy_df['labels']
    valid_df, test_df=train_test_split(dummy_df, train_size=dsplit, shuffle=True, random_state=123, stratify=strat)
    print('train_df length: ', len(train_df), '  test_df length: ',len(test_df), '  valid_df length: ', len(valid_df))
    print(train_df['labels'].value_counts())
    return train_df, test_df, valid_df

# %%
import os

for item in os.listdir("./Datasets"):
    print(item)
    

# %%
for folder in os.listdir("./Datasets"):
    path = os.path.join("./Datasets", folder)
    print(folder, "->", os.listdir(path))

# %%
import os

for root, dirs, files in os.walk("./Datasets"):
    for name in dirs:
        full = os.path.join(root, name)
        if not os.path.isdir(full):
            print("Removing bad dir:", full)
            os.rmdir(full)

# %%
def balance(train_df,max_samples, min_samples, column, working_dir, image_size):
    train_df=train_df.copy()
    train_df=trim (train_df, max_samples, min_samples, column)    
    aug_dir=os.path.join(working_dir, 'aug')
    if os.path.isdir(aug_dir):
        shutil.rmtree(aug_dir)
    os.mkdir(aug_dir)
    for label in train_df['labels'].unique():    
        dir_path=os.path.join(aug_dir,label)    
        os.mkdir(dir_path)
    total=0
    gen=ImageDataGenerator(horizontal_flip=True,  rotation_range=20, width_shift_range=.2,
                                  height_shift_range=.2, zoom_range=.2)
    groups=train_df.groupby('labels') 
    for label in train_df['labels'].unique():         
        group=groups.get_group(label)  
        sample_count=len(group)   
        if sample_count< max_samples: 
            aug_img_count=0
            delta=max_samples-sample_count  
            target_dir=os.path.join(aug_dir, label)     
            aug_gen=gen.flow_from_dataframe( group,  x_col='filepaths', y_col=None, target_size=image_size,
                                            class_mode=None, batch_size=1, shuffle=False, 
                                            save_to_dir=target_dir, save_prefix='aug-', color_mode='rgb',
                                            save_format='jpg')
            while aug_img_count<delta:
                images=next(aug_gen)            
                aug_img_count += len(images)
            total +=aug_img_count
    print('Total Augmented images created= ', total)
    if total>0:
        aug_fpaths=[]
        aug_labels=[]
        classlist=os.listdir(aug_dir)
        for klass in classlist:
            classpath=os.path.join(aug_dir, klass)     
            flist=os.listdir(classpath)    
            for f in flist:        
                fpath=os.path.join(classpath,f)         
                aug_fpaths.append(fpath)
                aug_labels.append(klass)
        Fseries=pd.Series(aug_fpaths, name='filepaths')
        Lseries=pd.Series(aug_labels, name='labels')
        aug_df=pd.concat([Fseries, Lseries], axis=1)
        ndf=pd.concat([train_df,aug_df], axis=0).reset_index(drop=True)
    else:
        ndf=train_df
    print (list(ndf['labels'].value_counts()) )
    return ndf 

# %%
train_df, test_df, valid_df = preprocess("./Datasets", .8, .1)

# %%
max_samples= 410
min_samples=0
column='labels'
new_dataset_dir = r'./'
img_size=(224,224)
ndf=balance(train_df,max_samples, min_samples, column, new_dataset_dir, img_size)

# %%
channels=3
batch_size=30
img_shape=(img_size[0], img_size[1], channels)
length=len(test_df)

# %%
test_batch_size=sorted([int(length/n) for n in range(1,length+1) if length % n ==0 and length/n<=80],reverse=True)[0]  
test_steps=int(length/test_batch_size)
print ( 'test batch size: ' ,test_batch_size, '  test steps: ', test_steps)

# %%
def scalar(img):    
    return img

# %%
trgen=ImageDataGenerator(preprocessing_function=scalar, horizontal_flip=True)

# %%
tvgen=ImageDataGenerator(preprocessing_function=scalar)

# %%
train_gen=trgen.flow_from_dataframe( ndf, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical',
                                    color_mode='rgb', shuffle=True, batch_size=batch_size)

# %%
test_gen=tvgen.flow_from_dataframe( test_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical',
                                    color_mode='rgb', shuffle=False, batch_size=test_batch_size)

# %%
valid_gen=tvgen.flow_from_dataframe( valid_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical',
                                    color_mode='rgb', shuffle=True, batch_size=batch_size)

# %%
classes=list(train_gen.class_indices.keys())

# %%
class_count=len(classes)

# %%
train_steps=int(np.ceil(len(train_gen.labels)/batch_size))

# %%
def show_image_samples(gen ):
    t_dict=gen.class_indices
    classes=list(t_dict.keys())    
    images,labels=next(gen) 
    plt.figure(figsize=(20, 20))
    length=len(labels)
    if length<25:   
        r=length
    else:
        r=25
    for i in range(r):
        plt.subplot(5, 5, i + 1)
        image=images[i]/255
        plt.imshow(image)
        index=np.argmax(labels[i])
        class_name=classes[index]
        plt.title(class_name, color='blue', fontsize=12)
        plt.axis('off')
    plt.show()

# %%
show_image_samples(train_gen)

# %%
model_name='Skin-Disease-Model'

# %%
base_model=EfficientNetB2(include_top=False, weights="imagenet",input_shape=img_shape, pooling='max') 

# %%
custom_model=base_model.output

# %%
custom_model=BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 )(custom_model)

# %%
custom_model = Dense(
    256,
    kernel_regularizer = l2(0.016),
    activity_regularizer = l1(0.006),
    bias_regularizer = l1(0.006),
    activation='relu'
)(custom_model)

# %%
custom_model=Dropout(rate=.45, seed=123)(custom_model)   

# %%
output=Dense(class_count, activation='softmax')(custom_model)

# %%
model=Model(inputs=base_model.input, outputs=output)

# %%
model.compile(
    Adamax(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# %%
model.summary()

# %%
epochs =40

# %%
patience= 1

# %%
stop_patience =3 

# %%
threshold=.9

# %%
factor=.5

# %%
dwell=True

# %%
freeze=False

# %%
epochs=5

# %%
batches=train_steps

# %%
callbacks=[LRA(model=model,base_model= base_model,patience=patience,stop_patience=stop_patience, threshold=threshold,
                   factor=factor,dwell=dwell, batches=batches,initial_epoch=0,epochs=epochs, ask_epoch=epochs )]

# %%
history=model.fit(x=train_gen,  epochs=epochs, verbose=0, callbacks=callbacks,  validation_data=valid_gen,
               validation_steps=None,  shuffle=False,  initial_epoch=0)

# %%
tr_plot(history,0)
subject='ulcers'
acc=model.evaluate( test_gen, batch_size=test_batch_size, verbose=1, steps=test_steps, return_dict=False)[1]*100
msg=f'accuracy on the test set is {acc:5.2f} %'
print_in_color(msg, (0,255,0),(55,65,80))
generator=train_gen
scale = 1
model_save_loc, csv_save_loc=saver("./", model, model_name, subject, acc, img_size, scale,  generator)

# %%
print_code=0
preds=model.predict(test_gen) 
print_info( test_gen, preds, print_code, "./", subject ) 

# %%

import os
print(os.path.exists('./TestSet/1.jpg'))

# %%
import os

os.makedirs("./TestSet", exist_ok=True)

# %%
import os

print("Folder exists:", os.path.exists('./TestSet'))
print("File exists:", os.path.exists('./TestSet/1.jpg'))

# %%
import shutil

shutil.copy("Datasets/Normal/1.jpg", "./TestSet/1.jpg")

# %%
import shutil
import os

os.makedirs("./TestSet", exist_ok=True)
shutil.copy("Datasets/Normal/1.jpg", "./TestSet/test.jpg")

# %%
img_path = './TestSet/test.jpg'


# %%
import cv2

img = cv2.imread(img_path)
print("Loaded:", img is not None)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# %%
store_path=os.path.join("./", 'storage')
if os.path.isdir(store_path):
    shutil.rmtree(store_path)
os.mkdir(store_path)
img_path=r'./TestSet/1.jpg'
img=cv2.imread(img_path,  cv2.IMREAD_REDUCED_COLOR_2)
img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
file_name=os.path.split(img_path)[1]
dst_path=os.path.join(store_path, file_name)
cv2.imwrite(dst_path, img)
print (os.listdir(store_path))

# %%


# %%
csv_path=csv_save_loc 
model_path=model_save_loc 
class_name, probability=predictor(store_path, csv_path,  model_path, crop_image = False) 
msg=f' image is of class {class_name} with a probability of {probability * 100: 6.2f} %'
print_in_color(msg, (0,255,255), (65,85,55))

# %%
import cv2
import numpy as np

def resize_image(img, size=(224,224)):
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def normalize_img(img):
    # keep pixel range 0-1 float32
    img = img.astype('float32') / 255.0
    return img

def remove_hair(img, kernel_size=17):
    """Simple hair removal via morphological closing + inpainting.
    img: RGB uint8 image
    returns: RGB uint8 image with reduced hair lines
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Black-hat to reveal hair-like dark lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    # Dilate threshold to cover hair width
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    # Inpaint on RGB image
    inpaint = cv2.inpaint(img, dilated, 1, cv2.INPAINT_TELEA)
    return inpaint

def gaussian_denoise(img, ksize=3):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

# %%
from tensorflow.keras import layers, Model, Input

def build_unet(input_shape=(224,224,3), filters=32):
    inp = Input(input_shape)
    # Encoder
    c1 = layers.Conv2D(filters, 3, activation='relu', padding='same')(inp)
    c1 = layers.Conv2D(filters, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D(2)(c1)

    c2 = layers.Conv2D(filters*2, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(filters*2, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D(2)(c2)

    c3 = layers.Conv2D(filters*4, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(filters*4, 3, activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D(2)(c3)

    # Bottleneck
    b = layers.Conv2D(filters*8, 3, activation='relu', padding='same')(p3)
    b = layers.Conv2D(filters*8, 3, activation='relu', padding='same')(b)

    # Decoder
    u3 = layers.UpSampling2D(2)(b)
    u3 = layers.Concatenate()([u3, c3])
    c4 = layers.Conv2D(filters*4, 3, activation='relu', padding='same')(u3)
    c4 = layers.Conv2D(filters*4, 3, activation='relu', padding='same')(c4)

    u2 = layers.UpSampling2D(2)(c4)
    u2 = layers.Concatenate()([u2, c2])
    c5 = layers.Conv2D(filters*2, 3, activation='relu', padding='same')(u2)
    c5 = layers.Conv2D(filters*2, 3, activation='relu', padding='same')(c5)

    u1 = layers.UpSampling2D(2)(c5)
    u1 = layers.Concatenate()([u1, c1])
    c6 = layers.Conv2D(filters, 3, activation='relu', padding='same')(u1)
    c6 = layers.Conv2D(filters, 3, activation='relu', padding='same')(c6)

    out = layers.Conv2D(1, 1, activation='sigmoid')(c6)  # binary mask
    model = Model(inp, out)
    return model

# Example compile
# unet = build_unet(input_shape=(224,224,3))
# unet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# unet.summary()

# %%
def segment_image(img_rgb, unet_model=None, target_size=(224,224), threshold=0.5):
    """Returns (mask_binary_uint8, masked_rgb)
    mask: HxW single channel uint8 (0 or 255)
    masked_rgb: original image where background is zeroed
    """
    # Preproc
    img_resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_AREA)
    inp = img_resized.astype('float32') / 255.0
    if unet_model is None:
        # fallback: simple Otsu on grayscale
        gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = (m > 0).astype('uint8') * 255
    else:
        p = unet_model.predict(np.expand_dims(inp, axis=0))[0,...,0]
        mask = (p >= threshold).astype('uint8') * 255

    # resize mask back to original image shape
    mask_full = cv2.resize(mask, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    # apply to original image
    masked = img_rgb.copy()
    masked[mask_full == 0] = 0
    return mask_full, masked

# %%
def crop_from_mask(img_rgb, mask, pad=10):
    # mask is uint8 0/255
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return False, img_rgb  # nothing found
    x1, x2 = np.min(xs), np.max(xs)
    y1, y2 = np.min(ys), np.max(ys)
    h, w = img_rgb.shape[:2]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    crop = img_rgb[y1:y2, x1:x2]
    return True, crop

# %%
# load models (once)
# unet = load_model('path/to/unet_weights.h5')   # if you trained and saved it
# classifier = load_model(model_save_loc)        # your existing classifier

def infer_image_pipeline(img_path, unet_model, classifier_model, classifier_input_size=(224,224)):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask, masked_img = segment_image(img, unet_model=unet_model, target_size=classifier_input_size)
    ok, crop = crop_from_mask(img, mask, pad=10)
    if not ok:
        # fallback: use full image resized
        roi = resize_image(img, classifier_input_size)
    else:
        roi = resize_image(crop, classifier_input_size)
    # match the scaling used by your training scalar function
    # if your model expects 0-255 or [-1,1], adapt here. Your scalar() returns img as-is, so:
    inp = np.expand_dims(roi.astype('float32'), axis=0)
    preds = classifier_model.predict(inp)
    idx = np.argmax(preds[0])
    prob = preds[0][idx]
    return idx, prob, mask, masked_img, roi


