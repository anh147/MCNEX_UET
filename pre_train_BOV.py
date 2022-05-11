import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.metrics import confusion_matrix
import cv2
import read_files
import vocabulary_helpers
import get_data
import plot_data
import os
import pickle


# categories which have more than 100 images: 80 - train 20 -test.
# 1. airplanes.
# 2. chandelier.
# 3. motorbikes.

# categories which have moret than 90 images: 70 - train 20 - test - butterfly.
# 1. butterlfy.

#categories = ['airplanes', 'chandelier', 'motorbikes', 'butterfly', 'revolver','spoon']

# categories = ['cr', 'wr']

path = 'G:\\Machine Learning\\BagOfVisualWords\\BagOfVisualWords\\train_data_ver2'
start_row, start_col= 270 , 834  #tọa độ để cắt ảnh
end_row, end_col=366, 1021

training_data = [] # tập mảng chứa dữ liệu để đưa vào mô hình
def create_training_data(): #hàm tạo dữ liệu train
    for img in os.listdir(path): 
        try:
            img_array = cv2.imread(os.path.join(path,img))
            new_img = img_array[start_row:end_row,start_col:end_col] #cắt ảnh  
            training_data.append([new_img, img]) #lưu ảnh vào mảng training_data
                # plt.imshow(new_img, cmap='gray')
                # plt.show()
        except Exception as err:
            pass
create_training_data()

print("Parsing image files into categories: ")

train_imgs=[] #tập ảnh
train_labels=[] #tập nhãn

for img, label in training_data:
    train_imgs.append(img)
    train_labels.append(label)

print("Generating vocabulary: ")
(f_vocabulary, i_vocab) = vocabulary_helpers.generate_vocabulary(train_imgs)
# (t_f_vocabulary, t_i_vocab) = vocabulary_helpers.generate_vocabulary(test_imgs)

n_clusters = 1000

# kmeans = vocabulary_helpers.generate_clusters(f_vocabulary, n_clusters)
kmeans = pickle.load(open("G:\\Machine Learning\\BagOfVisualWords\\BagOfVisualWords\\bov_pickle_1000.sav", 'rb'))

print("generating features: ")

print("Creating feature vectors for test and train images from vocabulary set.")
train_data = vocabulary_helpers.generate_features(i_vocab, kmeans, n_clusters)
# test_data = vocabulary_helpers.generate_features(t_i_vocab, kmeans, n_clusters)

# import random
# random.shuffle(train_data) #trộn ảnh
# print(training_data)

#tao tap anh va nhan
X=[] #tập ảnh
y=[] #tập nhãn

for img in train_data:
    X.append(img)
for i in train_labels:
    y.append(i)

# print(np.shape(X[-1]))
# print(type(X))
X = np.array(X).reshape(-1,1000)
y = np.array(y)
# print(type(X))
# print(type(y))



#tạo file để lưu tập X và y
pickle_out = open('X.pickel', 'wb')
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open('y.pickel', 'wb')
pickle.dump(y, pickle_out)
pickle_out.close()

#doc file
pickle_in = open('X.pickel', 'rb')
X_new = pickle.load(pickle_in)

pickle_in = open('y.pickel', 'rb')
y_new = pickle.load(pickle_in)

# print(X_new.shape)
# print(y_new.shape)
# print("Applying SVM classifier.")
# # SVM Classifier.
# clf = svm.SVC()
# fitted = clf.fit(train_data, train_labels)

# #luu clf
# with open('clf.pkl', 'wb') as f: #lưu kết quả vào file clf.pkl
#     pickle.dump(clf, f)

# predict = clf.predict(test_data)

# print("Actual:")
# print(test_labels)

# print("predicted:")
# print(predict)


# # Confusion matrix.
# test_labels = np.asarray(test_labels)
# cnf_matrix = confusion_matrix(predict, test_labels)
# np.set_printoptions(precision=2)


# plot_data.plot_confusion_matrix(cnf_matrix, classes=categories,
#                                 title='Confusion matrix')
# plt.show()
