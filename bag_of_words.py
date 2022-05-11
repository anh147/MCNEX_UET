import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.metrics import confusion_matrix

import read_files
import vocabulary_helpers
import get_data
import plot_data
from sklearn.model_selection import train_test_split
import pickle
import os
import cv2



categories = ['wr', 'cr']

# print("Training Images: ")
# print("Reading image files: ")

directory = 'G:\\NCKH\\Xu_ly_anh\\data' # đường dẫn đến file có tên là train_data
classes = ['lech', 'ko_lech']

start_row, start_col = 453 , 1091  #tọa độ để cắt ảnh
end_row, end_col = 593, 1212

training_data = [] # tập mảng chứa dữ liệu để đưa vào mô hình
def create_training_data(): #hàm tạo dữ liệu train
    for i in classes:
        path = os.path.join(directory,i) #chuyển đến file ảnh
        class_num = classes.index(i) #class_num sẽ là 0 hoặc 1
        if i == 'lech':
            for j in os.listdir('G:\\NCKH\\Xu_ly_anh\\data\\lech'):
                path_lech = os.path.join('G:\\NCKH\\Xu_ly_anh\\data\\lech',j) #chuyển đến file ảnh
                for img in os.listdir(path_lech): 
                    try:
                        img_array = cv2.imread(os.path.join(path_lech,img))
                        new_img = img_array[start_row:end_row,start_col:end_col] #cắt ảnh
                        training_data.append([new_img, class_num]) #lưu ảnh vào mảng training_data
                        # plt.imshow(new_img, cmap='gray')
                        # plt.show()
                    except Exception as err:
                        pass
        for img in os.listdir(path): 
            try:
                img_array = cv2.imread(os.path.join(path,img))
                new_img = img_array[start_row:end_row,start_col:end_col] #cắt ảnh
                training_data.append([new_img, class_num]) #lưu ảnh vào mảng training_data
                # plt.imshow(new_img, cmap='gray')
                # plt.show()
            except Exception as err:
                pass
create_training_data()

print("Parsing image files into categories: ")

X=[] #tập ảnh
y=[] #tập nhãn

for img, label in training_data:
    X.append(img)
    y.append(label)

train_imgs, test_imgs, train_labels, test_labels = train_test_split(X, y, test_size=0.12, random_state=101)

#os.mkdir("G:\\NCKH\\Xu_ly_anh\\data_test")
# for i in test_imgs: #lưu ảnh test
#   #  new_img = cv2.imread(i)
#     plt.imshow(i, cmap='gray')
#     plt.show()
 #   cv2.imwrite("G:\\NCKH\\Xu_ly_anh\\data_test\\{}".format(test_imgs.index(i)), new_img)

# train_dict = read_files.get_image_dict(categories, "train")
# test_dict = read_files.get_image_dict(categories, "test")

# print("Parsing image files into categories: ")
# [train_imgs, train_labels] = get_data.get_images_and_labels(train_dict)
# [test_imgs, test_labels] = get_data.get_images_and_labels(test_dict)

print("Generating vocabulary: ")
(f_vocabulary, i_vocab) = vocabulary_helpers.generate_vocabulary(train_imgs)
(t_f_vocabulary, t_i_vocab) = vocabulary_helpers.generate_vocabulary(test_imgs)

n_clusters = 1000

# kmeans = vocabulary_helpers.generate_clusters(f_vocabulary, n_clusters)
kmeans = pickle.load(open("bov_pickle_1000.sav", 'rb'))

print("generating features: ")

print("Creating feature vectors for test and train images from vocabulary set.")
train_data = vocabulary_helpers.generate_features(i_vocab, kmeans, n_clusters)
test_data = vocabulary_helpers.generate_features(t_i_vocab, kmeans, n_clusters)

# import random
# random.shuffle(train_data) #trộn ảnh
# print(training_data)

# tao tap anh va nhan
X=[] #tập ảnh
y=[] #tập nhãn

for img in train_data:
    X.append(img)
for i in train_labels:
    y.append(i)


X = np.array(X).reshape(-1,1000)
y = np.array(y)

# tạo file để lưu tập X và y
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

print(X_new.shape)
print(y_new.shape)
print("Applying SVM classifier.")
# SVM Classifier.
clf = svm.SVC()
fitted = clf.fit(train_data, train_labels)

with open('clf.pkl', 'wb') as f: #lưu kết quả vào file clf.pkl
    pickle.dump(clf, f)

# with open('clf.pkl', 'rb') as f: #mở file kết quả từ mô hình học máy lưu vào biến clf
#     clf = pickle.load(f)

predict = clf.predict(test_data)

print("Actual:")
print(test_labels)

print("predicted:")
print(predict)

print("eror:")
print(predict-test_labels)



# Confusion matrix.
test_labels = np.asarray(test_labels)
cnf_matrix = confusion_matrix(predict, test_labels)
np.set_printoptions(precision=2)

plot_data.plot_confusion_matrix(cnf_matrix, classes=categories,
                                title='Confusion matrix')
plt.show()


for i in predict-test_labels: #lưu ảnh test
    # new_img = cv2.imread(i)
    if i == -1:
        plt.imshow(test_imgs[i-1], cmap='gray')
        plt.show()
        print(test_labels[i-1])
