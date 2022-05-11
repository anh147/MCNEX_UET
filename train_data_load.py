from types import new_class
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

directory = 'train_data'
classes = ['wrong', 'correct']

# print(os.path.abspath(os.getcwd())) #duong dan hien tai

for i in classes:
    path = os.path.join(directory, i) # path = F:\python\CODE_thu\directory\i
    for img in os.listdir(path): #liet ke anh trong duong dan path
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array, cmap='gray')
        plt.show()
        break
    break

height, weight = np.shape(img_array)
print(height, weight)

start_row,start_col= int((height/10)),int((weight/10)*4)
end_row,end_col= int((height/10)*4),int((weight/10)*6) 
cropped=img_array[start_row:end_row,start_col:end_col]
h, w = np.shape(cropped)
print(h,w)
plt.imshow(cropped, cmap='gray')
plt.show()

training_data = []
def create_training_data():
    for i in classes:
        path = os.path.join(directory,i)
        class_num = classes.index(i)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_img = img_array[start_row:end_row,start_col:end_col]
                training_data.append([new_img, class_num])
                # plt.imshow(new_img, cmap='gray')
                # plt.show()
            except Exception as err:
                pass


create_training_data()
# print(training_data)
import random
random.shuffle(training_data) #tron anh len
# print(training_data)

#tao tap anh va nhan
X=[] #tap anh
y=[] #tap nhan

for img, label in training_data:
    X.append(img)
    y.append(label)

# print(np.shape(X[-1]))
# print(type(X))
X = np.array(X).reshape(-1,int((height/10)*3)*int((weight/10)*2))
y = np.array(y)
# print(type(X))
# print(type(y))


import pickle
#tao file de luu
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