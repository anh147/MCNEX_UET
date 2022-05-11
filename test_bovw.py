from itertools import Predicate
from sklearn.metrics import confusion_matrix
import vocabulary_helpers
from sklearn.model_selection import train_test_split
import pickle
import cv2

cap_check = cv2.VideoCapture(2) # Khai báo USB Camera Check Config
cap_check.set(3, 1920)
cap_check.set(4, 1080)


ret, test_imgs = cap_check.read()

def check_align(img):

    start_row, start_col = 453 , 1091  #tọa độ để cắt ảnh
    end_row, end_col = 593, 1212
    img = test_imgs[start_row:end_row,start_col:end_col] #cắt ảnh

    X=[]
    X.append(img)

    (t_f_vocabulary, t_i_vocab) = vocabulary_helpers.generate_vocabulary(X)

    n_clusters = 1000

    kmeans = pickle.load(open("bov_pickle_1000.sav", 'rb'))

    test_data = vocabulary_helpers.generate_features(t_i_vocab, kmeans, n_clusters)

    with open('clf.pkl', 'rb') as f: #mở file kết quả từ mô hình học máy lưu vào biến clf
        clf = pickle.load(f)

    predict = clf.predict(test_data)

    print("predicted:")
    print(predict)

    return predict