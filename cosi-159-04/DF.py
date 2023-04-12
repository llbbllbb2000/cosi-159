from deepface import DeepFace
import math
# import matplotlib.pyplot as plt
# from PIL import Image, ImageDraw
# import os
# import cv2
import numpy as np
import pandas as pd

# def show_img(imgs: list, img_names: list) -> None:
#     imgs_count = len(imgs)
#     for i in range(imgs_count):
#         ax = plt.subplot(1, imgs_count, i+1)
#         ax.imshow(imgs[i])
#         ax.set_title(img_names[i])
#         ax.set_xticks([])
#         ax.set_yticks([])
#     plt.tight_layout(h_pad=3)
#     plt.show()

# img_path = "images"
# for person_dir in os.listdir(img_path):
#     imgs = []
#     img_names = []
#     for file in os.listdir(os.path.join(img_path, person_dir)):
#         imgs.append(Image.open(os.path.join(img_path, person_dir, file)))
#         img_names.append(person_dir + '/' + file)
#     show_img(imgs, img_names)

# img_paths = [["./images/baijingting/0000.jpg", "./images/baijingting/0001.jpg"],
#              ["./images/baijingting/0000.jpg", "./images/zhaoliying/0001.jpg"],
#              ["./images/pengyuyan/0000.jpg", "./images/pengyuyan/0001.jpg"],
#              ["./images/Jim Carrey/0000.jpg", "./images/zhangziyi/0001.jpg"]]

# print("For the face verification:")
# for img1, img2 in img_paths :
#     print(DeepFace.verify(img1, img2)['verified'])

val = pd.read_csv("./fairface_label_val.csv")
val = val[['file', 'age', 'race']]
# val = val.iloc[:100]

# print(val)
age_map = {
    '0-2' : 0,
    '3-9' : 1,
    '10-19' : 2,
    '20-29' : 3,
    '30-39' : 4,
    '40-49' : 5,
    '50-59' : 6,
    '60-69' : 7,
    'more than 70' : 8
}
race_map = {'Black' : 0, 'black' : 0, 
            'East Asian' : 1, 'Southeast Asian' : 1, 'asian' : 1,
            'Indian' : 2, 'indian' : 2,
            'Latino_Hispanic' : 3, 'latino hispanic' : 3,
            'Middle Eastern' : 4, 'middle eastern' : 4,
            'White' : 5, 'white' : 5}

correct = np.zeros((len(age_map), len(race_map)))
total = np.zeros((len(age_map), len(race_map)))

for tup in val.itertuples() :
    file = tup[1]
    age = age_map[tup[2].strip()]
    race = race_map[tup[3]]

    total[age, race] = total[age, race] + 1
    ana = DeepFace.analyze(img_path = file, actions = ['age', 'race'], enforce_detection=False)
    o_age = ana[0]['age']

    pre_age = 0
    if o_age < 3 :
        pre_age = 0
    elif o_age < 10 :
        pre_age = 1
    elif o_age < 20 :
        pre_age = 2
    elif o_age < 30 :
        pre_age = 3
    elif o_age < 40 :
        pre_age = 4
    elif o_age < 50 :
        pre_age = 5
    elif o_age < 60 :
        pre_age = 6
    elif o_age < 70 :
        pre_age = 7
    else :
        pre_age = 8

    pre_race = race_map[ana[0]['dominant_race']]

    if pre_race == race and pre_age == age :
        correct[age, race] = correct[age, race] + 1

# now Y^ and Y are about the age
epsilon = np.zeros(len(age_map))
for i in range(len(age_map)) :
    min_P = 1.0
    max_P = 0
    for j in range(len(race_map)) :
        if correct[i, j] == 0:
            continue

        P = correct[i, j] / total[i, j]
        min_P = min(min_P, P)
        max_P = max(max_P, P)

    if (max_P <= 0) :
        epsilon[i] = -1
    else :
        epsilon[i] = math.log(max_P / min_P)

print(epsilon)
    
# print(wrong)
# print("Ground Truth:", age, race, "Predicted Result:" , pre_age, pre_race)