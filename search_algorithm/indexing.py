import glob
import os

file_path = sorted(glob.glob("/Users/ngocphu/Documents/Deep Fashion/search algorithm/output/*/*/*/*/*.jpg"))
print(len(file_path))
output = "/Users/ngocphu/Documents/Deep Fashion/search algorithm/output_sorted"
index = 0
for idx, img in enumerate(file_path):
    (path, name) = os.path.split(img) 
    # path = path[62:]
    new_name = str(index) + "_" + name
    index +=1
    # print(path)
    os.rename(img, os.path.join(path, new_name))