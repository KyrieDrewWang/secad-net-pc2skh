import os

data_dir = "/data/wc/SECAD-Net/data/secad_8192/h5"

DATA_base = "/data/wc/SECAD-Net/data/secad_8192/"
text_file_path_train = os.path.join(DATA_base, "data_train.txt")  
text_file_path_val  =  os.path.join(DATA_base, "data_val.txt")
text_file_path_test =  os.path.join(DATA_base, "data_test.txt")

s_lst = os.listdir(data_dir)

s_path_lst = []

for s in s_lst:
    s_path = os.path.join(data_dir, s)
    s_path_lst.append(s_path)
    
num = len(s_path_lst)

_s_path_lst = s_path_lst[:int(num*0.8)]

textfile = open(text_file_path_train, "w")
for im in _s_path_lst:
    textfile.write(im+"\n")
textfile.close()

_s_path_lst = s_path_lst[int(num*0.8):int(num*0.9)]
textfile = open(text_file_path_val, "w")
for im in _s_path_lst:
    textfile.write(im + "\n")
textfile.close()

_s_path_lst = s_path_lst[int(num*0.9):]
textfile = open(text_file_path_test, "w")
for im in _s_path_lst:
    textfile.write(im + "\n")
textfile.close()