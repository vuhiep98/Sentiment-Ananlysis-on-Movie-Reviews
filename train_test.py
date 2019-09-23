import os
import pandas as pd
from tqdm import tqdm

def get_file_list(direct):
    return [direct + f for f in os.listdir(direct) if ".txt" in f]

def read_txt_file(file_name):
    # imdb_score = file_name.replace(".txt", "").split("_")[1]
    with open(file_name, "r", encoding="utf-8") as reader:
        content = reader.read()
    return content

def create_data(data_type="train"):
    direct = "C:/Users/user/Desktop/Hiep/Subjects/NLP/Sentiment Analysis/aclImdb/" + data_type + "/"
    pos_direct = direct + "pos/"
    neg_direct = direct + "neg/"

    print("Get pos data...")
    pos_contents = []
    pos_file_list = get_file_list(pos_direct)
    for f in tqdm(pos_file_list):
        pos_contents.append(read_txt_file(f))
    pos_label = [1] * len(pos_contents)
    
    print("Get neg data...")
    neg_contents = []
    neg_file_list = get_file_list(neg_direct)
    for f in tqdm(neg_file_list):
        neg_contents.append(read_txt_file(f))
    neg_label = [0] * len(neg_contents)

    print("Create data frame...")
    data = pd.DataFrame({"content": pos_contents + neg_contents, "label": pos_label + neg_label})
    print("To 'csv' file...")
    data.to_csv(data_type + ".csv", index=False, encoding="utf-8")
    print("Success")



if __name__ == "__main__":
    # direct = "C:/Users/user/Desktop/Hiep/Subjects/NLP/Sentiment Analysis/aclImdb/train/pos/"
    # file_list = get_file_list(direct)
    # print(file_list[:10])
    # file_name = "0_9.txt"
    # print(read_txt_file(file_name)[0])
    # train_data = create_train_data()
    # print(train_data[:10])
    create_data("test")