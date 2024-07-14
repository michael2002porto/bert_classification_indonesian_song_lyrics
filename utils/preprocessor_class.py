import enum
import pickle
import torch
import os
import sys

import re

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import lightning as L
import pandas as pd

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from transformers import BertTokenizer

# untuk membuat progress bar
from tqdm import tqdm 

class PreprocessorClass(L.LightningDataModule):
    # 1. def __init__()
    # 2. def setup()

    def __init__(self,
                 preprocessed_dir,
                 batch_size = 10,
                 max_length = 100,):

        super(PreprocessorClass, self).__init__()

        self.label2id = {
            'semua usia': 0,
            'anak': 1,
            'remaja': 2,
            'dewasa': 3
        }

        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()

        # Merubah kalimat menjadi id, tokenize dan attention
        self.tokenizer = BertTokenizer.from_pretrained('indolem/indobert-base-uncased')

        self.max_length = max_length
        self.preprocessed_dir = preprocessed_dir

        self.batch_size = batch_size


    def clean_str(self, string):
        string = string.lower()
        string = re.sub(r"[^A-Za-z0-9(),!?\'\-`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        # Menghapus enter
        string = re.sub(r"\n", "", string)

        # Membersihkan elemen yang tidak perlu, seperti menghapus spasi 2
        string = re.sub(r"\'re", " \'re", string)

        # Mengecek digit atau bukan
        string = re.sub(r"\'d", " \'d", string)

        # Mengecek long atau bukan
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        string = string.strip()
        # Menghilangkan imbuhan
        return self.stemmer.stem(string)

    def load_data(self, path = "data/dataset_lyrics.xlsx"):
        dataset = pd.read_excel(path)
        dataset = dataset[["Title", "Lyric", "Age Class tag"]]
        
        # Mengetahui apa saja label yang ada di dalam dataset
        label_yang_ada = dataset["Age Class tag"].drop_duplicates()
        # print(label_yang_ada)
        
        # Konversi dari label text (semua usia) ke label id (0) dst
        dataset["Age Class tag"] = dataset["Age Class tag"].map(self.label2id)
        
        # Mengetahui apa saja label setelah dikonversi
        label_yang_ada = dataset["Age Class tag"].value_counts()
        # print(label_yang_ada)

        # sys.exit()
        
        return dataset

    def arrange_data(self, data):
        # Yang di lakukan
        # 1. Cleaning sentence
        # 2. Tokenizing data
        # 3. Arrange ke dataset (training, validation, testing)

        # type untuk tipe datanya, apakah training atau testing

        # y = label
        x_input_ids, x_token_type_ids, x_attention_mask, y = [], [], [], []
        for row, data in tqdm(data.iterrows(), total = data.shape[0], desc = "Preprocesing Song Lyrics"):
            '''
                'semua usia' = 0
                'anak' = 1
                'remaja' = 2
                'dewasa' = 3
            '''
            
            title = self.clean_str(data["Title"])
            lyric = self.clean_str(data["Lyric"])
            label = data["Age Class tag"]
            
            title = title.replace("lirik lagu ", "")

            # Mengubah label yang tadinya angka menjadi binary
            binary_lbl = [0] * len(self.label2id)
            binary_lbl[label] = 1
            
            # membuat tokenisasi
            tkn = self.tokenizer(
                f"{title} {lyric}", #batch_sentences
                max_length = 200,
                truncation = True,
                padding = "max_length",
            )
            x_input_ids.append(tkn['input_ids'])
            x_token_type_ids.append(tkn['token_type_ids'])
            x_attention_mask.append(tkn['attention_mask'])
            y.append(binary_lbl)

            #saya suka makan ... ...
            #input_ids = 3,1,5,0,0
            #attention_mask = 1,1,1,0,0
            
            # anak2 = 0
            # remaja = 1
            # dewasa = 2
            # semua umur = 3

            # y dewasa
            # y.append([0,0,1,0])

            # if row == 4:
            #     print("\n")
            #     print("row = ", row)
            #     print("label = ", label)
            #     print("x = ", f"{title} {lyric}")
            #     print("y = ", binary_lbl)
            #     sys.exit()
            #     break

        # Mengubah list ke tensor
        x_input_ids = torch.tensor(x_input_ids)
        x_token_type_ids = torch.tensor(x_token_type_ids)
        x_attention_mask = torch.tensor(x_attention_mask)
        y = torch.tensor(y)

        tensor_dataset = TensorDataset(
            x_input_ids,
            x_token_type_ids,
            x_attention_mask,
            y
        )

        # Ratio Training 80% overall data = (90% Train, 10% Validation)
        # Ratio Testing 20% overall data

        train_val_len = int(x_input_ids.shape[0] * 0.8)   #100 * 0.8 = 80
        train_len = int(train_val_len * 0.9)    #80 * 0.9 = 72
        val_len = train_val_len - train_len #80 - 72 = 8
        test_len = x_input_ids.shape[0] - train_val_len   #100 - 80 = 20

        train_set, val_set, test_set = torch.utils.data.random_split(
            tensor_dataset,
            [train_len, val_len, test_len]
        )

        # print("\n")
        # print("train_set = ", train_set)
        # print("val_set = ", val_set)
        # print("test_set = ", test_set)
        # sys.exit()
        
        if not os.path.exists(self.preprocessed_dir):
            os.makedirs(self.preprocessed_dir)
        
        # f untuk merubah ke string
        torch.save(train_set, f"{self.preprocessed_dir}/train.pt")
        torch.save(val_set, f"{self.preprocessed_dir}/valid.pt")
        torch.save(test_set, f"{self.preprocessed_dir}/test.pt")
        
        return train_set, val_set, test_set

    def arrange_train_data(self, data):
        # Yang di lakukan
        # 1. Cleaning sentence
        # 2. Tokenizing data
        # 3. Arrange ke dataset (training, validation, testing)

        # type untuk tipe datanya, apakah training atau testing

        # y = label
        x_input_ids, x_token_type_ids, x_attention_mask, y = [], [], [], []
        for row, data in tqdm(data.iterrows(), total = data.shape[0], desc = "Preprocesing Song Lyrics"):
            '''
                'semua usia' = 0
                'anak' = 1
                'remaja' = 2
                'dewasa' = 3
            '''
            
            title = self.clean_str(data["Title"])
            lyric = self.clean_str(data["Lyric"])
            label = data["Age Class tag"]
            
            title = title.replace("lirik lagu ", "")

            # Mengubah label yang tadinya angka menjadi binary
            binary_lbl = [0] * len(self.label2id)
            binary_lbl[label] = 1
            
            # membuat tokenisasi
            tkn = self.tokenizer(
                f"{title} {lyric}", #batch_sentences
                max_length = 200,
                truncation = True,
                padding = "max_length",
            )
            x_input_ids.append(tkn['input_ids'])
            x_token_type_ids.append(tkn['token_type_ids'])
            x_attention_mask.append(tkn['attention_mask'])
            y.append(binary_lbl)

            #saya suka makan ... ...
            #input_ids = 3,1,5,0,0
            #attention_mask = 1,1,1,0,0
            
            # anak2 = 0
            # remaja = 1
            # dewasa = 2
            # semua umur = 3

            # y dewasa
            # y.append([0,0,1,0])

            # if row == 4:
            #     print("\n")
            #     print("row = ", row)
            #     print("label = ", label)
            #     print("x = ", f"{title} {lyric}")
            #     print("y = ", binary_lbl)
            #     sys.exit()
            #     break

        # Mengubah list ke tensor
        x_input_ids = torch.tensor(x_input_ids)
        x_token_type_ids = torch.tensor(x_token_type_ids)
        x_attention_mask = torch.tensor(x_attention_mask)
        y = torch.tensor(y)

        tensor_dataset = TensorDataset(
            x_input_ids,
            x_token_type_ids,
            x_attention_mask,
            y
        )

        # Ratio Training 80% overall data = (90% Train, 10% Validation)
        # Ratio Testing 20% overall data

        train_len = int(x_input_ids.shape[0] * 0.9)   #100 * 0.9 = 90
        val_len = x_input_ids.shape[0] - train_len   #100 - 90 = 10

        train_set, val_set = torch.utils.data.random_split(
            tensor_dataset,
            [train_len, val_len]
        )

        # print("\n")
        # print("train_set = ", train_set)
        # print("val_set = ", val_set)
        # print("test_set = ", test_set)
        # sys.exit()
        
        if not os.path.exists(self.preprocessed_dir):
            os.makedirs(self.preprocessed_dir)
        
        # f untuk merubah ke string
        torch.save(train_set, f"{self.preprocessed_dir}/train.pt")
        torch.save(val_set, f"{self.preprocessed_dir}/valid.pt")
        
        return train_set, val_set

    def arrange_test_data(self, data):
        # Yang di lakukan
        # 1. Cleaning sentence
        # 2. Tokenizing data
        # 3. Arrange ke dataset (training, validation, testing)

        # type untuk tipe datanya, apakah training atau testing

        # y = label
        x_input_ids, x_token_type_ids, x_attention_mask, y = [], [], [], []
        for row, data in tqdm(data.iterrows(), total = data.shape[0], desc = "Preprocesing Song Lyrics"):
            '''
                'semua usia' = 0
                'anak' = 1
                'remaja' = 2
                'dewasa' = 3
            '''
            
            title = self.clean_str(data["Title"])
            lyric = self.clean_str(data["Lyric"])
            label = data["Age Class tag"]
            
            title = title.replace("lirik lagu ", "")

            # Mengubah label yang tadinya angka menjadi binary
            binary_lbl = [0] * len(self.label2id)
            binary_lbl[label] = 1
            
            # membuat tokenisasi
            tkn = self.tokenizer(
                f"{title} {lyric}", #batch_sentences
                max_length = 200,
                truncation = True,
                padding = "max_length",
            )
            x_input_ids.append(tkn['input_ids'])
            x_token_type_ids.append(tkn['token_type_ids'])
            x_attention_mask.append(tkn['attention_mask'])
            y.append(binary_lbl)

            #saya suka makan ... ...
            #input_ids = 3,1,5,0,0
            #attention_mask = 1,1,1,0,0
            
            # anak2 = 0
            # remaja = 1
            # dewasa = 2
            # semua umur = 3

            # y dewasa
            # y.append([0,0,1,0])

            # if row == 4:
            #     print("\n")
            #     print("row = ", row)
            #     print("label = ", label)
            #     print("x = ", f"{title} {lyric}")
            #     print("y = ", binary_lbl)
            #     sys.exit()
            #     break

        # Mengubah list ke tensor
        x_input_ids = torch.tensor(x_input_ids)
        x_token_type_ids = torch.tensor(x_token_type_ids)
        x_attention_mask = torch.tensor(x_attention_mask)
        y = torch.tensor(y)

        tensor_dataset = TensorDataset(
            x_input_ids,
            x_token_type_ids,
            x_attention_mask,
            y
        )
        
        if not os.path.exists(self.preprocessed_dir):
            os.makedirs(self.preprocessed_dir)
        
        # f untuk merubah ke string
        torch.save(tensor_dataset, f"{self.preprocessed_dir}/test.pt")
        
        return tensor_dataset

    def preprocessor(self,):
        # Menentukan dataset yang akan digunakan
        dataset = self.load_data(path = "data/dataset_lyrics.xlsx")

        if self.preprocessed_dir == "data/preprocessed/synthesized":
            dataset = self.load_data(path = "data/synthesized_lyrics.xlsx")

        if self.preprocessed_dir == "data/preprocessed/generated":
            dataset = self.load_data(path = "data/generated_lyrics.xlsx")

        if self.preprocessed_dir == "data/preprocessed/full_combination":
            original = self.load_data(path = "data/dataset_lyrics.xlsx")
            synthesized = self.load_data(path = "data/synthesized_lyrics.xlsx")
            generated = self.load_data(path = "data/generated_lyrics.xlsx")
            dataset = pd.concat([original, synthesized, generated], ignore_index=True)

        if self.preprocessed_dir == "data/preprocessed/split_combination":
            synthesized = self.load_data(path = "data/synthesized_lyrics.xlsx")
            generated = self.load_data(path = "data/generated_lyrics.xlsx")
            train_dataset = pd.concat([synthesized, generated], ignore_index=True)
            test_dataset = self.load_data(path = "data/dataset_lyrics.xlsx")

        # Mengecek apakah data train, valid, dan test sudah di preprocessed
        if not os.path.exists(f"{self.preprocessed_dir}/train.pt") \
            or not os.path.exists(f"{self.preprocessed_dir}/valid.pt") \
            or not os.path.exists(f"{self.preprocessed_dir}/test.pt"):

            if self.preprocessed_dir == "data/preprocessed/split_combination":
                train_data, valid_data = self.arrange_train_data(data = train_dataset)
                test_data = self.arrange_test_data(data = test_dataset)
            else:
                train_data, valid_data, test_data = self.arrange_data(data = dataset)

            print("Successfully created training, validation, and testing datasets")
        else:
            print("Load Preprocessed train and validation data data")
            train_data = torch.load(f"{self.preprocessed_dir}/train.pt")
            valid_data = torch.load(f"{self.preprocessed_dir}/valid.pt")
            print("Load Preprocessed test data")
            test_data = torch.load(f"{self.preprocessed_dir}/test.pt")

        return train_data, valid_data, test_data

    def setup(self, stage = None):
        # 100% data
        # 80% = Training
        # 20% = Testing
        
        train_set, val_set, test_set = self.preprocessor()
        
        if stage == "fit":
            self.train_data = train_set
            self.val_data = val_set
        elif stage == "test":
            self.test_data = test_set

    def train_dataloader(self):
        return DataLoader(
            self.train_data, 
            batch_size = self.batch_size, 
            shuffle = True,
            num_workers = 4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = 4,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = 4,
        )

# if __name__ == '__main__':
#     Pre = PreprocessorClass(preprocessed_dir = "data/preprocessed")
#     Pre.setup(stage = "fit")
#     train_data = Pre.train_dataloader()
#     print(train_data)