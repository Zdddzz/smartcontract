# -*- coding: utf-8 -*- 
# @Time : 2020-12-17 19:15
# @Author : Di Zhu


import re
import os
import pickle
from tqdm import tqdm
import random
from data_processor import BasicBlockProcessor, TokenIDManager
import hashlib
import torch
import torch.nn as nn
from init_models import init_transformer


def list_hash(data):
    string = ""
    step = 1000
    index = 0
    while index < len(data):
        string += str(data[index])
        index += step
    return hashlib.md5(string.encode()).hexdigest()


def deepcopy(data):
    if not os.path.exists('.tmp'):
        os.mkdir('.tmp')
    data_hash = list_hash(data)
    file_path = os.path.join('.tmp', data_hash + '.pkl')
    if not os.path.exists(file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    with open(file_path, 'rb') as f:
        return pickle.load(f)


class Dataset(object):
    """manage the dataset"""

    #dataset_dir='asm'
    def __init__(self, dataset_dir='/asm_final', opt_source_dir='source/opt/', asm_source_dir='source/asm/',
                 opt_vocab_path='opt_vocab.txt', asm_vocab_path='asm_vocab.txt',
                 opt_train_path='basic_blocks/opt_train.txt', asm_train_path='basic_blocks/asm_train.txt',
                 data_path='basic_blocks/data.pkl'):
        """
        init Dataset class
        :param dataset_dir: the root directory of asm_final
        :param opt_source_dir: the directory which stores source files on opt
        :param asm_source_dir: the directory which stores source files on asm
        :param opt_vocab_path: the vocabulary file of opt
        :param asm_vocab_path: the vocabulary file of asm
        :param opt_train_path: the file path stores opt basic blocks
        :param asm_train_path: the file path stores asm basic blocks
        :param data_path: the output pkl file
        """
        self.dataset_dir = os.path.abspath(os.path.join("E:/", dataset_dir))
        self.opt_source_dir = os.path.join(self.dataset_dir, opt_source_dir)
        self.asm_source_dir = os.path.join(self.dataset_dir, asm_source_dir)
        self.opt_vocab_path = os.path.join(self.dataset_dir, opt_vocab_path)
        self.asm_vocab_path = os.path.join(self.dataset_dir, asm_vocab_path)
        self.opt_train_path = os.path.join(self.dataset_dir, opt_train_path)
        self.asm_train_path = os.path.join(self.dataset_dir, asm_train_path)
        self.data_path = os.path.join(self.dataset_dir, data_path)

    def load(self):
        """load the dataset"""
        if not self.__been_preprocessed():
            self.preprocess()

        asm_basic_block_processor = BasicBlockProcessor('asm', self.asm_vocab_path)
        opt_basic_block_processor = BasicBlockProcessor('opt', self.opt_vocab_path)

        with open(self.data_path, 'rb') as f:
            raw_data = pickle.load(f)
            random.shuffle(raw_data)

        data = []
        for sample in tqdm(raw_data, desc="token2id"):
            if sample["opt"].__class__ is str:
                opt = sample["opt"]
                asm = sample["asm"]
            else:
                opt = sample["opt"].decode()
                asm = sample["asm"].decode()
            data.append({"opt": opt_basic_block_processor.to_ids(opt),
                         "asm": asm_basic_block_processor.to_ids(asm)})

        opt_token_id_manager = TokenIDManager(self.opt_vocab_path)
        asm_token_id_manager = TokenIDManager(self.asm_vocab_path)
        '''
        opt1 = opt.split(" ")
        print(opt_token_id_manager)
        print(type(opt))
        print(opt[3:10])
        print("************\n")
        print(data[3:10])
        '''
        return data, len(opt_token_id_manager.vocab_list) + 1, len(asm_token_id_manager.vocab_list) + 1  # [UNK]

    def preprocess(self):
        """preprocess the assembly files, generate the .pkl file and vocabulary files"""
        if os.path.exists(self.data_path):
            return

        if not os.path.exists(self.opt_train_path) and not os.path.exists(self.asm_train_path):
            opt_bbs = []
            asm_bbs = []
            asm_basic_block_processor = BasicBlockProcessor('asm', self.asm_vocab_path)
            opt_basic_block_processor = BasicBlockProcessor('opt', self.opt_vocab_path)

            asm_failed_sum = 0
            opt_failed_sum = 0
            for file_name in tqdm(os.listdir(self.opt_source_dir), "generating data set"):
                if file_name[0] == '.':
                    continue
                asm_source_file_dir = os.path.join(self.asm_source_dir, file_name)
                opt_source_file_dir = os.path.join(self.opt_source_dir, file_name)
                if os.path.exists(opt_source_file_dir) and os.path.exists(asm_source_file_dir):
                    dirs = os.listdir(opt_source_file_dir)
                    for file in dirs:
                        asm_source_file = os.path.join(asm_source_file_dir, file)
                        opt_source_file = os.path.join(opt_source_file_dir, file)
                        asm_blocks, asm_failed_count = self.__parser(asm_source_file, "asm")
                        opt_blocks, opt_failed_count = self.__parser(opt_source_file, "opt")

                        asm_failed_sum += asm_failed_count
                        opt_failed_sum += opt_failed_count

                        # remove the basic blocks which parse failed
                        if not len(asm_blocks) == len(opt_blocks):
                            keys = (set(asm_blocks.keys()) | set(opt_blocks.keys())) - (
                                    set(asm_blocks.keys()) & set(opt_blocks.keys()))
                            for key in keys:
                                if key in asm_blocks:
                                    asm_blocks.pop(key)
                                if key in opt_blocks:
                                    opt_blocks.pop(key)

                        for bb in asm_blocks:
                            normalized_bb = asm_basic_block_processor.normalize(asm_blocks[bb])
                            asm_bbs.append(normalized_bb)
                        for bb in opt_blocks:
                            normalized_bb = opt_basic_block_processor.normalize(opt_blocks[bb])
                            opt_bbs.append(normalized_bb)

            print("asm failed sum: ", asm_failed_sum)
            print("opt failed sum: ", opt_failed_sum)

            with open(self.asm_train_path, "w") as f_asm:
                f_asm.write("\n".join(asm_bbs))

            with open(self.opt_train_path, "w") as f_opt:
                f_opt.write("\n".join(opt_bbs))

            self.__vocab_generator(opt_bbs, asm_bbs)

        print("generating pkl file...")
        data = []
        with open(self.data_path, 'wb') as f_data, \
                open(self.opt_train_path, "rb") as f_opt, \
                open(self.asm_train_path, "rb") as f_asm:
            for bb_opt, bb_asm in zip(f_opt, f_asm):
                data.append(
                    {
                        "opt": bb_opt,
                        "asm": bb_asm
                    }
                )
            pickle.dump(data, f_data)
        print("done.")

    @staticmethod
    def __parser(filename, arch):
        """
        parse basic block ID
        :param filename: assembly file name
        :param arch: architecture，opt or asm
        :return: a map of basic blocks and their ids, failure count
        """

        blocks = {}
        blocks_id = {}

        label = ""
        for line in open(filename, 'r', encoding='UTF-8'):
            if line == "\n": continue
            if line.strip()[:2] == "/*":continue
            if line.strip()[:4] == "sub_":continue
            if line.strip()[:4] == "link":continue
            if line.strip()[:4] == "data":continue
            if re.match(r'^\.\.\.', line): continue
            if line.strip()[:3] == "aux": continue
            if line.strip() == "}": continue
            if re.match(r'    tag_[0-9_]+:', line):
                label = line.strip().split(":")[0]
                while blocks.__contains__(label):
                    label += "1"
                blocks[label] = []
            elif not label == "":
                instruction = line.strip()
                if arch is 'opt':
                    blocks[label].append(instruction)
                elif arch is 'asm':
                    blocks[label].append(instruction)

        failed_count = 0
        for i in blocks:
            ret = str(i)
            if len(ret) > 0:
                if arch == "opt":
                    try:
                        bid = str(ret).split(":")[0].split("_")[1]
                        blocks_id[bid] = blocks[i]
                    except KeyError:
                        pass
                elif arch == "asm":
                    try:
                        bid = str(ret).split(":")[0].split("_")[1]
                        blocks_id[bid] = blocks[i]
                    except KeyError:
                        pass
        for bid in list(blocks_id):
            bb = list(blocks_id[bid])
            if len(bb) == 0:
                del (blocks_id[bid])

            bb_str = ";".join(bb)
            if arch == "opt":
                bb_str = re.sub(r'tag_[0-9]+:', "", bb_str)
                bb = bb_str.split(";")
            elif arch == "asm":
                bb_str = re.sub(r'tag_[0-9]+:', "", bb_str)
                bb = bb_str.split(";")

            if len(bb) == 0 :
                blocks_id.pop(bid)
            else:
                blocks_id[bid] = bb
        return blocks_id, failed_count

    def __vocab_generator(self, opt_bbs, asm_bbs):
        """
        generate the vocabularies
        :param opt_bbs: opt basic blocks
        :param asm_bbs: asm basic blocks
        """
        print("generating vocabulary...")

        reserved_vocab = ["padding", "<s>", "</s>"]

        opt_tokens = self.__get_uniq_tokens(opt_bbs)
        asm_tokens = self.__get_uniq_tokens(asm_bbs)

        opt_vocab = reserved_vocab + opt_tokens
        asm_vocab = reserved_vocab + asm_tokens

        assert opt_vocab[0] is "padding", "「padding」 index should be 0 in opt vocab"
        assert asm_vocab[0] is "padding", "「padding」 index should be 0 in asm vocab"

        with open(self.asm_vocab_path, "w") as f_asm:
            f_asm.write("\n".join(asm_vocab))
        with open(self.opt_vocab_path, "w") as f_opt:
            f_opt.write("\n".join(opt_vocab))

    @staticmethod
    def __get_uniq_tokens(bbs):
        """get unique tokens from given basic blocks"""
        tokens = []
        insts = []
        for bb in bbs:
            insts += bb.split("\n")
        for inst in insts:
            tokens += inst.split()
        tokens = list(set(tokens))
        print(tokens)
        return tokens

    def __been_preprocessed(self):
        """judge where the data has been preprocessed yet"""
        if os.path.exists(self.data_path) and os.path.exists(self.opt_vocab_path) and os.path.exists(
                self.asm_vocab_path):
            return True
        else:
            return False


class DataLoader(object):
    def __init__(self, data, batch_size, max_len=256, token_pad_idx=0,
                 require_negative_samples=True, seed=2020):
        """
        init DataLoader
        :param data: all data in MISA
        :param batch_size: batch size for model training
        :param max_len: max len of basic block
        :param token_pad_idx: padding token index
        :param require_negative_samples: if need negative samples
        :param seed: random seed
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.require_negative_samples = require_negative_samples

        if max_len is not None:
            self.max_len = max_len
        else:
            self.max_len = self.get_max_len(data)

        random.seed(seed)
        random.shuffle(data)

        data = self.padding(data, token_pad_idx)

        if self.require_negative_samples:
            #data = self.insert_negative_samples(data, method='mix')
            data = self.insert_negative_samples(data, method='differentiation')

        x = int(0.8 * len(data))
        self.train = data[:x]
        self.val = data[x:]

        print("Train data")
        print(len(self.train))
        print("Val data")
        print(len(self.val))

        random.shuffle(self.train)
        random.shuffle(self.val)

    def get_train_and_val_size(self):
        return len(self.train), len(self.val)

    def insert_negative_samples(self, raw_data, n=3, method='differentiation'):#If you use random negative sampling generation you need to modify method to random and re-run data_manager
        data_hash = list_hash(raw_data)
        data_path = os.path.join('.tmp', data_hash + '_' + method + "_" + str(n) + ".pkl")
        if os.path.exists(data_path):
            with open(data_path, 'rb') as f:
                return pickle.load(f)

        random.seed(2020)
        if method == 'random':
            data_copy = deepcopy(raw_data)
            for sample in tqdm(raw_data, "generating random negative samples"):
                random_samples = random.sample(data_copy, n + 1)
                try:
                    random_samples.remove(sample)
                except BaseException:
                    random_samples = random_samples[:n]

                if sample.keys().__contains__("negative"):
                    break

                for index in range(len(random_samples)):
                    if index < n / 2:
                        if index == 0:
                            sample['negative'] = random_samples[index]["asm"]
                            sample['negative_encoder'] = 1
                        else:
                            raw_data.append({"opt": sample["opt"], "asm": sample["asm"],
                                             "negative": random_samples[index]["asm"], "negative_encoder": 1})
                    else:
                        raw_data.append({"opt": sample["opt"], "asm": sample["asm"],
                                         "negative": random_samples[index]["opt"], "negative_encoder": 0})

        elif method =='differentiation':
            print("differentiation  @@@@@")
            with_opt_encoding_file = os.path.join('.tmp', data_hash + "_opt_encoding.pkl")

            if os.path.exists(with_opt_encoding_file):
                with open(with_opt_encoding_file, 'rb') as f:
                    data_copy = pickle.load(f)
            else:
                pretrained_transformer, _ = init_transformer()
                encoder = pretrained_transformer.encoder.to(self.device)
                encoder.no_grads()
                n_gpu = torch.cuda.device_count()
                if n_gpu > 1:
                    encoder = torch.nn.DataParallel(encoder)
                pos = torch.LongTensor([list(range(1, 1 + len(raw_data[0]['opt'])))]).to(self.device)
                data_copy = deepcopy(raw_data)
                for sample in tqdm(data_copy, "opt encoding"):
                    opt = torch.LongTensor([sample["opt"]]).to(self.device)
                    masks = opt.gt(0).long()
                    opt_pos = torch.mul(pos, masks)
                    sample["opt_encoding"] = encoder(opt, opt_pos)[0].sum(1).to('cpu').numpy()
                with open(with_opt_encoding_file, 'wb') as f:
                    pickle.dump(data_copy, f)

            raw_data = deepcopy(data_copy)
            token = 0
            for sample in tqdm(raw_data, "generating differentiation negative samples"):
                if sample.keys().__contains__("negative"):
                    break
                sample_opt_encoding = torch.Tensor(sample["opt_encoding"])

                if token > len(data_copy) - 100:
                    random.shuffle(data_copy)
                    token = 0
                random_samples = data_copy[token:100 + token]
                token += 100

                encodings = []
                for r_sample in random_samples:
                    encodings.append(r_sample["opt_encoding"])
                encodings = torch.Tensor(encodings).to(self.device)
                sample_opt_encodings = sample_opt_encoding.repeat(100, 1).to(self.device)

                distances = nn.PairwiseDistance().to(self.device)(encodings, sample_opt_encodings).to(
                    'cpu').numpy().tolist()

                try:
                    self_index = distances.index(0)
                    distances[self_index] = 10000
                except BaseException:
                    pass

                indices = sorted(range(len(distances)), key=lambda i: distances[i])[int(n * 2 / 3):]
                random.shuffle(indices)
                for i in range(len(indices)):
                    index = indices[i]
                    if i < len(indices) / 2:
                        if i == 0:
                            sample['negative'] = random_samples[index]["asm"]
                            sample['negative_encoder'] = 1
                        else:
                            raw_data.append({"opt": sample["opt"], "asm": sample["asm"],
                                             "negative": random_samples[index]["asm"], "negative_encoder": 1})
                    else:
                        raw_data.append({"opt": sample["opt"], "asm": sample["asm"],
                                         "negative": random_samples[index]["opt"], "negative_encoder": 0})


        elif method == 'mix':
            #random_negatives = self.insert_negative_samples(deepcopy(raw_data), int(n * 2 / 3), 'random')
            diff_negatives = self.insert_negative_samples(deepcopy(raw_data), int(n * 2 / 3), 'differentiation')
            with_opt_encoding_file = os.path.join('.tmp', data_hash + "_opt_encoding.pkl")

            if os.path.exists(with_opt_encoding_file):
                with open(with_opt_encoding_file, 'rb') as f:
                    data_copy = pickle.load(f)
            else:
                pretrained_transformer, _ = init_transformer()
                encoder = pretrained_transformer.encoder.to(self.device)
                encoder.no_grads()
                n_gpu = torch.cuda.device_count()
                if n_gpu > 1:
                    encoder = torch.nn.DataParallel(encoder)
                pos = torch.LongTensor([list(range(1, 1 + len(raw_data[0]['opt'])))]).to(self.device)
                data_copy = deepcopy(raw_data)
                for sample in tqdm(data_copy, "opt encoding"):
                    opt = torch.LongTensor([sample["opt"]]).to(self.device)
                    masks = opt.gt(0).long()
                    opt_pos = torch.mul(pos, masks)
                    sample["opt_encoding"] = encoder(opt, opt_pos)[0].sum(1).to('cpu').numpy()
                with open(with_opt_encoding_file, 'wb') as f:
                    pickle.dump(data_copy, f)

            raw_data = deepcopy(data_copy)
            token = 0
            for sample in tqdm(raw_data, "generating hard negative samples"):
                if sample.keys().__contains__("negative"):
                    break
                sample_opt_encoding = torch.Tensor(sample["opt_encoding"])

                if token > len(data_copy) - 100:
                    random.shuffle(data_copy)
                    token = 0
                random_samples = data_copy[token:100 + token]
                token += 100

                encodings = []
                for r_sample in random_samples:
                    encodings.append(r_sample["opt_encoding"])
                encodings = torch.Tensor(encodings).to(self.device)
                sample_opt_encodings = sample_opt_encoding.repeat(100, 1).to(self.device)

                distances = nn.PairwiseDistance().to(self.device)(encodings, sample_opt_encodings).to(
                    'cpu').numpy().tolist()

                try:
                    self_index = distances.index(0)
                    distances[self_index] = 10000
                except BaseException:
                    pass

                indices = sorted(range(len(distances)), key=lambda i: distances[i])[:int(n / 3)]
                random.shuffle(indices)
                for i in range(len(indices)):
                    index = indices[i]
                    if i < len(indices) / 2:
                        if i == 0:
                            sample['negative'] = random_samples[index]["asm"]
                            sample['negative_encoder'] = 1
                        else:
                            raw_data.append({"opt": sample["opt"], "asm": sample["asm"],
                                             "negative": random_samples[index]["asm"], "negative_encoder": 1})
                    else:
                        raw_data.append({"opt": sample["opt"], "asm": sample["asm"],
                                         "negative": random_samples[index]["opt"], "negative_encoder": 0})

            for sample in raw_data:
                if sample.keys().__contains__("opt_encoding"):
                    sample.pop("opt_encoding")
            #raw_data = raw_data + random_negatives
            raw_data = raw_data + diff_negatives

        with open(data_path, 'wb') as f:
            pickle.dump(raw_data, f)
        return raw_data

    def padding(self, data, token_pad_idx):
        index = 0
        while index < len(data):
            opt = data[index]['opt']
            asm = data[index]['asm']
            if len(opt) > self.max_len or len(asm) > self.max_len:
                data.pop(index)
            else:
                data[index]['opt'] = opt + [token_pad_idx] * (self.max_len - len(opt))
                data[index]['asm'] = asm + [token_pad_idx] * (self.max_len - len(asm))
                index += 1
        return data

    @staticmethod
    def get_max_len(data):
        max_len = 0
        for sample in data:
            max_len = max(len(sample["opt"]), len(sample["asm"]), max_len)
        return max_len

    def data_iterator(self, data_split, shuffle=True):
        if data_split == "train":
            data_list = self.train
        elif data_split == "val":
            data_list = self.val
        else:
            raise Exception

        if shuffle:
            random.shuffle(data_list)

        if not self.require_negative_samples:

            for i in range(len(data_list) // self.batch_size):
                inputs = [data["opt"] for data in data_list[i * self.batch_size:(i + 1) * self.batch_size]]
                outputs = [data["asm"] for data in data_list[i * self.batch_size:(i + 1) * self.batch_size]]

                batch_inputs = torch.LongTensor(inputs)
                batch_outputs = torch.LongTensor(outputs)

                pos = torch.LongTensor(list(range(1, 1 + self.max_len))).expand([self.batch_size, self.max_len])
                input_masks = batch_inputs.gt(0).long()
                output_masks = batch_outputs.gt(0).long()

                input_pos = torch.mul(pos, input_masks)
                output_pos = torch.mul(pos, output_masks)

                batch_inputs, input_pos, batch_outputs, output_pos = (
                    batch_inputs.to(self.device),
                    input_pos.to(self.device),
                    batch_outputs.to(self.device),
                    output_pos.to(self.device)
                )
                assert len(batch_outputs) == len(batch_inputs) == len(input_pos) == len(output_pos)
                yield batch_inputs, input_pos, batch_outputs, output_pos
        else:

            for i in range(len(data_list) // self.batch_size):
                opts = [data["opt"] for data in data_list[i * self.batch_size:(i + 1) * self.batch_size]]
                asms = [data["asm"] for data in data_list[i * self.batch_size:(i + 1) * self.batch_size]]
                negatives = [data["negative"] for data in data_list[i * self.batch_size:(i + 1) * self.batch_size]]
                negative_encoders = [data["negative_encoder"] for data in
                                     data_list[i * self.batch_size:(i + 1) * self.batch_size]]

                batch_opts = torch.LongTensor(opts)
                batch_asms = torch.LongTensor(asms)
                batch_negatives = torch.LongTensor(negatives)
                batch_negative_encoders = torch.LongTensor(negative_encoders)

                pos = torch.LongTensor(list(range(1, 1 + self.max_len))).expand([self.batch_size, self.max_len])
                opt_masks = batch_opts.gt(0).long()
                asm_masks = batch_asms.gt(0).long()
                negative_masks = batch_negatives.gt(0).long()

                opt_pos = torch.mul(pos, opt_masks)
                asm_pos = torch.mul(pos, asm_masks)
                negative_pos = torch.mul(pos, negative_masks)

                batch_opts, opt_pos, batch_asms, asm_pos, batch_negatives, negative_pos, negative_encoders = (
                    batch_opts.to(self.device),
                    opt_pos.to(self.device),
                    batch_asms.to(self.device),
                    asm_pos.to(self.device),
                    batch_negatives.to(self.device),
                    negative_pos.to(self.device),
                    batch_negative_encoders.to(self.device)
                )
                assert len(batch_opts) == len(opt_pos) == len(batch_asms) == len(asm_pos) == len(
                    batch_negatives) == len(negative_pos) == len(negative_encoders)
                yield batch_opts, opt_pos, batch_asms, asm_pos, batch_negatives, negative_pos, negative_encoders


if __name__ == '__main__':
    print(Dataset().dataset_dir)
    print(Dataset().data_path)
    Dataset().preprocess()