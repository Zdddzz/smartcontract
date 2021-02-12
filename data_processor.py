# -*- coding: utf-8 -*- 
# @Time : 2020-12-14 16:04 
# @Author : Di Zhu
import re
import os


class TokenIDManager(object):
    def __init__(self, vocab_path):
        self.vocab_list = self.load_vocab_list(vocab_path)
        self.token2id_map = self.__generate_token2id_vocab_map()
        self.id2token_map = self.__generate_id2token_vocab_map()

    @staticmethod
    def load_vocab_list(vocab_path):
        """load vocabulary"""
        print(vocab_path)
        assert os.path.exists(vocab_path)
        with open(vocab_path, 'r') as f:
            vocab_list = f.readlines()
            print(vocab_list)
        return vocab_list

    def token2id(self, token):
        """convert token to id"""
        if self.token2id_map.__contains__(token):
            vocab_id = self.token2id_map[token]
        else:
            vocab_id = len(self.token2id_map)
        return vocab_id

    def __generate_token2id_vocab_map(self):
        """generate the map from token to id"""
        vocab_map = {}
        for i in range(len(self.vocab_list)):
            vocab_map[self.vocab_list[i].strip()] = i
        return vocab_map

    def __generate_id2token_vocab_map(self):
        """generate the map from id to token"""
        vocab_map = {}
        for i in range(len(self.vocab_list)):
            vocab_map[i] = self.vocab_list[i].strip()
        return vocab_map


class InstructionProcessor(object):
    def __init__(self, arch):
        """
        InstructionProcessor
        :param arch:opt or asm
        """
        if arch.lower() == 'opt':
            self.normalizer = self.optPass
        elif arch.lower() == 'asm':
            self.normalizer = self.asmPass
        else:
            raise Exception

    def normalize(self, inst):
        """
        normalize assembly instruction
        :param inst: assembly instruction
        :return: normalized assembly instruction
        """
        inst = self.normalizer.remove_space(inst)
        inst = self.normalizer.push(inst)
        inst = self.normalizer.bb_label2(inst)
        inst = self.normalizer.bb_label(inst)
        inst = self.normalizer.bb_label3(inst)
        inst = self.normalizer.swap(inst)
        inst = self.normalizer.dup(inst)
        inst = self.normalizer.stop(inst)
        return inst

    class optPass(object):
        @classmethod
        def remove_space(cls, inst):
            return str(inst).replace(" ", "").replace("\t", "")


        @classmethod
        def push(cls, inst):
            return re.sub(r"0x[0-9a-fA-F]+", "var", inst)

        @classmethod
        def swap(cls, inst):
            return re.sub(r"swap[0-9]+", "swap", inst)

        @classmethod
        def dup(cls, inst):
            return re.sub(r"dup[0-9]+", "dup", inst)


        @classmethod
        def bb_label(cls, inst):
            return re.sub(r"tag_[0-9]+", "BB", inst)

        @classmethod
        def bb_label2(cls, inst):
            return re.sub(r'tag_0_[0-9]+', "BB", inst)

        @classmethod
        def bb_label3(cls, inst):
            return  re.sub(r'sub_[0-9]+', "BB", inst)



        @classmethod
        def stop(cls, inst):
            if inst == 'revert' or inst == 'selfdestruct' or inst == 'return':
                inst = f'{inst} stop'
            return inst


    class asmPass(object):
        @classmethod
        def remove_space(cls, inst):
            return str(inst).replace(" ", "")


        @classmethod
        def push(cls, inst):
            return re.sub(r"0x[0-9a-fA-F]+", "var", inst)

        @classmethod
        def swap(cls, inst):
            return re.sub(r"swap[0-9]+", "swap", inst)

        @classmethod
        def dup(cls, inst):
            return re.sub(r"dup[0-9]+", "dup", inst)

        @classmethod
        def bb_label(cls, inst):
            return re.sub(r"tag_[0-9]+", "BB", inst)

        @classmethod
        def bb_label2(cls, inst):
            return re.sub(r'tag_0_[0-9]+', "BB", inst)

        @classmethod
        def bb_label3(cls, inst):
            return re.sub(r'sub_[0-9]+', "BB", inst)

        @classmethod
        def stop(cls, inst):
            if inst == 'revert' or inst == 'selfdestruct' or inst == 'return':
                inst = f'{inst} stop'
            return inst



class BasicBlockProcessor(object):
    def __init__(self, arch, vocab_path):
        self.instruction_processor = InstructionProcessor(arch)
        self.token_id_manager = TokenIDManager(vocab_path)

    def normalize(self, basic_block):
        for index in range(0, len(basic_block)):
            inst = basic_block[index]
            inst = self.instruction_processor.normalize(inst)
            basic_block[index] = inst
        basic_block = " \n ".join(basic_block).replace("\t", " ")
        return " ".join(basic_block.split())

    def to_ids(self, basic_block):
        token_list = ["<s>"] + basic_block.strip().split() + ["</s>"]
        ids = []
        for token in token_list:
            ids.append(self.token_id_manager.token2id(token))
        return ids

