from os import listdir
from os.path import join
import string
from math import floor
from random import shuffle
from copy import deepcopy
import re

class PostReader:
    ''' General definition : 1.0 => positive, present ; 0.0 => negative, no present '''

    pos_directory = "pos"
    neg_directory = "neg"

    def __init__(self, base_directory, ignored_word_file):
        ''' Hypothesis : base_directory contains the following directories : pos and neg '''
        self.base_directory = base_directory
        self.ignored_word_file = ignored_word_file
        self.pos_files = listdir(join(self.base_directory, self.pos_directory))
        self.neg_files = listdir(join(self.base_directory, self.neg_directory))

        # read ignored words
        with open(self.ignored_word_file, 'r', -1, 'utf-8') as file: self.ignored_words = [str.strip(line) for line in file.readlines()]

        print("pr : creating word set ...")
        self.create_word_set()
        print("pr : creating messages set ...")
        self.create_messages_set()

        print("pr : creating training and verification set ...")
        shuffle(self.messages_set)
        index = floor(len(self.messages_set) * 0.8)
        self.training_set = self.messages_set[:index]
        self.verification_set = self.messages_set[index:]

    def create_word_set(self):
        ''' Create list containing all words in corpus
        '''
        self.word_set = set()
        self.word_set = self.word_set.union(self.read_files(self.pos_files, join(self.base_directory, self.pos_directory)))
        self.word_set = self.word_set.union(self.read_files(self.neg_files, join(self.base_directory, self.neg_directory)))

    def create_messages_set(self):
        ''' Create list of tuples containing each message and its classification
        '''
        self.messages_set = list()
        message_classification = 1.0
        self.fill_messages_set(self.pos_files, join(self.base_directory, self.pos_directory), message_classification)
        message_classification = 0.0
        self.fill_messages_set(self.neg_files, join(self.base_directory, self.neg_directory), message_classification)

    def fill_messages_set(self, files, directory, message_classification):
        for filename in files:
            words = self.read_files([filename], directory)

            message_words = dict.fromkeys(self.word_set, 0.0)

            for word in words:
                message_words[word] = 1.0

            self.messages_set.append((message_words, message_classification))

    def read_files(self, files, directory):
        ''' Returns a list containing all words present in given files
        '''
        words_list = set()
        for filename in files:
            with open(join(directory, filename), 'r', -1, 'utf-8') as file:
                for line in file.readlines():
                    line = self.filter_text(line)
                    words_list = words_list.union(set([word for word in line.split(' ') if len(word) > 0]))
        return words_list

    def filter_text(self, text):
        text = re.sub(r'[%s\n]' % re.escape(string.punctuation), '', text)
        text = re.sub(r'\d', '', text)
        for word in self.ignored_words : text = re.sub(r'\b%s\b' % word, '', text)
        text = str.lower(text)
        return text

    def get_word_set(self):
        return self.word_set

    def get_training_set(self):
        ''' Returns 80% of messages_set
        '''
        return self.training_set

    def get_verification_set(self):
        ''' Returns 20% of message_set
        '''
        return self.verification_set
