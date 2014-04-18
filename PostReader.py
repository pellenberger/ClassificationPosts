class PostReader:
    def __init__(self, base_directory, ignored_word_file):
        self.base_directory = base_directory
        self.ignored_word_file = ignored_word_file

    def read_files(self):
        pass

    def get_training_set(self):
        # General definition: true=>positive, false=>negative
        # return hashmap with <message_word_list, POSITIVE/NEGATIVE>
        pass

    def get_verification_set(self):
        # return the 20% of the messages in form hashmap with <message_word_list, POSITIVE/NEGATIVE>
        return []

    def get_word_set(self):
        # return all the present words in the learning set.
        return []


#class PostReaderTagged:

