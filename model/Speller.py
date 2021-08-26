import editdistance
import re

class Speller():
    def __init__(self, char_list, corpus_file):
        corpus = open(corpus_file, encoding='utf8').read()
        self.chars = ''.join(char_list)
        self.non_letter = ''.join(c for c in char_list if not self.__is_cyrillic(c))
        self.dictionary = self.__get_dictionary(corpus)

    def __is_cyrillic(self, char):
        return bool(re.match('[а-яА-ЯёЁ]', char))

    def __get_dictionary(self, corpus):
        non_letter_re = '[' + self.non_letter.replace(' ', '') + ']'
        non_letter_re = non_letter_re.replace('[UNK]', '')
        corpus = re.sub('[!(),-.:;?#]', '', corpus).lower()
        dictionary = set(str.split(corpus))
        return dictionary

    def __get_closest_word(self, word, min_dist_coef):
        flag = word[0].isupper()
        res_word = word = word.lower()
        min_dist = min_dist_coef * len(word)
        for dict_word in self.dictionary:
            dist = editdistance.eval(word, dict_word)
            if dist == 0:
                return word.capitalize() if flag else word
            elif dist < min_dist:
                min_dist = dist
                res_word = dict_word
        return res_word.capitalize() if flag else res_word

    def __compute_label(self, label, min_dist_coef):
        start_i = -1
        res_label = ''
        for i in range(len(label)):
            if label[i] in self.non_letter:
                if start_i >= 0:
                    res_label += self.__get_closest_word(label[start_i:i], min_dist_coef)
                    start_i = -1
                res_label += label[i]
            elif start_i < 0:
                start_i = i
        if start_i >= 0:
            res_label += self.__get_closest_word(label[start_i:i+1], min_dist_coef)
        return res_label

    def compute_batch(self, labels, min_dist_coef=0.3):
        return [self.__compute_label(label, min_dist_coef) for label in labels]

    def compute_img(self, label, min_dist_coef=0.3):
        return self.__compute_label(label, min_dist_coef)
