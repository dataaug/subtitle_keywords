
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import TfidfVectorizer
import jieba


class IDF():
    def __init__(self) -> None:
        # 停用词
        with open('./data/hit_stopwords.txt', 'r', encoding='utf-8') as fr:
            stopwords = fr.readlines()
            stopwords = [x.strip() for x in stopwords]
            STOPWORDS = set([stopword for stopword in stopwords if stopword])

        # 影视字幕读取
        with open('./data/subtitle.txt', 'r', encoding='utf-8') as fr:
            lines = fr.readlines()
            lines = [line.strip() for line in lines]
            data = [list(jieba.cut(x)) for x in lines]
            corpus = [' '.join(x) for x in data]  # 将分好的词用空格分开

        # vec = TfidfVectorizer(ngram_range=(1, 2), stop_words=STOPWORDS) # 这里考虑了双词问题
        # 我们暂时用单个词
        self.vec = TfidfVectorizer(ngram_range=(1, 1), stop_words=STOPWORDS)
        self.vec.fit_transform(corpus)

        # If you want to get the idf value for a particular word, here "hello"
        # idf = tf.idf_

    def get_idf(self, word):
        try:
            idf_value = self.vec.idf_[self.vec.vocabulary_[word]]
            return idf_value
        except:
            return 0.0

    def extract(self, sentence):  # 输入句子根据其idf返回关键词
        len_sent = len(sentence)
        num_keyword = len_sent // 5  #  每5个词返回一个关键词
        word_list = list(jieba.cut(sentence))
        word_idf = [self.get_idf(word) for word in word_list]
        word_idf_pair = list(zip(word_list, word_idf))
        word_idf_pair.sort(key=lambda x: x[1], reverse=True)
        word_idf_pair = word_idf_pair[:num_keyword]

        return [x[0] for x in word_idf_pair]  # 仅返回需要的关键词


if __name__ == '__main__':
    tool = IDF()
    print(tool.extract('他们的扫描设备古老，但很管用'))
