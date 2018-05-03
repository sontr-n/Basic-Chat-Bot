import json

from gensim import corpora, models, similarities

from preprocessing import transforming_tokens as transf
from collections import defaultdict


# We're using tf-idf model stands for term frequency-inverse document frequency


class TextSimilar(object):

    def __init__(self):
        #load data
        f_data = json.load(open("data.json"))
        self.ques = [d["ques"].lower() for d in f_data]
        self.ans = [d["ans"] for d in f_data]


    def predict(self, input):
        #convert text into vector
        vec = self.dictionary.doc2bow(transf(input))

        #the vector is represented in tfidf model
        vec_tfidf = self.index[vec]

        sims = sorted(enumerate(vec_tfidf), key=lambda item: -item[1])
        return sims[0][0]

    def load_model(self):
        try:
            # self.corpus = corpora.MmCorpus("tmp/corpus.mm")
            # self.model = models.TfidfModel.load("tmp/model.tfidf")
            self.dictionary = corpora.Dictionary.load("tmp/dictionary.dict")
            self.index =  similarities.MatrixSimilarity.load("tmp/matrix_space.index")
        except Exception:
            print("index doesn't exist")


    def create_dictionary(self):
        texts = [[token for token in transf(q)] for q in self.ques]
        frequency = defaultdict(int)
        for text in texts:
            for token in text:
                frequency[token] += 1
        texts = [[token for token in text if frequency[token] > 1] for text in texts]
        dictionary = corpora.Dictionary(texts)
        dictionary.save("tmp/dictionary.dict")


    def create_model(self):
        dictionary = corpora.Dictionary.load("tmp/dictionary.dict")
        corpus = [dictionary.doc2bow(transf(q)) for q in self.ques]
        corpora.MmCorpus.serialize("tmp/corpus.mm", corpus)
        tfidf = models.TfidfModel(corpus, normalize=True)
        tfidf.save("tmp/model.tfidf")
        # model = models.LsiModel(corpus_tfidf, id2word=dictionary)
        # model.save("tmp/model.lsi")

    def pre_trained_model(self):
        self.create_dictionary()
        self.create_model()
        tfidf = models.TfidfModel.load("tmp/model.tfidf")
        corpus = corpora.MmCorpus("tmp/corpus.mm")
        index = similarities.MatrixSimilarity(tfidf[corpus])
        index.save("tmp/matrix_space.index")

    def answer(self, input):
        ques_num = self.predict(input)
        if ques_num == 0:
            return "Xin lỗi, mình chưa hiểu ý của bạn"
        return self.ans[ques_num]

