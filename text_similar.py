import json

from gensim import corpora, models
from gensim.matutils import corpus2csc

from preprocessing import transforming_tokens as transf

# We're using tf-idf model stands for term frequency-inverse document frequency


class TextSimilar(object):

    def __init__(self):
        #load data
        f_data = json.load(open("data.json"))
        self.ques = [d["ques"].lower() for d in f_data]
        self.ans = [d["ans"] for d in f_data]


    def predict(self, input):
        #convert text into vector
        print(transf(input))
        vec = self.dictionary.doc2bow(transf(input))
        #the vector is represented in tfidf model
        vec_tfidf = self.model[vec]
        if len(vec_tfidf) == 0:
            return None
        vec_tfidf = sorted(vec_tfidf, key=lambda item: item[1])
        print(vec_tfidf)
        return self.ques[int(vec_tfidf[-1][0])]



    def load_model(self):
        try:
            self.corpus = corpora.MmCorpus("tmp/corpus.mm")
            self.model = models.TfidfModel.load("tmp/model.tfidf")
            self.dictionary = corpora.Dictionary.load("tmp/dictionary.dict")
        except Exception:
            print("some files didn't exist")


    def create_dictionary(self):
        texts = [[token for token in transf(q)] for q in self.ques]
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

    # def answer(self):
