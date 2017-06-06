import sys
import os
import logging
import time
import multiprocessing
from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.corpora import TextCorpus, MmCorpus, Dictionary

# ======================= create logger ======================= 
logging.basicConfig(format='\n\n==%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logger.info("Running %s" % ' '.join(sys.argv))

# ======================= load data ======================= 

# create word->word_id mapping, 3h
logger.info('Starting loading data')
wiki = WikiCorpus("./data/enwiki-20170520-pages-articles.xml.bz2") # see: https://dumps.wikimedia.org/enwiki/
# wiki.dictionary.save("./model_pv/wiki_dict.dict")
# wiki = MmCorpus("./model_pv/wiki_corpus.mm")  # Revive a corpus
# wiki_dict = Dictionary.load("./model_pv/wiki_dict.dict")  # Load a dictionary
logger.info('Finished loading data')

class TaggedWikiDocument(object):
    def __init__(self, wiki):
        self.wiki = wiki
        self.wiki.metadata = True
    def __iter__(self):
        for content, (page_id, title) in self.wiki.get_texts():
            yield TaggedDocument([c.decode("utf-8") for c in content], [title])

documents = TaggedWikiDocument(wiki) # 17544907 articles

# ======================= train the model ======================= 
logger.info('Starting training the model')
cores = multiprocessing.cpu_count()
print('number of cores:',cores)
model = Doc2Vec(dm=0, dbow_words=1, size=300, window=8, min_count=19, iter=10, workers=cores) # DBOW
# model = Doc2Vec(dm=1, dm_mean=1, size=300, window=8, min_count=19, iter =10, workers=cores) # PV-DM w/average

model.build_vocab(documents)
print(str(model))
logger.info('Built vocabulary')

model.train(documents)
model.save('./model_pv/wiki_model.mm')
logger.info('Finished training the model')