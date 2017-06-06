
import sys
import os
import logging
import time
import multiprocessing
from six import iteritems
import smart_open

from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.corpora import TextCorpus, MmCorpus, Dictionary
from gensim.utils import simple_preprocess

# ======================= create logger ======================= 
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logger.info("Running %s" % ' '.join(sys.argv))

# ======================= load data ======================= 
# 12 min
logger.info('Starting loading data')
wiki = WikiCorpus("./data/enwiki-20170520-pages-articles.xml.bz2",dictionary={}) 
logger.info('Finished loading data')
 
class TaggedWikiDocument(object):
    def __init__(self, wiki):
        self.wiki = wiki
        self.wiki.metadata = True
    def __iter__(self):
        for content, (page_id, title) in self.wiki.get_texts():
            yield TaggedDocument([c.decode("utf-8") for c in content], [title])
   
documents = TaggedWikiDocument(wiki)

# # ======================= train the model ======================= 
logger.info('Starting training the model')
model = Doc2Vec(dm=0, dbow_words=1, size=300, window=8, min_count=19, iter=10, workers=multiprocessing.cpu_count()) # DBOW
# model = Doc2Vec(dm=1, dm_mean=1, size=300, window=8, min_count=19, iter =10, workers=cores) # PV-DM w/average

model.build_vocab(documents) 
model.train(documents,total_examples=model.corpus_count, epochs=model.iter)
model.save('./model_pv/wiki_model.mm')
logger.info('Finished training the model')
