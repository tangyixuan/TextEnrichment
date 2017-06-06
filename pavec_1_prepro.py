'''
Resources
===
1. simple doc2vec:
https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-lee.ipynb
https://rare-technologies.com/doc2vec-tutorial/
===
2. save & load wikidata
https://williambert.online/2012/05/relatively-quick-and-easy-gensim-example-code/
===
3. doc2vec on wikidata:
https://markroxor.github.io/gensim/static/notebooks/doc2vec-wikipedia.html
http://textminingonline.com/training-word2vec-model-on-english-wikipedia-by-gensim

Y, June 2017
'''

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
wiki.dictionary.save("./model_pv/wiki_dict.dict")
# wiki = MmCorpus("./model_pv/wiki_corpus.mm")  # Revive a corpus
# wiki_dict = Dictionary.load("./model_pv/wiki_dict.dict")  # Load a dictionary
logger.info('Finished loading data')

logger.info('Starting serializing data')
MmCorpus.serialize("./model_pv/wiki_corpus.mm", wiki)
logger.info('Finished serializing data')

# 1.5h
logger.info('Starting saving text')
count = 0
output_file = open('./data/enwiki.text', 'w')
for text in wiki.get_texts():
    if count < 3:
        print('\ntext: ', text)
    count +=1
    output_file.write(b' '.join(text).decode('utf-8') + '\n')
    count+=1
    if count % 10000 == 0:
        logger.info("Saved " + str(count) + " articles")
output_file.close()
logger.info('Finished saving text')


class TaggedWikiDocument(object):
    def __init__(self, wiki):
        self.wiki = wiki
        self.wiki.metadata = True
    def __iter__(self):
        for content, (page_id, title) in self.wiki.get_texts():
            yield TaggedDocument([c.decode("utf-8") for c in content], [title])

documents = TaggedWikiDocument(wiki) # 17544907 articles

# ======================= pre-process ======================= 
# 1.45h
# target vocab size (as in paper): 915,715, min_count = 0: 8,874,790, 2: 4,396,090, 10: 1,402,481, 18: 970,774, 19: 937,856

logger.info('Decide number of min-count')
pre=Doc2Vec(min_count=0)
pre.scan_vocab(documents=documents)
 
for num in range(0,20):
    print('min_count: {}, size of vocab: '.format(num), pre.scale_vocab(min_count=num, dry_run=True)['memory']['vocab']/700)