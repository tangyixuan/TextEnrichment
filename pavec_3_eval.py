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
# ======================= evaluate ======================= 

# model = Doc2Vec.load('./model_pv/wiki_model.mm')

print(model.docvecs.most_similar(positive=["Data Mining"], topn=20))

sentences = []
for s in sentences:
    vector = model.infer_vector(s)



# import gensim.models as g
# 
# test_docs="toy_data/test_docs.txt"
# output_file="toy_data/test_vectors.txt"
# 
# #inference hyper-parameters
# start_alpha=0.01
# infer_epoch=1000
# 
# #load model
# m = g.Doc2Vec.load(model)
# test_docs = [ x.strip().split() for x in codecs.open(test_docs, "r", "utf-8").readlines() ]
# 
# #infer test vectors
# output = open(output_file, "w")
# for d in test_docs:
#     output.write( " ".join([str(x) for x in m.infer_vector(d, alpha=start_alpha, steps=infer_epoch)]) + "\n" )
# output.flush()
# output.close()
