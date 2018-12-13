from VQA.PythonHelperTools.vqaTools.vqa import VQA
import numpy as np
import re
import pickle
import nltk
from collections import Counter
 

from Environment import dataDir, dataSubType
versionType ='v2_' # this should be '' when using VQA v2.0 dataset
taskType    ='OpenEnded' # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
dataType    ='mscoco'  # 'mscoco' only for v1.0. 'mscoco' for real and 'abstract_v002' for abstract for v1.0.
annFile     ='%s/Annotations/%s%s_%s_annotations.json'%(dataDir, versionType, dataType, dataSubType)
quesFile    ='%s/Questions/%s%s_%s_%s_questions.json'%(dataDir, versionType, taskType, dataType, dataSubType)
imgDir 		= '%s/Images/%s/' %(dataDir, dataSubType)
databaseFile ='%s/Database/%s.pickle'%(dataDir, dataSubType)

vqa = VQA(annFile,quesFile)
questions = [nltk.word_tokenize(question['question']) for question in vqa.questions['questions']]
answers = [[answer['answer'] for answer in ann['answers']] for ann in vqa.dataset['annotations']]
ids = [ann['question_id'] for ann in vqa.dataset['annotations']]
image_ids = [ann['image_id'] for ann in vqa.dataset['annotations']]

database = {
    'questions': questions,
    'answers': answers,
    'ids': ids,
    'image_ids': image_ids
}

with open(databaseFile, 'wb') as fp:
    pickle.dump(database, fp)
