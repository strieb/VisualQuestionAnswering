import numpy as np
import re
import pickle
from collections import Counter
import json

def getMostcommon(tokens, n):
    common = Counter(tokens).most_common(n)
    bag = {answer: idx for idx, (answer, number) in enumerate(common)}
    commonLength = sum([count for key, count in common])
    print('Coverage: ' + str(commonLength/len(tokens)))
    return bag

def getAnswer(answers):
    counter = Counter(answers).most_common(1)
    return counter[0][0]


dataDir		='C:/ml/VQA'
versionType ='v2_' # this should be '' when using VQA v2.0 dataset
taskType    ='OpenEnded' # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
dataType    ='mscoco'  # 'mscoco' only for v1.0. 'mscoco' for real and 'abstract_v002' for abstract for v1.0.
dataSubType ='train2014'
databaseFile ='%s/Database/%s.pickle'%(dataDir, dataSubType)
questionsEncFile = '%s/Database/questions.json'%(dataDir)
answersEncFile = '%s/Database/answers.json'%(dataDir)

with open(databaseFile, 'rb') as fp:
    database = pickle.load(fp)
tokens = [str.lower(token) for question in database['questions'] for token in question]
questionEncoding = getMostcommon(tokens,2048)
answers = [getAnswer(answers) for answers in database['answers']]
# answers = [answer for answers in database['answers'] for answer in answers]
answerEncoding = getMostcommon(answers,2048)

with open(questionsEncFile, 'w') as fp:
    json.dump(questionEncoding, fp)
with open(answersEncFile, 'w') as fp:
    json.dump(answerEncoding, fp)