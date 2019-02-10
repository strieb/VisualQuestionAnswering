import numpy as np
import re
import pickle
from collections import Counter
import json

def getMostcommon(tokens, n):
    common = Counter(tokens).most_common(20000)
    bag = {answer: idx for idx, (answer, number) in enumerate(common) if number >= n}
    print('Length: '+str(len(bag)))
    l1 = sum([number for (anser,number) in common])
    l2 = sum([number for (anser,number) in common if number >= n])
    print('Length: '+str(l2/l1))
    return bag

def getAnswer(answers):
    counter = Counter(answers).most_common(1)
    return counter[0][0]


from Environment import DATADIR
dataSubType ='train2014'
versionType ='v2_' # this should be '' when using VQA v2.0 dataset
taskType    ='OpenEnded' # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
dataType    ='mscoco'  # 'mscoco' only for v1.0. 'mscoco' for real and 'abstract_v002' for abstract for v1.0.
databaseFile ='%s/Database/%s.pickle'%(DATADIR, dataSubType)
questionsEncFile = '%s/Database/questions.json'%(DATADIR)
answersEncFile = '%s/Database/answers.json'%(DATADIR)

with open(databaseFile, 'rb') as fp:
    database = pickle.load(fp)
tokens = [str.lower(token) for question in database['questions'] for token in question]
questionEncoding = getMostcommon(tokens,5)
# answers = [getAnswer(answers) for answers in database['answers']]
answers = [answer for answers in database['answers'] for answer in answers]
answerEncoding = getMostcommon(answers,100)

with open(questionsEncFile, 'w') as fp:
    json.dump(questionEncoding, fp)
with open(answersEncFile, 'w') as fp:
    json.dump(answerEncoding, fp)