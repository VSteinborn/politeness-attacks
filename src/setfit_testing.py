from datasets import load_dataset
from datasets import Dataset
import pandas as pd
import numpy as np
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer, SetFitModel
import transformers
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

seed = 0
transformers.set_seed(seed)

train_df = pd.read_csv('train_few_shot.csv', header=None, names=['label', 'text'])
train_ds = Dataset.from_pandas(train_df)
test_df = pd.read_csv('test_few_shot.csv', header=None, names=['label', 'text'])
test_ds = Dataset.from_pandas(test_df)
test_ds = Dataset.from_pandas(test_df)

def write_res(f1, precision, recall, tn, fn, tp, fp):
    f = open('res_setfit/res_{}'.format(seed), 'w')
    f.write('f1 score: {}\n'.format(f1))
    f.write('precision: {}\n'.format(precision))
    f.write('recall: {}\n'.format(recall))
    f.write('tn: {}, fn: {}, tp: {}, fp: {}\n'.format(tn, fn, tp, fp))
    f.close()

model = SetFitModel.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

trainer = SetFitTrainer(
    model=model,
    train_dataset=train_ds,
    loss_class=CosineSimilarityLoss,
    batch_size=32,
    num_iterations=20,
    num_epochs=1
)

trainer.train()
trainer.model._save_pretrained('output_setfit_{}'.format(seed))

predicted_labels = trainer.model(test_ds['text'])

f = open('res_labels_gender_test.txt', 'w')
f.write('\n'.join(str(i) for i in predicted_labels))
f.close()

true_labels = list(test_df.label)
f1 = f1_score(true_labels, predicted_labels, average='macro')
precision = precision_score(true_labels, predicted_labels, average='macro')
recall = recall_score(true_labels, predicted_labels, average='macro')
conf_matrix = confusion_matrix(true_labels, predicted_labels)
tn, fn, tp, fp = conf_matrix[0][0], conf_matrix[1][0], conf_matrix[1][1], conf_matrix[0][1]
print('f1: {}'.format(f1))
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('tn: {} fn: {} tp: {} fp: {}'.format(tn, fn, tp, fp))

write_res(f1, precision, recall, tn, fn, tp, fp)
