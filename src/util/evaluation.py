from torch.nn import Softmax, LogSoftmax
from util.stats import countModelParam
from tqdm import tqdm
import torch


def processOneBatch(model, batch, device):
    b_input_ids = batch[0].to(device)
    b_labels = batch[0].to(device)
    b_masks = batch[1].to(device)
    outputs  = model(b_input_ids,  attention_mask = b_masks)
    return outputs


def evaluateDataset(dataloader, model, device, **kwargs):
    logSoftmax = LogSoftmax(dim=1) 
    softmax= Softmax(dim=1)
    outputs=[]
    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=model.config._name_or_path):
        with torch.no_grad():
            out = processOneBatch(model, batch, device)
            logit = out.logits.to('cpu')
            del out
            if kwargs.get('applicationKey'):
                batchLogProbs=logSoftmax(logit)
                batchProbs=softmax(logit)
                batchResult = getResultsRow(batch[2], model, batchLogits=logit, batchLogProbs=batchLogProbs, batchProbs = batchProbs, **kwargs) 
                outputs.extend(batchResult)
                pass
            else:
                batchLogits = getLogitsOfMaskToken(batch, logit)
                batchLogProbs = logSoftmax(batchLogits)
                batchResult = getResultsRow(batch[2] , model, batchLogProbs=batchLogProbs, batchLogits=batchLogits, **kwargs)
                outputs.extend(batchResult)
    return outputs


def getLogitsOfMaskToken(batch, batchLogits):
    elementsInBatch = len(batch[0])
    batchLogits = batchLogits[range(elementsInBatch),list(batch[3])]
    return batchLogits


def getResultsRow(batchMetaData, model, **kwargs):
    result=[]
    metaDataRow=[{key: value[i] for key, value in batchMetaData.items()} for i in range(len(batchMetaData["sentence"]))]
    if kwargs.get("applicationKey"):
        for logit, logProb, prob, metaData in zip(kwargs.get('batchLogits'),kwargs.get("batchLogProbs"), kwargs.get("batchProbs"), metaDataRow):
            searchTermRow={}
            searchTermRow.update(metaData)
            searchTermRow.update({"logitScore": float(logit[0])})
            searchTermRow.update({"logProb": float(logProb[0]) })
            searchTermRow.update({"prob": float(prob[0]) })
            searchTermRow.update({"modelParams": countModelParam(model)})
            searchTermRow.update({"modelName": model.config._name_or_path})
            result.append(searchTermRow)
    else:
        for logit, logProb, metaData in zip(kwargs.get("batchLogits"), kwargs.get("batchLogProbs"), metaDataRow):
            logitScoreDict = getScoreForEachSearchTerm(logit, kwargs.get("existingSearchTerms"))
            logProbScoreDict = getScoreForEachSearchTerm(logProb, kwargs.get("existingSearchTerms"))
            for gender in kwargs.get("existingSearchTerms").keys():
                searchTermRow={}
                searchTermRow.update(metaData)
                searchTermRow.update({"logitScore": logitScoreDict[gender]})
                searchTermRow.update({"genderTerm": gender})
                searchTermRow.update({"logProb": logProbScoreDict[gender]})
                searchTermRow.update({"modelParams": countModelParam(model)})
                searchTermRow.update({"modelName": model.config._name_or_path})
                result.append(searchTermRow)
    return result


def getScoreForEachSearchTerm(scoreVector, existingSearchTerms):
    results={}
    for genderTerm_en, genderTermID in existingSearchTerms.items():
        score = scoreVector[genderTermID]
        results[genderTerm_en]=float(score)
    return results