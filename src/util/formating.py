from datetime import date
import itertools
import pandas as pd
import torch
from torch.utils.data import Dataset

def dateStamp():
	return date.today().isoformat().replace('-','')

def filePrefix(startString,endString):
	return "{}{}{}".format(startString, dateStamp(), endString)

def buildSentences(sentenceCombinations, templateKey, **kwargs):
    template = sentenceCombinations["templates"][templateKey]
    args=[]
    if templateKey == "SNL":
        args.extend(
            [
                flattenGenderDict(sentenceCombinations["locations"]),
                sentenceCombinations["speakerVerbNouns"].items(),
                sentenceCombinations["speakerVerbEndings"].items(),
                sentenceCombinations["narratorVerbs"].items(),
            ]
        )
    elif templateKey == "SN":
        args.extend(
            [
                sentenceCombinations["speakerVerbNouns"].items(),
                sentenceCombinations["speakerVerbEndings"].items(),
                sentenceCombinations["narratorVerbs"].items()
            ]
        )
    elif templateKey in ["HN","HNS","HNT"]:
        args.extend(
            [
                [[text["Text"], text["Toxic/Not Toxic"]] for text in sentenceCombinations["toxicityDataset"]],
                sentenceCombinations["narratorVerbs"].items(),
            ]
        )
    elif templateKey in ["HMF","H"]:
        args.extend(
            [
                [[text["Text"], text["Toxic/Not Toxic"]] for text in sentenceCombinations["toxicityDataset"]],
            ]
        ) 
    if kwargs.get('applicationKey') == "CB":
         args.extend(
              [
                sentenceCombinations["genderTerms"].items()
              ]
         )
    sentenceIterable = itertools.product(*args)
    dfLine=[]
    for sentPieces in sentenceIterable:
        metaData, metaData_en = getSentenceMetadata(sentPieces, templateKey, **kwargs)
        sentence=insertTermsinSentence(template, metaData)
        metaData_en["sentence"]=sentence
        dfLine.append(metaData_en)
    sentenceData = pd.DataFrame(dfLine)
    return sentenceData

def flattenGenderDict(dictionary):
    result=[]
    for gender, terms in dictionary.items():
        for termEn, termJp in terms.items():
            result.append((gender, termEn, termJp))
    return result

def getSentenceMetadata(sentPieces, templateKey, **kwargs):
    if templateKey == "SNL":
        locGender, location_en, loc = sentPieces[0]
        speakerNoun_en, speakerNoun = sentPieces[1]
        speakerVerbEnding_en, speakerVerbEnding = sentPieces[2]
        narratorVerb_en, narratorVerb = sentPieces[3]
        metaData_en = {
            "locationGender": locGender,
            "location": location_en,
            "speakerNoun": speakerNoun_en,
            "speakerVerbEnding": speakerVerbEnding_en,
            "narratorVerb": narratorVerb_en,
        }
        metaData = {
            "location": loc,
            "speakerNoun": speakerNoun,
            "speakerVerbEnding": speakerVerbEnding,
            "narratorVerb": narratorVerb,
        }
        metaData["mask"]="{}"
    elif templateKey == "SN":
        speakerNoun_en, speakerNoun = sentPieces[0]
        speakerVerbEnding_en, speakerVerbEnding = sentPieces[1]
        narratorVerb_en, narratorVerb = sentPieces[2]
        metaData_en= {
            "speakerNoun": speakerNoun_en,
            "speakerVerbEnding": speakerVerbEnding_en,
            "narratorVerb": narratorVerb_en,
        }
        metaData = {
            "speakerNoun": speakerNoun,
            "speakerVerbEnding": speakerVerbEnding,
            "narratorVerb": narratorVerb,
        }
        metaData["mask"]="{}"
    elif templateKey in ["HN","HNS","HNT"]:
        tweet, toxicLabel = sentPieces[0]
        narratorVerb_en, narratorVerb = sentPieces[1]
        metaData_en= {
            "tweet": tweet,
            "narratorVerb": narratorVerb_en,
            "toxicLabel": toxicLabel,
        }
        metaData = {
            "tweet": tweet,
            "narratorVerb": narratorVerb,
            "toxicLabel": toxicLabel,
        }
        metaData["mask"]="{}"
    elif templateKey in ["HMF","H"]:
        tweet, toxicLabel = sentPieces[0]
        metaData_en= {
            "tweet": tweet,
            "toxicLabel": toxicLabel
        }
        metaData = {
            "tweet": tweet,
            "toxicLabel": toxicLabel
        }
        metaData["mask"]="{}"

    if kwargs.get('applicationKey') == "CB":
        genderTerm_en, genderTerm =sentPieces[-1]
        metaData["mask"]=genderTerm
        metaData_en["genderTerm"]=genderTerm_en
    return metaData, metaData_en

def insertTermsinSentence(template, metaData):
    sentence=template.format(**metaData)
    return sentence

def getMaxTokenLength(sentences, tokenizer):
    return max([len(tokenizer.encode(s)) for s in sentences])

def insertMaskInSentence(sentencedf, tokenizer):
    maskString=tokenizer.mask_token
    return sentencedf["sentence"].format(maskString)

def findIndexInInputMachingQuery(input: torch.Tensor, querry) -> int:
    maskTokens = input == querry
    return int(maskTokens.nonzero(as_tuple=True)[0])

def tokenize_seq(sentence, tokenizer, max_length):
  return tokenizer(sentence, truncation=True, max_length=max_length, padding="max_length", return_tensors='pt')

class SentenceDataset(Dataset): 
    def __init__(self, tokenizer, sentenceData, maxInputLength, **kwargs):
        self.tokenizer = tokenizer 
        self.input_ids = []
        self.attn_masks = []
        self.meta_data=[]
        self.mask_index=[]
        self.applicationKey=kwargs.get('applicationKey')

        for row in sentenceData.to_dict(orient='records'):      
            encodings = tokenize_seq(row["sentence"], tokenizer, maxInputLength)
            self.input_ids.append(encodings['input_ids'].squeeze(0))
            self.attn_masks.append(encodings['attention_mask'].squeeze(0))
            self.meta_data.append(row)
            if self.applicationKey == "CB":
                pass
            else:
                maskIndex=findIndexInInputMachingQuery(encodings['input_ids'].squeeze(0), tokenizer.mask_token_id)
                self.mask_index.append(maskIndex)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        if self.applicationKey == "CB":
            return self.input_ids[idx], self.attn_masks[idx], self.meta_data[idx]
        else:
            return self.input_ids[idx], self.attn_masks[idx], self.meta_data[idx], self.mask_index[idx] 


def getTokenIDOfExistingSearchTerms(tokenizer, genderSearchTerms):
    searchTokenIDS={}
    for genderTerm_en, genderTerm in genderSearchTerms.items():
        try:
            genderTermID = tokenizer.vocab[genderTerm]
            searchTokenIDS[genderTerm_en]=genderTermID
        except:
            continue
    return searchTokenIDS