import argparse
import json
import pandas as pd
import os
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification
import torch
from torch.cuda import is_available as cuda_available


def readArgs():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--lang", help="Language to evaluate", required=True, choices=["ja", "ko"]
	)
	parser.add_argument(
		"--data_files_json", help="File that points to locations of data source files", 
		default="./data/file-locations.json"
	)
	parser.add_argument(
		"--template", required=True, choices=["SNL", "SN", "HN", "HNS", "HNT", "HMF", "H"] 
	)
	parser.add_argument(
		"--id", help="experiment ID to name files", default='test'
	)
	parser.add_argument(
		"--batchsize", help="indicatte batch size", default=None, type=int
	)
	parser.add_argument(
		"--application", help = "type of application", choices=["CB"], default=None
	)
	args = parser.parse_args()
	return args


def loadFiles(args):
	with open(args.data_files_json) as f:
		dataFilePaths = json.load(f)

	sentenceCombinations={}
	for fileType, fileLocation in dataFilePaths[args.lang].items():
		with open(fileLocation) as f:
			fileData=json.load(f)
			sentenceCombinations[fileType]=fileData

	return sentenceCombinations


def dfSave(dfLines, runId):
	dfOutput=pd.DataFrame(dfLines)
	try:
		os.mkdir('./out/{0}'.format(runId))
	except FileExistsError:
		print("Writing to existing directory.")
	except:
		print("Parent directory does not exist.")
	dfOutput.to_csv(f'./out/{runId}/{runId}-raw.csv')


def loadModel(modelName, device, **kwargs):
	if kwargs.get('applicationKey') == 'CB':
		tokenizer = AutoTokenizer.from_pretrained(modelName)
		model = AutoModelForSequenceClassification.from_pretrained(modelName)
	else:
		tokenizer = AutoTokenizer.from_pretrained(modelName)
		model = AutoModelForMaskedLM.from_pretrained(modelName)
	model.eval()
	model.to(device)
	return model, tokenizer

def getDevice(batchSize):
	if batchSize and cuda_available():
		return torch.device("cuda")
	else:
		return torch.device("cpu")