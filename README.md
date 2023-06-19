# Politeness Stereotypes and Attack Vectors: Gender Stereotypes in Japanese and Korean Language Models

# Requirements

Setup your python environment via:

```
pip install -r requirements.txt
```

We used Python 3.9.12 for this project.

# Experiments 

## Representaional Bias Experiments

Run representational bias experiments via:

```
python ./src/maskScores.py	
	--lang LANG				Language (ja/ko)
	--template  TEMP		Template to use (SN: rep. bias template, SNL: modified location template)
	--id ID					ID string used to label experiment
	[--batchsize N] 		BatchSize for GPU (omit for CPU)
```

## Allocational Bias Experiments

Run allocational experiments via:

```
python ./src/hateDetectScores.py	
	--lang LANG				Language (ja)
	--template  TEMP		Template to use (HN: rep. bias template, HNS: attack/test template, HNT: training template, HMF: gender_only template, H: tweet_only)
	--application APP		Application (CB: Cyberbullying)
	--id ID					ID string used to label experiment
	[--batchsize N]			BatchSize for GPU (omit for CPU)
```

# Note:
 - `ja-toxicity-dataset.json` needs to be downloaded from [Surge AI](https://www.surgehq.ai/datasets/japanese-hate-speech-insults-and-toxicity-dataset)