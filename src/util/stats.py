from math import sqrt

def cohensD(series1, series2):
	mean1=series1.mean()
	mean2=series2.mean()
	pooled_sdv=sqrt((len(series1-1)*series1.var()+len(series2-1)*series2.var())/(len(series1)+len(series2)-2))
	return abs(mean1-mean2)/(pooled_sdv)

def countModelParam(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)