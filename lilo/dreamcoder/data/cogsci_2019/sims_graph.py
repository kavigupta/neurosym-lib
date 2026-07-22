"""
Makes a paired bar plot showing similarity matrix correlations.
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == "__main__":

	sims = []
	metric_to_plot = ['unigrams', 'bigrams', 'embeddings'][2]
	with(open('sims.txt')) as f:
		for line in f:
			line = line.strip()
			domain, metric, expertise = line.split()[0].split(",")
			if metric == metric_to_plot:
				for correlation in line.split()[1].split(","):
					sims.append([str(domain),str(metric),str(expertise),float(correlation)])
	
	
	sims = pd.DataFrame(sims, columns=['Domain', "Metric", "Expertise", "Correlation"])
	sims.Domain = sims.Domain.astype(str)
	sims.Metric = sims.Metric.astype(str)
	sims.Expertise = sims.Expertise.astype(str)
	print(sims.dtypes)

	# Make a grouped barplot
	print(sims)

	g = sns.catplot(x="Domain", y="Correlation", hue="Expertise",  kind="bar", data=sims)
	g.set_ylabels("Correlation with Solution Similarities")
	g.savefig('./sim_correlations_pval_%s.png' % metric_to_plot)
	
	

	
