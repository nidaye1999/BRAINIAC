import pandas as pd
import re
import os

dir_path = "XXXXX"  # directory contains connectivity data

annot = list(pd.read_parquet(dir_path + "ABCD_fd0p2mm_censor-10min_conndata-network_connectivity.parquet"))[1:]
pair = set(map(lambda x: frozenset(re.sub("_[0-9]+", "", x).split("-")), annot))
data = pd.DataFrame(index=annot, columns=sorted([list(i)[0] + "-" + list(i)[0] for i in pair if len(i) == 1]) + sorted([sorted(i)[0] + "-" + sorted(i)[1] for i in pair if len(i) != 1]))
for i in data.index:
	data.loc[i, "-".join(sorted(re.sub("_[0-9]+", "", i).split("-")))] = 1

os.makedirs("cache", exist_ok=True)
# filter data and save
data.loc[:, sorted([list(i)[0] + "-" + list(i)[0] for i in pair if len(i) == 1])].fillna(0).to_csv("cache/annotation_binary__within.csv")
