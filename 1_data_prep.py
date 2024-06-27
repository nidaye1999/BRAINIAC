import os, pickle, argparse
from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from scipy.stats import boxcox
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Description for arguments")
parser.add_argument("-y", help="y, e.g., nihtbx_cryst_uncorrected", required=True, dest="y")
parser.add_argument("-a", help="annotation csv file under folder 'cache'", required=True, dest="annotation")
parser.add_argument("-c", help="covariates, e.g., age, sex, etc.", required=False, dest="covariate", nargs="+")
parser.add_argument("--alpha_true", required=False, default=None, type=float, dest="alpha_true")
parser.add_argument("--beta_true", required=False, default=None, type=float, dest="beta_true")
parser.add_argument("--h2_true", required=False, default=None, type=float, dest="h2_true")
parser.add_argument("--block_number", required=False, default=20, type=int, dest="block_number")
parser.add_argument("--random_block_size", required=False, default="False", dest="random_block_size")
parser.add_argument("--data_path", help="directory contains data, check code for details", required=True, dest="data_path")
parser.add_argument("--seed", required=False, default=10, type=int, dest="seed")
args = parser.parse_args()

np.random.seed(args.seed)
args.random_block_size = eval(args.random_block_size)
args.covariate = args.covariate if args.covariate is not None else []

if args.data_path[-1] != "/":
	args.data_path += "/"

data_abcd = (temp := pd.read_parquet(args.data_path + "rds4_full_table_v82.parquet"))[temp["eventname"] == "baseline_year_1_arm_1"]
data_abcd.set_index("subjectid", inplace=True)
data_abcd.index.name = None
data_abcd = data_abcd[~data_abcd.index.duplicated()]

data_pc = pd.read_csv(args.data_path + "DAIRC/ABCD_20220428.updated.nodups.curated_pcair.tsv", index_col=0, sep="\t")
data_pc = data_pc[~data_pc.index.duplicated()]

data_cbcl_pfactor = pd.read_csv(args.data_path + "DAIRC/ABCD_lavaan_cbcl_pfactor.csv", index_col=0)
data_cbcl_pfactor.index.name = None
data_cbcl_pfactor = data_cbcl_pfactor[~data_cbcl_pfactor.index.duplicated()]

connectivity = pd.read_parquet(args.data_path + "Curated/ABCD_fd0p2mm_censor-10min_conndata-network_connectivity.parquet")
connectivity.set_index("id", inplace=True)
connectivity.index.name = None
connectivity = connectivity[~connectivity.index.duplicated()]

data = pd.merge(pd.merge(data_abcd, data_pc, left_index=True, right_index=True), data_cbcl_pfactor.drop(columns="site_id_l"), left_index=True, right_index=True)[[args.y] + args.covariate].dropna()
data = pd.get_dummies(data, prefix_sep="__", drop_first=True, dtype=int)
args.covariate = list(data)[1:]

data = pd.merge(data, connectivity, left_index=True, right_index=True).dropna()

# control for covariates (linear regression)
temp, args.covariate = deepcopy(args.covariate), []
for i in temp:
	temp_1 = pd.concat([data[args.covariate + [i]], pd.DataFrame(np.ones((data.shape[0], 1)), index=data.index, columns=["intercept"])], axis=1)
	if np.linalg.matrix_rank(temp_1) < temp_1.shape[1]:
		print("\x1b[0;30;41m" + "Found linear dependence in covariates:" + "\x1b[0m", i)
	else:
		args.covariate.append(i)

if len(args.covariate) > 0:
	print("\x1b[0;30;42m" + "control for covariates" + "\x1b[0m")
	print(args.covariate)
	temp = np.concatenate([scale(data[args.covariate]), np.ones((data.shape[0], 1))], axis=1)
	lm_pred = (temp @ (np.linalg.pinv(temp.T @ temp, hermitian=True) @ temp.T @ data[list(connectivity) + [args.y]])).set_index(data.index)
else:
	print("\x1b[0;30;42m" + "no covariate controled" + "\x1b[0m")
	lm_pred = pd.DataFrame(np.zeros([len(data.index), len(list(connectivity) + [args.y])]), index=data.index, columns=list(connectivity) + [args.y])

# X, Y & Z
x = pd.DataFrame(scale(data[list(connectivity)] - lm_pred[list(connectivity)]), index=data.index, columns=list(connectivity))
y = pd.DataFrame(scale(boxcox((y_res_org := data[args.y] - lm_pred[args.y]) - min(y_res_org.min() - 1, 0))[0]), index=data.index, columns=[args.y])

os.makedirs("plot", exist_ok=True)
pd.DataFrame({"org": y_res_org, "boxcox": y[args.y]}).hist(bins=50)
plt.savefig("plot/" + args.y + "_boxcox.png")
plt.close()

z = pd.DataFrame(scale(temp := pd.read_csv('cache/' + args.annotation, index_col=0)), index=temp.index, columns=temp.columns)

# =================================================
# connectivity partition
if args.random_block_size:
	t = np.random.uniform(0.2, 0.95, args.block_number)
	block_size = (t / t.sum() * connectivity.shape[1]).astype(int)
else:
	block_size = np.array([int(connectivity.shape[1] / args.block_number)] * args.block_number)

for _ in range(connectivity.shape[1] - block_size.sum()):
	block_size[np.random.choice(args.block_number)] += 1

lambda_diag, lambda_off_diag = [], []
for i in range(args.block_number):
	lambda_diag.append((temp := x.iloc[:, block_size[:i].sum():block_size[:(i + 1)].sum()]).T @ temp / x.shape[0])
	lambda_off_diag.append(temp.T @ x.iloc[:, list(range(block_size[:i].sum())) + list(range(block_size[:(i + 1)].sum(), x.shape[1]))] / x.shape[0])

b_hat_all = x.T @ y / x.shape[0]

# =================================================
# save
with open("cache/ImgData__" + str(x.shape[0]) + "__" + str(x.shape[1]) + "__" + args.annotation.split("__")[-1].split(".")[0] + "__" + args.y + ".pickle", "wb") as f:
	pickle.dump({"args": vars(args), "b_hat_all": b_hat_all, "z": z, "block_size": block_size, "alpha_true": args.alpha_true, "beta_true": args.beta_true, "h2_true": args.h2_true, "x": x, "y": y, "lambda_diag": lambda_diag, "lambda_off_diag": lambda_off_diag}, f)
