import argparse, jax, os, pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import scale
from scipy.stats import boxcox

jax.config.update("jax_enable_x64", True)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

parser = argparse.ArgumentParser(description="Description for arguments")
parser.add_argument("-i", help="pickle file name used for MCMC (under folder 'cache')", required=True, dest="data_pickle")
parser.add_argument("-c", help="list of indices for annotation (covariates)", required=False, nargs="+", type=int, dest="idx_annot")
parser.add_argument("-t", help="list of target variable in HCP", required=True, nargs="+", dest="test_vec")
parser.add_argument("--data_path", help="directory contains data, check code for details", required=True, dest="data_path")
parser.add_argument("--svd_thrd", required=False, default=1.1, type=float, dest="svd_thrd")
args = parser.parse_args()

args.idx_annot = sorted(args.idx_annot) if args.idx_annot is not None else []

if args.data_path[-1] != "/":
	args.data_path += "/"


def truncated_svd(x):
	u, s, vh = np.linalg.svd(x, full_matrices=False)
	# u, s, vh = jax.numpy.linalg.svd(data["x"].to_numpy(), full_matrices=False)
	if args.svd_thrd >= 1:
		return u, s, vh
	else:
		for i in range(1, s.shape[0] + 1):
			if sum(s[:i] ** 2) / sum(s ** 2) >= args.svd_thrd:
				break
		return u[:, :i], s[:i], vh[:i, :]


# load x, y, HCPD data, Rong's model beta
with open("cache/" + args.data_pickle, "rb") as f:
	data = pickle.load(f)

mat_hcp = pd.read_parquet(args.data_path + "Curated/HCPD_fd0p2mm_censor-10min_conndata-network_connectivity.parquet").set_index("id")
mat_hcp.index = "HCD" + mat_hcp.index
pheno_hcp = pd.merge(pd.read_parquet(args.data_path + "hcpd_v1.parquet").set_index("src_subject_id"), pd.read_csv(args.data_path + "DAIRC/HCPD_lavaan_cbcl_pfactor.csv").set_index("src_subject_id").drop(columns="subjectkey"), left_index=True, right_index=True)
betas_rong = pd.read_parquet("beta/" + args.data_pickle.split(".")[0] + "/beta__" + "_".join(map(str, args.idx_annot)) + ".parquet")

print("calculating SVD")
svd_out = truncated_svd(data["x"])
print("SVD done")

pvs_score = pd.DataFrame(scale(mat_hcp) @ svd_out[2].T @ scale(svd_out[0]).T @ data["y"].to_numpy(), index=mat_hcp.index, columns=["pvs_score"])
rong_score = pd.DataFrame(scale(mat_hcp) @ betas_rong.to_numpy(), index=mat_hcp.index, columns=["rong_score"])

xy = pd.concat([pheno_hcp.loc[:, args.test_vec + ["age", "sex"]], pvs_score, rong_score], axis=1, join="inner").dropna()
xy.loc[:, ["pvs_score", "rong_score"]] = scale(xy.loc[:, ["pvs_score", "rong_score"]])
for y_hcp in list(args.test_vec):
	for variables in [["age", "sex"], ["age", "sex", "pvs_score"], ["age", "sex", "rong_score"]]:
		x = sm.add_constant(pd.get_dummies(xy[variables], prefix_sep="__", drop_first=True, dtype=int))
		y = pd.DataFrame({y_hcp: boxcox(xy[y_hcp] - min(xy[y_hcp].min() - 1, 0))[0]}, index=xy.index)
		model = sm.OLS(y, x)
		results = model.fit()
		print(results.summary())
