import os, pickle, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from PIL import Image

parser = argparse.ArgumentParser(description="Description for arguments")
parser.add_argument("-i", help="pickle file name used for MCMC (under folder 'cache')", required=True, dest="data_pickle")
parser.add_argument("-c", help="list of indices for annotation (covariates)", required=False, nargs="+", type=int, dest="idx_annot")
parser.add_argument("--burn_in", required=False, default=3000, type=int, dest="burnIn")
args = parser.parse_args()

args.idx_annot = sorted(args.idx_annot) if args.idx_annot is not None else []
os.makedirs("summary/" + args.data_pickle.split(".")[0], exist_ok=True)
os.makedirs("plot/" + args.data_pickle.split(".")[0], exist_ok=True)
os.makedirs("beta/" + args.data_pickle.split(".")[0], exist_ok=True)

files = sorted([i for i in os.listdir("mcmc_output") if args.data_pickle.split(".")[0] + "__" + "_".join(map(str, args.idx_annot)) + "__" in i])
if len(files) == 0:
	print("\x1b[0;30;41m" + "No result pickle file found, exit" + "\x1b[0m")
	exit()

with open("cache/" + args.data_pickle, "rb") as f:
	data = pickle.load(f)

var_y = data["y"].var().item()

with open("mcmc_output/" + files[0], "rb") as f:
	argument = pickle.load(f)["args"]

seed = np.full(argument["n_mcmc_chain"], np.nan)
mcmc_time = np.full(argument["n_mcmc_chain"], np.nan)
alpha_running_chain = np.full([int(argument["nIter"] / argument["thin"]), len(argument["idx_annot"]), argument["n_mcmc_chain"]], np.nan)
alpha_accp_rate = np.full([argument["nIter"], len(argument["idx_annot"]), argument["n_mcmc_chain"]], np.nan)
beta_running_chain = np.full([int(argument["nIter"] / argument["thin"]), data["x"].shape[1], argument["n_mcmc_chain"]], np.nan)
beta_cover = np.full(argument["n_mcmc_chain"], np.nan)
h2_running_chain = np.full([int(argument["nIter"] / argument["thin"]), argument["n_mcmc_chain"]], np.nan)
var_x_beta = np.full([int(argument["nIter"] / argument["thin"]), argument["n_mcmc_chain"]], np.nan)

for idx in range(argument["n_mcmc_chain"]):
	try:
		with open("mcmc_output/" + args.data_pickle.split(".")[0] + "__" + "_".join(map(str, args.idx_annot)) + "__" + str(idx) + ".pickle", "rb") as f:
			temp = pickle.load(f)
	except:
		continue

	seed[idx] = temp["seed"]
	mcmc_time[idx] = temp["time"]
	alpha_running_chain[:, :, idx] = temp["alpha_running_chain"] if temp["alpha_running_chain"] is not None else np.nan
	alpha_accp_rate[:, :, idx] = temp["alpha_accp_rate"] if temp["alpha_accp_rate"] is not None else np.nan
	beta_running_chain[:, :, idx] = temp["beta_running_chain"]
	beta_cover[idx] = temp["beta_cover_true"] if temp["beta_cover_true"] is not None else np.nan
	h2_running_chain[:, idx] = temp["h2_running_chain"]
	var_x_beta[:, idx] = (data["x"].to_numpy() @ temp["beta_running_chain"].T).var(axis=0)

df_seed = pd.DataFrame(seed[np.newaxis, :], columns=range(argument["n_mcmc_chain"]), index=pd.MultiIndex.from_tuples([("Seed", "")]))
df_alpha_header = pd.DataFrame("", columns=range(argument["n_mcmc_chain"]), index=pd.MultiIndex.from_tuples([("Alpha", "")]))
df_alpha = pd.DataFrame("", columns=range(argument["n_mcmc_chain"]), index=pd.MultiIndex.from_tuples(zip([""] * 5, ["true", "2.5%", "median", "97.5%", "accept_rate"]))) if len(argument["idx_annot"]) == 0 else pd.DataFrame(np.concatenate([np.concatenate([np.full([1, argument["n_mcmc_chain"]], data["alpha_true"].to_numpy()[i]) if data["alpha_true"] is not None else np.full([1, argument["n_mcmc_chain"]], np.nan), np.nanpercentile(alpha_running_chain[int(args.burnIn / argument["thin"]):, i, :], [2.5, 50, 97.5], axis=0), np.nanmean(alpha_accp_rate[args.burnIn:, i, :], axis=0)[np.newaxis, :]], axis=0) for i in range(len(argument["idx_annot"]))], axis=0), columns=range(argument["n_mcmc_chain"]), index=pd.MultiIndex.from_tuples(zip([""] * 5 * len(argument["idx_annot"]), [j + str(i) for i in argument["idx_annot"] for j in ["true_", "2.5%_", "median_", "97.5%_", "accept_rate_"]])))
df_h2_header = pd.DataFrame("", columns=range(argument["n_mcmc_chain"]), index=pd.MultiIndex.from_tuples([("h2", "")]))
df_h2 = pd.DataFrame(np.concatenate([np.full([1, argument["n_mcmc_chain"]], data["h2_true"].iloc[0, 0]) if data["h2_true"] is not None else np.full([1, argument["n_mcmc_chain"]], np.nan), np.nanpercentile(h2_running_chain[int(args.burnIn / argument["thin"]):, :], [2.5, 50, 97.5], axis=0)], axis=0), columns=range(argument["n_mcmc_chain"]), index=pd.MultiIndex.from_tuples(zip([""] * 4, ["true", "2.5%", "median", "97.5%"])))
df_mcmc_time = pd.DataFrame(mcmc_time[np.newaxis, :] / 3600, columns=range(argument["n_mcmc_chain"]), index=pd.MultiIndex.from_tuples([("Time", "hours")]))

summary_each_chain = pd.concat([df_seed, df_alpha_header, df_alpha, df_h2_header, df_h2, df_mcmc_time], axis=0)

df_alpha_header = pd.DataFrame(columns=range(3), index=pd.MultiIndex.from_tuples([("Alpha summary", "")]))
df_alpha = pd.DataFrame(columns=range(3), index=pd.MultiIndex.from_tuples(zip([""], ["alpha_cover_0"]))) if len(argument["idx_annot"]) == 0 else pd.DataFrame(columns=range(3), index=pd.MultiIndex.from_tuples(zip([""] * len(argument["idx_annot"]), ["alpha_cover_0_" + str(i) for i in argument["idx_annot"]])))
df_alpha.iloc[:, 0] = np.nan if len(argument["idx_annot"]) == 0 else np.nanmean(np.concatenate([(np.sign(np.nanpercentile(alpha_running_chain[int(args.burnIn / argument["thin"]):, i, :], 2.5, axis=0)) != np.sign(np.nanpercentile(alpha_running_chain[int(args.burnIn / argument["thin"]):, i, :], 97.5, axis=0)))[np.newaxis, :] for i in range(len(argument["idx_annot"]))], axis=0), axis=1)
df_alpha_accept_rate = pd.DataFrame(columns=range(3), index=pd.MultiIndex.from_tuples(zip([""], ["accept rate mean/2.5%/97.5%"]))) if len(argument["idx_annot"]) == 0 else pd.DataFrame(np.concatenate([np.nanmean((temp := summary_each_chain.loc[(slice(None), summary_each_chain.index.get_level_values(1).str.contains("accept_rate_")), :]), axis=1)[:, np.newaxis], np.nanpercentile(temp, [2.5, 97.5], axis=1).T], axis=1), columns=range(3), index=pd.MultiIndex.from_tuples(zip([""] * len(argument["idx_annot"]), ["accept rate mean/2.5%/97.5%_" + str(i) for i in argument["idx_annot"]])))
df_alpha_summary = pd.DataFrame(columns=range(3), index=pd.MultiIndex.from_tuples(zip([""] * 4, ["mean_of_median", "overall_mean", "overall_2.5%/50%/97.5%", "overall_0_percentile"]))) if len(argument["idx_annot"]) == 0 else pd.DataFrame(np.concatenate([np.array([[np.nanmean(np.nanmedian(alpha_running_chain[int(args.burnIn / argument["thin"]):, i, :], axis=0)), np.nan, np.nan], [np.nanmean(alpha_running_chain[int(args.burnIn / argument["thin"]):, i, :]), np.nan, np.nan], np.nanpercentile(alpha_running_chain[int(args.burnIn / argument["thin"]):, i, :], [2.5, 50, 97.5]), [(alpha_running_chain[int(args.burnIn / argument["thin"]):, i, :] < 0).sum() / np.prod(alpha_running_chain[int(args.burnIn / argument["thin"]):, i, :].shape), np.nan, np.nan]]) for i in range(len(argument["idx_annot"]))], axis=0), columns=range(3), index=pd.MultiIndex.from_tuples(zip([""] * 4 * len(argument["idx_annot"]), [j + str(i) for i in argument["idx_annot"] for j in ["mean_of_median_", "overall_mean_", "overall_2.5%/50%/97.5%_", "overall_0_percentile_"]])))
df_h2_header = pd.DataFrame(columns=range(3), index=pd.MultiIndex.from_tuples([("h2 summary", "")]))
df_h2_summary = pd.DataFrame(np.array([[np.nanmean(np.nanmedian(h2_running_chain[int(args.burnIn / argument["thin"]):, :], axis=0)), np.nan, np.nan], [np.nanmean(h2_running_chain[int(args.burnIn / argument["thin"]):, :]), np.nan, np.nan], np.nanpercentile(h2_running_chain[int(args.burnIn / argument["thin"]):, :], [2.5, 50, 97.5])]), columns=range(3), index=pd.MultiIndex.from_tuples(zip([""] * 3, ["mean_of_median", "overall_mean", "overall_2.5%/50%/97.5%"])))
r2 = var_x_beta / data["y"].to_numpy().flatten().var()
df_r2_header = pd.DataFrame(columns=range(3), index=pd.MultiIndex.from_tuples([("R2 summary", "")]))
df_r2_summary = pd.DataFrame(np.array([[np.nanmean(np.nanmedian(r2[int(args.burnIn / argument["thin"]):, :], axis=0)), np.nan, np.nan], [np.nanmean(r2[int(args.burnIn / argument["thin"]):, :]), np.nan, np.nan], np.nanpercentile(r2[int(args.burnIn / argument["thin"]):, :], [2.5, 50, 97.5])]), columns=range(3), index=pd.MultiIndex.from_tuples(zip([""] * 3, ["mean_of_median", "overall_mean", "overall_2.5%/50%/97.5%"])))

summary_overall = pd.concat([df_alpha_header, df_alpha, df_alpha_accept_rate, df_alpha_summary, df_h2_header, df_h2_summary, df_r2_header, df_r2_summary], axis=0)

pd.concat([summary_each_chain, summary_overall], axis=0).to_csv("summary/" + args.data_pickle.split(".")[0] + "/summary__" + "_".join(map(str, argument["idx_annot"])) + ".csv")
pd.DataFrame(np.nanmedian(beta_running_chain[int(args.burnIn / argument["thin"]):, :, :], axis=[0, 2])[:, np.newaxis], index=list(data["x"]), columns=["beta"]).to_parquet("beta/" + args.data_pickle.split(".")[0] + "/beta__" + "_".join(map(str, argument["idx_annot"])) + ".parquet", index=True)

ncol, dpi = 3, 300
# plot h2
fig, ax = plt.subplots(figsize=[7, 4.8], nrows=ceil(argument["n_mcmc_chain"] / ncol), ncols=ncol, sharex=True, sharey=True)
h2_cover_rate = []
h2_min, h2_max = h2_running_chain[int(args.burnIn / argument["thin"]):, :].min(), h2_running_chain[int(args.burnIn / argument["thin"]):, :].max()
r2_min, r2_max = r2[int(args.burnIn / argument["thin"]):, :].min(), r2[int(args.burnIn / argument["thin"]):, :].max()
plot_min, plot_max = min(h2_min, r2_min), max(h2_max, r2_max)
for i in range(argument["n_mcmc_chain"]):
	ax[int(i / ncol), i % ncol].plot(range(int(args.burnIn / argument["thin"])), h2_running_chain[:int(args.burnIn / argument["thin"]), i], linestyle=":", lw=0.8, c="blue")
	ax[int(i / ncol), i % ncol].plot(range(int(args.burnIn / argument["thin"]), int(argument["nIter"] / argument["thin"])), h2_running_chain[int(args.burnIn / argument["thin"]):, i], lw=0.8, c="blue")
	ax[int(i / ncol), i % ncol].plot(range(int(args.burnIn / argument["thin"])), r2[:int(args.burnIn / argument["thin"]), i], linestyle=":", lw=0.8, c="green")
	ax[int(i / ncol), i % ncol].plot(range(int(args.burnIn / argument["thin"]), int(argument["nIter"] / argument["thin"])), r2[int(args.burnIn / argument["thin"]):, i], lw=0.8, c="green")
	ax[int(i / ncol), i % ncol].axvline(x=int(args.burnIn / argument["thin"]), linestyle="--", lw=0.8, c="black")
	burn_in = (args.burnIn / argument["thin"] - (x_range := ax[int(i / ncol), i % ncol].get_xlim())[0]) / (x_range[1] - x_range[0])
	ax[int(i / ncol), i % ncol].set_ylim(plot_min - 0.05 * (plot_max - plot_min), plot_max + 0.05 * (plot_max - plot_min))
	if data["h2_true"] is not None:
		if summary_each_chain.loc[("", "2.5%"), i] <= data["h2_true"].to_numpy().item() <= summary_each_chain.loc[("", "97.5%"), i]:
			ax[int(i / ncol), i % ncol].axhline(y=data["h2_true"].to_numpy().item(), xmin=burn_in, linestyle="--", lw=0.8, c="grey")
			h2_cover_rate += [1]
		else:
			ax[int(i / ncol), i % ncol].axhline(y=data["h2_true"].to_numpy().item(), xmin=burn_in, linestyle="--", lw=0.8, c="red")
			h2_cover_rate += [0]
	else:
		ax[int(i / ncol), i % ncol].axhline(y=np.nanmedian(h2_running_chain[int(args.burnIn / argument["thin"]):, :]), xmin=burn_in, linestyle="--", lw=0.8, c="brown")
	ax[int(i / ncol), i % ncol].set_title(str(i))

if data["h2_true"] is not None:
	fig.suptitle("h2_hat\n(Iter=" + str(argument["nIter"]) + ",burnIn=" + str(args.burnIn) + ",thin=" + str(argument["thin"]) + ",cover true value=" + str(np.mean(h2_cover_rate)) + ",idx_annot=[" + ",".join(map(str, argument["idx_annot"])) + "])")
	fig.tight_layout()
	fig.savefig("plot/" + args.data_pickle.split(".")[0] + "/h2_" + str(data["h2_true"].to_numpy().item()) + "__" + "_".join(map(str, argument["idx_annot"])) + ".png", dpi=dpi)
else:
	fig.suptitle("h2_hat:blue;var(XBeta_hat)/varY:green\n(Iter=" + str(argument["nIter"]) + ",burnIn=" + str(args.burnIn) + ",thin=" + str(argument["thin"]) + ",idx_annot=[" + ",".join(map(str, argument["idx_annot"])) + "])")
	fig.tight_layout()
	fig.savefig("plot/" + args.data_pickle.split(".")[0] + "/h2__" + "_".join(map(str, argument["idx_annot"])) + ".png", dpi=dpi)

plt.close()

# plot alpha
if len(argument["idx_annot"]) != 0:
	for idx in range(len(argument["idx_annot"])):
		fig, ax = plt.subplots(figsize=[7, 4.8], nrows=ceil(argument["n_mcmc_chain"] / ncol), ncols=ncol, sharex=True, sharey=True)
		alpha_cover_rate = []
		for i in range(argument["n_mcmc_chain"]):
			ax[int(i / ncol), i % ncol].axhline(y=0, lw=0.8, c="green")
			ax[int(i / ncol), i % ncol].plot(range(int(args.burnIn / argument["thin"])), alpha_running_chain[:int(args.burnIn / argument["thin"]), idx, i], linestyle=":", lw=0.8, c="blue")
			ax[int(i / ncol), i % ncol].plot(range(int(args.burnIn / argument["thin"]), int(argument["nIter"] / argument["thin"])), alpha_running_chain[int(args.burnIn / argument["thin"]):, idx, i], lw=0.8, c="blue")
			ax[int(i / ncol), i % ncol].axvline(x=int(args.burnIn / argument["thin"]), linestyle="--", lw=0.8, c="black")
			burn_in = (args.burnIn / argument["thin"] - (x_range := ax[int(i / ncol), i % ncol].get_xlim())[0]) / (x_range[1] - x_range[0])
			if data["alpha_true"] is not None:
				if summary_each_chain.loc[("", "2.5%_" + str(idx)), i] <= data["alpha_true"].to_numpy().flatten()[idx] <= summary_each_chain.loc[("", "97.5%_" + str(idx)), i]:
					ax[int(i / ncol), i % ncol].axhline(y=data["alpha_true"].to_numpy().flatten()[idx], xmin=burn_in, linestyle="--", lw=0.8, c="grey")
					alpha_cover_rate += [1]
				else:
					ax[int(i / ncol), i % ncol].axhline(y=data["alpha_true"].to_numpy().flatten()[idx], xmin=burn_in, linestyle="--", lw=0.8, c="red")
					alpha_cover_rate += [0]
			else:
				ax[int(i / ncol), i % ncol].axhline(y=np.nanmedian(alpha_running_chain[int(args.burnIn / argument["thin"]):, idx, :]), xmin=burn_in, linestyle="--", lw=0.8, c="brown")
			ax[int(i / ncol), i % ncol].set_title(str(i))

		if data["alpha_true"] is not None:
			fig.suptitle("alpha running plot\n(Iter=" + str(argument["nIter"]) + ",burnIn=" + str(args.burnIn) + ",thin=" + str(argument["thin"]) + ",cover true value=" + str(np.mean(alpha_cover_rate)) + ",idx_annot=" + str(argument["idx_annot"][idx]) + "/[" + ",".join(map(str, argument["idx_annot"])) + "])")
			fig.tight_layout()
			fig.savefig("plot/" + args.data_pickle.split(".")[0] + "/alpha_" + str(data["alpha_ture"].to_numpy().flatten()[idx]) + "__" + "_".join(map(str, argument["idx_annot"])) + "__" + str(idx) + ".png", dpi=dpi)
		else:
			fig.suptitle("alpha running plot:blue\n(Iter=" + str(argument["nIter"]) + ",burnIn=" + str(args.burnIn) + ",thin=" + str(argument["thin"]) + ",idx_annot=" + str(argument["idx_annot"][idx]) + "/[" + ",".join(map(str, argument["idx_annot"])) + "])")
			fig.tight_layout()
			fig.savefig("plot/" + args.data_pickle.split(".")[0] + "/alpha__" + "_".join(map(str, argument["idx_annot"])) + "__" + str(idx) + ".png", dpi=dpi)

		plt.close()

	plots = sorted([i for i in os.listdir("plot/" + args.data_pickle.split(".")[0]) if "alpha__" + "_".join(map(str, argument["idx_annot"])) + "__" in i])
	temp = []
	for i in plots:
		temp.append(Image.open("plot/" + args.data_pickle.split(".")[0] + "/" + i))
		os.remove("plot/" + args.data_pickle.split(".")[0] + "/" + i)

	new_im = Image.new('RGB', (temp[0].size[0], temp[0].size[1] * len(argument["idx_annot"])), (250, 250, 250))

	for idx, i in enumerate(temp):
		new_im.paste(i, (0, idx * temp[0].size[1]))

	new_im.save("plot/" + args.data_pickle.split(".")[0] + "/alpha__" + "_".join(map(str, argument["idx_annot"])) + ".png", "PNG")
