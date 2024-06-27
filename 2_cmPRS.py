import os, pickle, argparse, time, jax
import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.special import softmax
from scipy.stats import t
from scipy.optimize import minimize
from multiprocess.pool import Pool

os.makedirs("mcmc_output", exist_ok=True)
jax.config.update("jax_enable_x64", True)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
parser = argparse.ArgumentParser(description="Description for arguments")
parser.add_argument("-i", help="input data (pickle file, under folder 'cache')", required=True, dest="data_pickle")
parser.add_argument("-c", help="list of indices for annotation (covariates)", required=False, nargs="+", type=int, dest="idx_annot")
parser.add_argument("--step_size", required=False, default=0.04, type=float, dest="step_size")
parser.add_argument("--thin", required=False, default=10, type=int, dest="thin")
parser.add_argument("--n_iter", required=False, default=10000, type=int, dest="nIter")
parser.add_argument("--use_rand_cov", required=False, default="False", dest="useRandCov")
parser.add_argument("--alpha_sigma", required=False, default=10, type=float, dest="alpha_sigma")
parser.add_argument("--h2_initial", required=False, default=0.5, type=float, dest="h2_initial")
parser.add_argument("--n_h2_grid", required=False, default=100, type=int, dest="n_h2_grid")
parser.add_argument("--seed", required=False, default=10, type=int, dest="seed")
parser.add_argument("--adj_bias", required=False, default="True", dest="adjBias")
parser.add_argument("--full_quad", required=False, default="True", dest="full_quad")
parser.add_argument("--n_mcmc_chain", required=False, default=6, type=int, dest="n_mcmc_chain")
args = parser.parse_args()

args.useRandCov, args.adjBias, args.full_quad = eval(args.useRandCov), eval(args.adjBias), eval(args.full_quad)

# read-in data
with open("cache/" + args.data_pickle, "rb") as f:
	data = pickle.load(f)

b_hat_all, block_size = data["b_hat_all"].to_numpy().flatten(), data["block_size"].tolist()
(n_subject, n_connectivity) = data["x"].shape
alpha_true, beta_true, h2_true = data["alpha_true"], data["beta_true"], data["h2_true"]
lambda_diag, lambda_off_diag = {i: j.to_numpy() for i, j in enumerate(data["lambda_diag"])}, {i: j.to_numpy() for i, j in enumerate(data["lambda_off_diag"])}
y2 = (data["y"].T @ data["y"]).iloc[0, 0]

args.idx_annot = sorted(args.idx_annot) if args.idx_annot is not None else []
np.random.seed(args.seed)

if args.useRandCov:
	args.idx_annot = [-1]
	z = pd.DataFrame(np.random.normal(size=[n_connectivity, 1]), index=list(data["x"]), columns=["normal"])
	alpha_true = pd.Series(np.zeros(len(args.idx_annot)), index=["alpha"])
else:
	if len(args.idx_annot) > 0:
		z = data["z"].iloc[:, args.idx_annot]
	else:
		z = pd.DataFrame(np.zeros([n_connectivity, 1]), index=list(data["x"]), columns=["zeros"])

n_covariate = len(args.idx_annot)
z = z.to_numpy()
off_diag_row = {i: list(range(sum(block_size[:i]), sum(block_size[:(i + 1)]))) for i in range(len(block_size))}
off_diag_col = {i: list(range(sum(block_size[:i]))) + list(range(sum(block_size[:(i + 1)]), n_connectivity)) for i in range(len(block_size))}


def cm_variance(idx, seed):
	def log_p_h2(h2_grid: np.ndarray, y2: int | float, beta: np.ndarray, b_hat: np.ndarray, quad: int | float, psi: np.ndarray, a: int | float, b: int | float) -> np.ndarray:
		temp = psi.sum() * (beta ** 2 / psi).sum() / h2_grid + (y2 - 2 * n_subject * (b_hat * beta).sum() + n_subject * quad) / (1 - h2_grid)
		log_p = (-0.5 * n_connectivity + a - 1) * np.log(h2_grid) + (-0.5 * n_subject + b - 1) * np.log(1 - h2_grid) - 0.5 * temp
		return log_p

	def log_p_alpha(x_alpha: int | float, index: int, mask: list[int], alpha: np.ndarray, z_numpy: np.ndarray, alpha_sigma: np.ndarray, beta: np.ndarray, h2: int | float) -> int | float:
		psi = np.exp(z_numpy[:, mask] @ alpha[mask] + z_numpy[:, index] * x_alpha)
		log_p = -0.5 * (z_numpy[:, index] * x_alpha - np.log(psi.sum())).sum() - 0.5 * (psi.sum() / h2) * (beta ** 2 / psi).sum() - 0.5 * x_alpha ** 2 / alpha_sigma[index]
		return -log_p.item()

	def log_p_alpha_first_derivative(x_alpha: int | float, index: int, mask: list[int], alpha: np.ndarray, z_numpy: np.ndarray, alpha_sigma: np.ndarray, beta: np.ndarray, h2: int | float) -> int | float:
		psi = np.exp(z_numpy[:, mask] @ alpha[mask] + z_numpy[:, index] * x_alpha)
		temp = (beta ** 2 * z_numpy[:, index] / psi).sum()
		grad = -0.5 * (z_numpy[:, index] - (z_numpy[:, index] * psi).sum() / psi.sum()).sum() + 0.5 * (psi.sum() / h2) * temp - 0.5 * ((z_numpy[:, index] * psi).sum() / h2) * (beta ** 2 / psi).sum() - x_alpha / alpha_sigma[index]
		return -grad.item()

	def draw_alpha_log(index: int, alpha: np.ndarray, z_numpy: np.ndarray, alpha_sigma: np.ndarray, beta: np.ndarray, h2: int | float, gtol_bfgs: float) -> tuple[int | float, int]:
		mask = list(range(index)) + list(range(index + 1, n_covariate))
		out = minimize(fun=log_p_alpha, x0=alpha[index], args=(index, mask, alpha, z_numpy, alpha_sigma, beta, h2), method="BFGS", jac=log_p_alpha_first_derivative, options={"gtol": gtol_bfgs})
		alpha_new = t.rvs(df=4, loc=out.x, scale=args.step_size)  # scale increase, the std of random variates increase.

		log_p_alpha_initial = -log_p_alpha(alpha[index], index, mask, alpha, z_numpy, alpha_sigma, beta, h2)
		log_q_alpha_initial = np.log(t.pdf(alpha[index], df=4, loc=out.x, scale=args.step_size))
		log_p_alpha_new = -log_p_alpha(alpha_new, index, mask, alpha, z_numpy, alpha_sigma, beta, h2)
		log_q_alpha_new = np.log(t.pdf(alpha_new, df=4, loc=out.x, scale=args.step_size))

		if np.exp(log_q_alpha_initial) == 0 or np.exp(log_q_alpha_new) == 0:
			rho = min(0, log_p_alpha_new - log_p_alpha_initial)
		else:
			rho = min(0, log_p_alpha_new + log_q_alpha_initial - log_p_alpha_initial - log_q_alpha_new)

		return (alpha_new, 1) if np.log(np.random.uniform()) < rho else (alpha[index], 0)

	print("------------------------------")
	print("job:", idx)
	print("seed:", seed)
	print("------------------------------")
	np.random.seed(seed)

	# Prior: P(alpha) ~ MVN(0,alpha_sigma), non-informative
	if n_covariate > 0:
		alpha_sigma = np.array([args.alpha_sigma] * n_covariate)
		alpha = np.zeros(n_covariate)
		alpha_running_chain = np.full([int(args.nIter / args.thin), n_covariate], np.nan)
		alpha_accp_rate = np.full([args.nIter, n_covariate], np.nan)
	else:
		alpha = np.array([0])

	# Prior: P(h2) ~ Beta (a,b)
	a, b = 1, 1
	h2 = args.h2_initial
	h2_running_chain = np.full(int(args.nIter / args.thin), np.nan)
	h2_grid = np.linspace(0, 1, args.n_h2_grid, endpoint=False) + 1 / (2 * args.n_h2_grid)

	beta = np.random.normal(scale=(h2 / n_connectivity) ** 0.5, size=n_connectivity)
	beta_running_chain = np.full([int(args.nIter / args.thin), n_connectivity], np.nan)
	psi = np.ones(n_connectivity)

	# MCMC
	mcmc_start = time.time()
	for iteration in range(args.nIter):
		print(iteration)
		time_flag = time.time()

		# draw beta
		w = np.random.normal(size=n_connectivity)
		quad = 0
		for i in range(len(block_size)):
			if args.adjBias:
				b_hat = b_hat_all[off_diag_row[i]] - lambda_off_diag[i] @ beta[off_diag_col[i]]
			else:
				b_hat = b_hat_all[off_diag_row[i]]

			sum2 = jax.numpy.array(lambda_diag[i] + np.diag((1 - h2) * psi.sum() / (psi[off_diag_row[i]] * n_subject * h2)))
			dinvt_chol = jax.numpy.linalg.cholesky(sum2)
			beta_temp = solve_triangular(dinvt_chol, b_hat, lower=True) + ((1 - h2) / n_subject) ** 0.5 * w[off_diag_row[i]]
			beta[off_diag_row[i]] = solve_triangular(dinvt_chol.T, beta_temp)
			quad += beta[off_diag_row[i]] @ lambda_diag[i] @ beta[off_diag_row[i]]

			if args.full_quad:
				quad += beta[off_diag_col[i]] @ lambda_off_diag[i].T @ beta[off_diag_row[i]]

		# draw h2 (GGS)
		log_w = log_p_h2(h2_grid, y2, beta, b_hat_all, quad, psi, a, b)
		log_w = log_w - np.max(log_w)
		h2 = h2_grid[np.random.choice(range(args.n_h2_grid), p=softmax(log_w))]
		print("current h2 draw:", h2)

		# draw alpha 1 by 1; use optimization within MH
		if n_covariate > 0:
			alpha_accp = np.full(n_covariate, np.nan)
			for annot in range(n_covariate):
				alpha[annot], alpha_accp[annot] = draw_alpha_log(annot, alpha, z, alpha_sigma, beta, h2, 1e-5)

			psi = np.exp(z @ alpha)
			print("current alpha draw:", alpha)

			alpha_accp_rate[iteration, :] = alpha_accp

		if (iteration + 1) % args.thin == 0:
			beta_running_chain[(idx_chain := int((iteration + 1) / args.thin) - 1), :] = beta
			h2_running_chain[idx_chain] = h2
			if n_covariate > 0:
				alpha_running_chain[idx_chain, :] = alpha

		print(time.time() - time_flag)
	mcmc_time = time.time() - mcmc_start
	# end of MCMC

	# Report results
	print("------------------------------")
	with open("mcmc_output/" + args.data_pickle.split(".")[0] + "__" + "_".join(map(str, args.idx_annot)) + "__" + str(idx) + ".pickle", "wb") as file:
		pickle.dump({"args": vars(args), "seed": seed, "time": mcmc_time, "alpha_running_chain": alpha_running_chain if n_covariate > 0 else None, "alpha_accp_rate": alpha_accp_rate if n_covariate > 0 else None, "h2_running_chain": h2_running_chain, "beta_running_chain": beta_running_chain, "beta_cover_true": np.nanmean(((beta_ci := np.nanpercentile(beta_running_chain, [2.5, 97.5], axis=0))[1, :] >= beta_true) & (beta_ci[0, :] <= beta_true)) if beta_true is not None else None}, file)
	return "job " + str(idx) + " finish"


with Pool() as pool:
	result = pool.starmap_async(cm_variance, zip(list(range(args.n_mcmc_chain)), np.random.choice(1000, args.n_mcmc_chain, False)))
	for item in result.get():
		print(item)
