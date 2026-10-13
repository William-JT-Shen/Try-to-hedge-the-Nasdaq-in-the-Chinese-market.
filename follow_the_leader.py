import numpy as np
from numpy.linalg import svd
from scipy.optimize import minimize


def estimate_num_factors_ic2(R, kmax=30, zero_tol=0.05):
    """
    用 Bai-Ng IC2 估计因子数。
    R: T x N 矩阵，每列一个候选工具的收益。
    kmax: 最大候选因子数。
    zero_tol: 若第一特征值占比低于该阈值，则视为 0 个因子。
    """
    T, N = R.shape
    Rc = R - R.mean(axis=0, keepdims=True)
    U, S, Vt = svd(Rc, full_matrices=False)

    total_var = np.sum(S ** 2)
    if total_var <= 1e-12:
        return 0, np.array([])

    first_ratio = (S[0] ** 2) / total_var
    if first_ratio < zero_tol:
        return 0, np.array([])

    kmax = min(kmax, len(S))
    ic2_values = []

    for k in range(1, kmax + 1):
        recon = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        E = Rc - recon
        sigma2 = np.mean(E ** 2)
        penalty = k * (N + T) / (N * T) * np.log(min(N, T))
        ic2_values.append(np.log(sigma2 + 1e-12) + penalty)

    k_hat = int(np.argmin(ic2_values)) + 1
    return k_hat, np.array(ic2_values)


def pca_factors(R, k):
    """
    用 PCA 估计因子和载荷。
    R: T x N 收益矩阵。
    k: 因子数。
    返回 F_hat: T x k, Lambda_hat: N x k。
    """
    T, N = R.shape
    Rc = R - R.mean(axis=0, keepdims=True)
    U, S, Vt = svd(Rc, full_matrices=False)

    F_hat = np.sqrt(T) * U[:, :k]
    Lambda_hat = (Vt[:k, :].T * S[:k]) / np.sqrt(T)
    return F_hat, Lambda_hat


def _corr_with_vector(X, y):
    """
    计算 X 的每一列与 y 的相关系数。
    X: T x N, y: T。
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    num = Xc.T @ yc
    den = np.sqrt((Xc ** 2).sum(axis=0) * (yc ** 2).sum())
    return num / (den + 1e-12)


def follow_the_leader_select(R, F_hat, kmax_res=10, max_iter=20):
    """
    Follow-the-Leader 选指数。
    R: T x N 候选工具收益。
    F_hat: T x k 估计因子。
    返回 leaders_per_factor, union。
    """
    T, N = R.shape
    k = F_hat.shape[1]
    leaders_per_factor = []

    for j in range(k):
        fj = F_hat[:, j]
        corr = np.abs(_corr_with_vector(R, fj))
        order = np.argsort(-corr)
        P = [int(order[0])]

        for _ in range(max_iter):
            # 构造检验回归：R[:, P] 与除 fj 外的其他因子
            X_parts = [R[:, P]]
            other_factors = [F_hat[:, l] for l in range(k) if l != j]
            if other_factors:
                X_parts.append(np.column_stack(other_factors))
            X = np.column_stack(X_parts)
            X = np.column_stack([np.ones(T), X])  # 截距

            # 对每个候选工具回归，得到残差矩阵
            E = np.zeros((T, N))
            for i in range(N):
                beta, *_ = np.linalg.lstsq(X, R[:, i], rcond=None)
                E[:, i] = R[:, i] - X @ beta

            kmax_here = min(kmax_res, E.shape[1] - 1, E.shape[0] - 1)
            k_res, _ = estimate_num_factors_ic2(
                E, kmax=max(1, kmax_here), zero_tol=0.05
            )

            if k_res == 0:
                break

            # 辅助回归：fj 对当前 P 回归，取残差
            Xp = np.column_stack([np.ones(T), R[:, P]])
            beta, *_ = np.linalg.lstsq(Xp, fj, rcond=None)
            u = fj - Xp @ beta

            # 选与残差 u 相关性最高、且尚未在 P 中的工具
            corr_u = np.abs(_corr_with_vector(R, u))
            corr_u[P] = -np.inf
            new_idx = int(np.argmax(corr_u))
            P.append(new_idx)

        leaders_per_factor.append(sorted(set(P)))

    union = sorted(set().union(*leaders_per_factor)) if leaders_per_factor else []
    return leaders_per_factor, union


def solve_nonneg_weights(B, b_target, gamma=1e-3):
    """
    在 w >= 0, sum(w)=1 约束下，求 B^T w ≈ -b_target。
    B: m x k 候选组合的因子载荷。
    b_target: k 目标因子暴露。
    返回 w: m。
    """
    m = B.shape[0]

    def objective(w):
        factor_exposure = B.T @ w
        return np.sum((factor_exposure + b_target) ** 2) + gamma * np.sum(w ** 2)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * m
    w0 = np.ones(m) / m

    res = minimize(
        objective, w0, bounds=bounds, constraints=constraints, method="SLSQP"
    )
    return res.x


def greedy_select(B, b_target, K=5):
    """
    若 Follow-the-Leader 并集超过 K，用前向贪心压到 K 个。
    B: N x k 全部候选工具的因子载荷。
    b_target: k 目标因子暴露。
    K: 预算，通常 3 到 5。
    """
    N = B.shape[0]
    selected = []
    remaining = list(range(N))

    for _ in range(min(K, N)):
        best_i, best_obj = None, np.inf
        for i in remaining:
            S = selected + [i]
            w = solve_nonneg_weights(B[S, :], b_target)
            exposure = B[S, :].T @ w
            obj = np.sum((exposure + b_target) ** 2)
            if obj < best_obj:
                best_obj = obj
                best_i = i
        selected.append(best_i)
        remaining.remove(best_i)

    return selected