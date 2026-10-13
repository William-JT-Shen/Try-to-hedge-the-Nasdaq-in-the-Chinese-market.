import numpy as np
import cvxpy as cp


def drawdown_gamma(gamma0, D, D_max, eps=1e-6):
    """
    按回撤调整风险厌恶。
    gamma0: 初始风险厌恶
    D: 当前回撤
    D_max: 最大可接受回撤
    eps: 防止分母为零的小常数
    """
    return gamma0 * D_max / max(D_max - D, eps)


def solve_mpc_short_replication(
    R_hat, r0_hat, B, b0, Sigma_hat,
    w_prev, gamma, kappa=0.004, rho=0.0005,
    w_max=0.4, H=15
):
    """
    R_hat: (H, n) 可投资指数预测收益
    r0_hat: (H,) 目标指数预测收益
    B: (n, k) 因子载荷矩阵
    b0: (k,) 目标指数因子载荷
    Sigma_hat: (n, n) 预测协方差矩阵
    w_prev: (n+1,) 上一期权重，最后一位为现金
    gamma: 当前风险厌恶
    kappa: 交易成本惩罚标量
    rho: 持有成本/正则化标量
    w_max: 单指数权重上限
    H: 规划步数
    返回：w_next (n+1,)
    """
    H = min(H, R_hat.shape[0])
    n = R_hat.shape[1]
    k = B.shape[1]

    w = cp.Variable((H, n + 1))
    obj = 0
    constraints = []

    for h in range(H):
        w_risk = w[h, :n]
        w_cash = w[h, n]

        # 1) 空头复制跟踪误差：组合收益 + 目标收益 ≈ 0
        err = R_hat[h, :] @ w_risk + r0_hat[h]
        obj += cp.sum_squares(err)

        # 2) 因子暴露匹配：B^T w_risk + b0 ≈ 0
        obj += 0.5 * cp.sum_squares(B.T @ w_risk + b0)

        # 3) 风险惩罚
        obj += gamma * cp.quad_form(w_risk, Sigma_hat)

        # 4) 交易成本
        if h == 0:
            dw = w[h, :] - w_prev
        else:
            dw = w[h, :] - w[h - 1, :]
        obj += kappa * cp.norm1(dw)

        # 5) 持有成本/正则化
        obj += rho * cp.sum_squares(w_risk)

        # 6) 权重约束：不能做空、单指数上限、权重和为 1
        constraints += [
            cp.sum(w[h, :]) == 1,
            w[h, :n] >= 0,
            w[h, :n] <= w_max,
            w[h, n] >= 0,
        ]

    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.ECOS, verbose=False)

    if w.value is None:
        raise RuntimeError("优化未求解成功")

    return w.value[0, :]


def backtest_short_replication(
    R_actual, r0_actual, R_hat_all, r0_hat_all,
    B, b0, Sigma_hat_all,
    gamma0=5.0, D_max=0.10, H=15,
    kappa=0.004, rho=0.0005, w_max=0.4
):
    """
    日频回撤控制循环示意。
    R_actual: (T, n) 实际可投资指数收益
    r0_actual: (T,) 实际目标指数收益
    R_hat_all: (T, H, n) 每日预测的可投资指数收益
    r0_hat_all: (T, H) 每日预测的目标指数收益
    B: (n, k) 因子载荷
    b0: (k,) 目标因子载荷
    Sigma_hat_all: (T, n, n) 每日预测协方差
    """
    T, n = R_actual.shape
    w_prev = np.zeros(n + 1)
    w_prev[-1] = 1.0  # 初始全现金

    V = 1.0
    M = 1.0

    weights = []
    drawdowns = []

    for t in range(T - H):
        D = 1.0 - V / M
        gamma = drawdown_gamma(gamma0, D, D_max)

        w_next = solve_mpc_short_replication(
            R_hat_all[t], r0_hat_all[t], B, b0, Sigma_hat_all[t],
            w_prev, gamma, kappa, rho, w_max, H
        )

        # 执行第一笔，计算组合收益（现金收益设为 0）
        r_port = R_actual[t, :] @ w_next[:n]

        # 交易成本近似
        turnover = np.sum(np.abs(w_next - w_prev))
        cost = kappa * turnover

        V = V * (1.0 + r_port) - cost
        M = max(M, V)

        w_prev = w_next
        weights.append(w_next)
        drawdowns.append(1.0 - V / M)

    return np.array(weights), np.array(drawdowns)