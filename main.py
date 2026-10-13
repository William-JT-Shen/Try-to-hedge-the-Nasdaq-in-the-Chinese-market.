# ============================================================
# main.py
# ============================================================
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

import follow_the_leader
from backtest_replication import run_replication_backtest
from rl_short_replication import ShortIndexReplicationEnv, ActorCritic, ppo_update
from mpc_risk_control import (
    drawdown_gamma,
    solve_mpc_short_replication,
    backtest_short_replication,
)

# Windows 中文控制台默认编码为 GBK，无法输出 “⚠”“✓” 等字符，
# 会把 backtest_replication.py 里的提示打印变成 UnicodeEncodeError。
# 统一把标准输出切到 UTF-8（errors="replace" 兜底），保证脚本能跑完。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ------------------------------------------------------------
# 1. 读取 A 股候选指数
# ------------------------------------------------------------
folder = Path("history_board_concept_data")
files = sorted([f for f in folder.iterdir() if f.suffix == ".xlsx"])

price_dict = {}
for f in files:
    stem = f.stem
    if "_" not in stem:
        continue
    name, code = stem.rsplit("_", 1)
    df = pd.read_excel(f)
    if "date" not in df.columns or "close" not in df.columns:
        continue
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    if len(df) < 200:
        continue
    price_dict[stem] = df["close"]

prices = pd.DataFrame(price_dict).sort_index()
returns = prices.pct_change().dropna(how="all")

# ------------------------------------------------------------
# 2. 读取纳指目标指数，并构造目标收益 r0
# ------------------------------------------------------------
ixic = pd.read_excel("纳指历史行情数据.xlsx")
ixic["date"] = pd.to_datetime(ixic["date"])
ixic = ixic.sort_values("date").set_index("date")
target_price = ixic["close"]
target_ret = target_price.pct_change()

# 对齐 A 股候选指数与纳指的共同日期。
# 若共同日期太少，可改为按 A 股交易日 reindex 后 ffill，再计算收益。
common_idx = returns.index.intersection(target_ret.dropna().index)
R_df = returns.loc[common_idx].dropna(axis=1, how="any")
r0 = target_ret.loc[common_idx]

R = R_df.values.astype(float)
r0 = r0.values.astype(float)

print("候选指数数：", R.shape[1])
print("对齐后交易日数：", R.shape[0])

# ------------------------------------------------------------
# 3. Step 1：调用 follow_the_leader.py
# ------------------------------------------------------------
k_hat, ic2_values = follow_the_leader.estimate_num_factors_ic2(
    R, kmax=10, zero_tol=0.05
)
print("估计因子数 k_hat =", k_hat)

F_hat, Lambda_hat = follow_the_leader.pca_factors(R, k_hat)

leaders_per_factor, union = follow_the_leader.follow_the_leader_select(
    R, F_hat, kmax_res=10, max_iter=20
)
print("每个因子的 leader 数：", [len(x) for x in leaders_per_factor])
print("并集大小：", len(union))

# 目标指数因子暴露 b0 的估计
B_target = np.linalg.lstsq(F_hat, r0, rcond=None)[0]

K = 5
if len(union) > K:
    selected = follow_the_leader.greedy_select(Lambda_hat, B_target, K=K)
else:
    selected = union

w = follow_the_leader.solve_nonneg_weights(
    Lambda_hat[selected, :], B_target, gamma=1e-3
)

selected_names = R_df.columns[selected].tolist()
print("最终选出的指数：", selected_names)
print("非负权重：", w)
print("组合因子暴露：", Lambda_hat[selected, :].T @ w)
print("目标相反暴露：", -B_target)

# ------------------------------------------------------------
# 4. Step 2：BackTrader 回测入口
# ------------------------------------------------------------
# 注意：初稿回测代码使用 hedge_sign 控制“交易权重”和“评估符号”。
# 若仅评估非负权重合成空头的效果，可令 hedge_sign=-1；
# 若真实交易不允许做空，则需要用反向 ETF、期权或场外衍生品承接。
leader_dfs = []
for name in selected_names:
    df = pd.DataFrame({"close": prices[name].loc[common_idx]})
    leader_dfs.append(df)

target_df = pd.DataFrame({"close": target_price.loc[common_idx]})

target_weights = {i: float(w[i]) for i in range(len(selected))}

tracking = run_replication_backtest(
    leader_dfs=leader_dfs,
    target_df=target_df,
    target_weights=target_weights,
    rebalance_days=21,
    window=21,
    hedge_sign=-1,      # 评估空头暴露时对组合收益取反
    initial_cash=1_000_000,
)
print(tracking["annualized_rte"], tracking["mean_corr"])

# ------------------------------------------------------------
# 5. Step 3：PPO 动态调仓入口
# ------------------------------------------------------------
env = ShortIndexReplicationEnv(
    returns=R,
    target_returns=r0,
    window=20,
    rebalance=5,
    lambda_turn=0.01,
    lambda_risk=0.1,
    dw_max=0.1,
)

state = env.reset()
state_dim = state.shape[0]
action_dim = env.N

policy = ActorCritic(state_dim, action_dim, hidden=128)
optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

# 最小训练循环示例。实际使用时建议增加 episode 数、验证集和早停。
num_episodes = 50
for ep in range(num_episodes):
    state = env.reset()
    states, actions, logps, rewards, dones = [], [], [], [], []
    done = False
    while not done:
        action, logp, value = policy.act(state)
        next_state, reward, done, _ = env.step(action)
        states.append(state)
        actions.append(action)
        logps.append(logp)
        rewards.append(reward)
        dones.append(float(done))
        state = next_state

    ppo_update(
        policy,
        optimizer,
        states=np.array(states),
        actions=np.array(actions),
        old_logps=np.array(logps),
        rewards=np.array(rewards),
        dones=np.array(dones),
        gamma=0.99,
        lam=0.95,
        clip_eps=0.2,
        epochs=10,
        batch_size=64,
    )
    if (ep + 1) % 10 == 0:
        print(f"PPO episode {ep + 1}/{num_episodes} 完成")

# ------------------------------------------------------------
# 6. Step 4：MPC 风险控制入口
# ------------------------------------------------------------
# 这里用简单历史均值作为预测，仅用于跑通接口。
# 实际使用时请替换为滚动均值、ARIMA、机器学习或因子模型预测。
H = 15
# 注意：Lambda_hat[selected, :] 只有被选中的那几个指数，
# 因此预测收益、协方差和实际收益都必须只保留 selected 这些列，
# 否则 MPC 里 B^T @ w_risk 会出现维度不匹配。
R_sel = R[:, selected]
R_hat_all = np.repeat(R_sel[:, None, :], H, axis=1)
r0_hat_all = np.repeat(r0[:, None], H, axis=1)
Sigma_hat = np.cov(R_sel.T)
Sigma_hat_all = np.repeat(Sigma_hat[None, :, :], len(R_sel), axis=0)

weights_mpc, drawdowns_mpc = backtest_short_replication(
    R_actual=R_sel,
    r0_actual=r0,
    R_hat_all=R_hat_all,
    r0_hat_all=r0_hat_all,
    B=Lambda_hat[selected, :],
    b0=B_target,
    Sigma_hat_all=Sigma_hat_all,
    gamma0=5.0,
    D_max=0.10,
    H=H,
    kappa=0.004,
    rho=0.0005,
    w_max=0.4,
)

print("MPC 权重形状：", weights_mpc.shape)
print("MPC 最大回撤：", np.nanmax(drawdowns_mpc))