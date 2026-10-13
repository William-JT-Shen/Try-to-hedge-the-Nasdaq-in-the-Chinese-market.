import numpy as np
import pandas as pd
import backtrader as bt
from collections import deque


# ============================================================
# 1. 自定义分析器：计算滚动跟踪误差与滚动相关系数
# ============================================================
class TrackingErrorAnalyzer(bt.Analyzer):
    """
    在每个 bar 上记录组合收益率和目标指数收益率，
    回测结束后计算滚动跟踪误差和滚动相关系数。
    """
    params = (
        ('target_data', None),   # 目标指数的数据馈送
        ('window', 21),          # 滚动窗口长度，默认 21 个交易日
        ('risk_free', 0.0),      # 日度无风险利率，默认 0
        ('hedge_sign', -1),      # +1 表示多头复制，-1 表示空头暴露
    )

    def __init__(self):
        self.portfolio_returns = []
        self.target_returns = []
        self.dates = []

    def next(self):
        dt = self.strategy.datas[0].datetime.date(0)
        self.dates.append(dt)

        # 组合收益率：从 broker 读取每日组合价值变化
        if len(self.strategy) > 0:
            port_value = self.strategy.broker.getvalue()
            self.portfolio_returns.append(port_value)
        else:
            self.portfolio_returns.append(self.strategy.broker.startingcash)

        # 目标指数收益率
        target_data = self.p.target_data
        if target_data is not None and len(target_data) > 1:
            tret = target_data.close[0] / target_data.close[-1] - 1.0
        else:
            tret = 0.0
        self.target_returns.append(tret)

    def stop(self):
        # 将组合价值转换为日度收益率
        pv = np.array(self.portfolio_returns, dtype=float)
        pr = np.diff(pv) / pv[:-1]
        pr = np.concatenate([[0.0], pr])  # 首日收益率为 0

        tr = np.array(self.target_returns, dtype=float)

        # 若为空头暴露模式，组合收益取反
        if self.p.hedge_sign < 0:
            pr = -pr

        L = self.p.window
        n = len(pr)
        rte_series = np.full(n, np.nan)
        corr_series = np.full(n, np.nan)

        for t in range(L - 1, n):
            window_pr = pr[t - L + 1: t + 1]
            window_tr = tr[t - L + 1: t + 1]

            diff = window_pr - window_tr
            rte_series[t] = np.std(diff, ddof=1)

            if np.std(window_pr) > 1e-12 and np.std(window_tr) > 1e-12:
                corr_series[t] = np.corrcoef(window_pr, window_tr)[0, 1]
            else:
                corr_series[t] = 0.0

        # 存为分析结果
        self.rets['rte_series'] = rte_series
        self.rets['corr_series'] = corr_series
        self.rets['dates'] = self.dates
        self.rets['mean_rte'] = np.nanmean(rte_series)
        self.rets['mean_corr'] = np.nanmean(corr_series)
        self.rets['annualized_rte'] = np.nanmean(rte_series) * np.sqrt(252)


# ============================================================
# 2. 定期再平衡策略
# ============================================================
class IndexReplicationStrategy(bt.Strategy):
    """
    按第一阶段输出的目标权重进行定期再平衡。
    """
    params = (
        ('target_weights', None),    # dict: {data_index: weight}
        ('rebalance_days', 21),      # 每 21 个交易日再平衡一次
        ('hedge_sign', -1),          # -1 表示空头暴露，+1 表示多头复制
    )

    def __init__(self):
        self.day_count = 0

    def next(self):
        self.day_count += 1

        # 仅在再平衡日执行
        if self.day_count % self.p.rebalance_days != 0:
            return

        # 按目标权重调整持仓
        for data_idx, weight in self.p.target_weights.items():
            # 空头暴露模式下权重取负
            effective_weight = weight * self.p.hedge_sign
            self.order_target_percent(
                data=self.datas[data_idx],
                target=effective_weight
            )


# ============================================================
# 3. 回测主流程
# ============================================================
def run_replication_backtest(
    leader_dfs,           # list of DataFrame，每个 leader 指数的历史价格
    target_df,            # DataFrame，目标指数的历史价格
    target_weights,       # dict: {leader_index: weight}，来自第一阶段
    rebalance_days=21,
    window=21,
    hedge_sign=-1,
    initial_cash=1_000_000
):
    """
    执行指数复制回测，返回跟踪误差和相关系数时间序列。
    """
    cerebro = bt.Cerebro()

    # --- 添加 leader 数据 ---
    for i, df in enumerate(leader_dfs):
        # 注意：BackTrader 的市价单默认以次日 open 成交。
        # 若 open 缺失（open=None），成交价会是 NaN，一手交易后 broker 净值即变成 NaN，
        # 跟踪误差与相关系数随之失真。这里用收盘价充当 open/high/low；
        # 若要更真实，可在 df 中补入真实 OHLC 并改成 open='open', high='high', low='low'。
        data = bt.feeds.PandasData(
            dataname=df,
            datetime=None,        # 使用 DataFrame 的索引作为日期
            open='close', high='close', low='close', close='close',
            volume=None, openinterest=None
        )
        cerebro.adddata(data, name=f'leader_{i}')

    # --- 添加目标指数数据 ---
    target_data = bt.feeds.PandasData(
        dataname=target_df,
        datetime=None,
        open='close', high='close', low='close', close='close',
        volume=None, openinterest=None
    )
    cerebro.adddata(target_data, name='target')

    # --- 构建目标权重字典：leader 在 self.datas 中的索引 ---
    weight_dict = {}
    for i in range(len(leader_dfs)):
        weight_dict[i] = target_weights.get(i, 0.0)

    # --- 添加策略 ---
    cerebro.addstrategy(
        IndexReplicationStrategy,
        target_weights=weight_dict,
        rebalance_days=rebalance_days,
        hedge_sign=hedge_sign
    )

    # --- 添加分析器 ---
    cerebro.addanalyzer(
        TrackingErrorAnalyzer,
        target_data=target_data,
        window=window,
        hedge_sign=hedge_sign,
        _name='tracking'
    )

    # --- 设置资金 ---
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=0.001)  # 0.1% 交易成本

    # --- 运行回测 ---
    results = cerebro.run()
    strat = results[0]
    tracking = strat.analyzers.tracking.get_analysis()

    print("=" * 50)
    print(f"回测完成")
    print(f"平均滚动跟踪误差 (日度): {tracking['mean_rte']:.6f}")
    print(f"年化滚动跟踪误差:       {tracking['annualized_rte']:.4%}")
    print(f"平均滚动相关系数:       {tracking['mean_corr']:.4f}")

    if tracking['annualized_rte'] > 0.02:
        print("\n⚠ 年化跟踪误差超过 2%，建议：")
        print("  1. 加入稀疏指数追踪方法进行权重优化")
        print("  2. 增加备选指数数量")
    else:
        print("\n✓ 跟踪误差在可接受范围内。")

    return tracking