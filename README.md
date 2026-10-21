# A股量化工具

用 AKShare 免费行情数据做两件事：**选股筛选** 和 **盘中实时信号提醒**。命令行运行，结果输出为终端表格、CSV 和带K线图的 HTML 报告。

> ⚠ 仅供学习研究，不构成投资建议。

## 安装

需要 Python 3.9 及以上（macOS 自带的 python3 即可）。

```bash
cd ~/claude-workspace/stock
python3 -m venv .venv && source .venv/bin/activate
pip3 install -r requirements.txt
```

先用离线模拟数据试一遍（不需要联网）：

```bash
python3 main.py --demo screen -s all --report
python3 main.py --demo monitor --once
```

## 常用命令

| 目的 | 命令 |
|---|---|
| 查看内置策略及参数 | `python3 main.py strategies` |
| 沪深300 今日 MACD 金叉 | `python3 main.py screen -s macd -u hs300` |
| 全市场活跃股近 3 日的均线/MACD 买点，并生成报告 | `python3 main.py screen -s ma_cross,macd -u all -l 3 --report` |
| 三个技术策略同时扫描（看共振数） | `python3 main.py screen -s all -u zz500 -l 2 --report` |
| 卖出信号扫描自选股 | `python3 main.py screen -s all -u watchlist -d sell` |
| 多因子打分 Top 30 | `python3 main.py screen -s multi_factor --top 30 --report` |
| 盘中监控自选股 | `python3 main.py monitor` |
| 临时监控几只股票，30 秒刷新 | `python3 main.py monitor --codes 600519,300750 -i 30` |
| 个股K线 + 指标图 | `python3 main.py chart 600519 000858` |
| 预下载日线缓存 | `python3 main.py update -u hs300` |

**股票池 `-u`**：`hs300`、`zz500`、`zz1000`、`sz50`、`kc50`、`cyb`、`all`（过滤后按成交额取前 N，默认 300）、`watchlist`（自选股）、`index:000905`（任意中证指数代码），或者直接写 `600519,000858`。

结果保存在 `output/` 目录：`screen_*.csv`（Excel 可以直接打开）、`report_*.html`（报告会自动在浏览器打开，加 `--no-open` 可以关掉）、`alerts.log`（提醒记录）。

## 内置策略

| 名称 | 逻辑 | 主要参数 |
|---|---|---|
| `ma_cross` 双均线 | MA5 上穿 MA20 买入（可要求放量），下穿卖出 | fast / slow / volume_confirm |
| `macd` MACD | DIF 上穿 DEA 为金叉、下穿为死叉；可以只认零轴上方的金叉 | fast / slow / signal / above_zero |
| `mean_reversion` 布林带+RSI | 跌破下轨且 RSI<30 之后重新站回下轨时买入；突破上轨且 RSI>70 之后跌回上轨时卖出 | boll_period / boll_std / rsi_low / rsi_high |
| `multi_factor` 多因子 | 全市场快照中取 60 日动量、年初至今涨幅、PE、PB、换手率、市值，先 MAD 去极值、再 z-score 标准化，按权重加总后排名 | factors 的权重与方向、max_pe |

所有参数都在 `config.yaml` 里改。

## 尾盘选股（每天 14:30）

每天输出 10 只候选股。两个模型同时计算、分别保存、分别复盘：

| 模型 | 做法 |
|---|---|
| `rule` 反转规则（默认主模型） | 当日涨幅、K线实体、偏离20日线、20日涨幅、收盘相对均价、量比、换手、波动率、RSI6 这 9 项，在股票池内越弱越冷清，得分越高 |
| `pattern` 尾盘回落型 | 当天在股票池里属于小跌（涨幅 20%–60% 分位）、收盘在当天低位（最低的 40%）、换手不在最高的 40%、波动不在最高的 20%，收盘位置越低排名越前。由 2345 条量价规则分段检验后总结而来 |
| `ml` 每日训练模型 | 每次运行用最近 120 个交易日重新训练 HistGradientBoosting，输入 20 个量价特征（横截面排名），预测次日收益在池内的排名；训练只用标签已经确定的数据 |

两个模型都会剔除涨跌停股票和成交额最低的 30%。

```bash
python3 main.py tail                # 复盘 + 今日选股（以反转规则为主，ML 作对照）
python3 main.py tail -m ml          # 以 ML 模型为主
python3 main.py tail -m pattern     # 以尾盘回落型为主
python3 main.py tail --backtest     # 两个模型的历史回测对比（ML 为滚动训练）
python3 main.py tail --review       # 只看历史复盘
python3 main.py tail -u hs300       # 换股票池
```

在中证500 真实数据上的回测：两个模型都只接近股票池平均，ML 模型比反转规则还差一些。它们都不能可靠预测次日涨跌，请当作观察名单。每天的复盘会积累两个模型的真实胜率，可以据此比较。

回测有两处近似：用收盘价代替 14:30 的价格；使用当前的指数成分股，存在幸存者偏差。

建议每个交易日 14:30–14:50 之间运行一次。

## 实时监控

`monitor` 在交易时段（9:30–11:30、13:00–15:00）按设定间隔刷新自选股：

1. **价格提醒**：在 `watchlist` 里给股票加 `above` / `below` / `pct_up` / `pct_down`
2. **策略信号**：把最新价拼成当天的K线，再跑一遍策略，出现金叉/死叉等信号时提醒

提醒会发到终端（买入红色、卖出绿色）和 macOS 系统通知，也可以推送到手机：在 `config.yaml` 的 `monitor.notify.webhook` 中填好 `type`（`wecom` / `dingtalk` / `feishu` / `serverchan`）和 `url`。如果用钉钉机器人，安全设置里的关键词请设成“信号”或“提醒”。

同一只股票、同一个信号，一天内只提醒一次（重启后也不会重复）。盘中信号基于还没收盘的K线，收盘前可能消失。

## 目录结构

```
main.py              命令行入口
config.yaml          全部配置（股票池过滤、策略参数、自选股、通知）
quant/
  data.py            AKShare 数据获取 + 本地缓存；MockProvider 离线模拟数据
  indicators.py      MA / EMA / MACD / RSI / 布林带 / ATR
  strategies/        策略（technical.py 技术面，multi_factor.py 多因子）
  screener.py        选股扫描
  monitor.py         实时监控
  notifier.py        通知通道
  report.py          HTML 图表报告（Plotly，红涨绿跌）
  tail.py            尾盘选股：打分、回测、复盘
scripts/
  install_schedule.sh  macOS 定时任务安装（launchd）
tests/               单元测试：python -m pytest -q
```

## 自己写策略

在 `quant/strategies/technical.py` 中继承 `Strategy`，实现 `compute(df)`，输出 `signal` 列（1 买 / -1 卖 / 0）和 `reason` 列，然后在 `strategies/__init__.py` 的 `TECHNICAL` 里注册。`screen`、`monitor`、`chart` 都能直接用上。

```python
class MyStrategy(Strategy):
    name, label = "my", "放量突破20日新高"
    def compute(self, df):
        hh = df["high"].rolling(20).max().shift()
        up = (df["close"] > hh) & (df["volume"] > 2 * df["volume"].rolling(20).mean())
        df["signal"] = up.astype(int)
        df["reason"] = up.map({True: "放量创20日新高", False: ""})
        return df
```

## 扩展数据：情绪、资金、分时、竞价、消息

`tail` 跑完三个模型后，会自动做第 3 步“综合评估”。加 `--fast` 可以跳过。

| 类别 | 用到的数据 | 来源（自动切换） |
|---|---|---|
| 尾盘量价 | 14:30 后涨跌、尾盘成交占比、是否站稳分时均价线、量价相关性、急拉/跳水形态 | 1 分钟K线：东方财富 → 新浪 |
| 大单 | 单笔 ≥20 万元的主动买入/卖出占比、大单净额 | 逐笔成交：东方财富 → 腾讯 |
| 活跃度 | 量比、换手、成交额/5日均、60 日高低位置 | 日线 |
| 资金 | 主力（超大单+大单）净流入、净占比 | 东方财富 |
| 龙虎榜 | 机构/游资/北向净额、买一是否一家独大、机构游资是否共振 | 东方财富 → 新浪（新浪无席位明细） |
| 市场情绪 | 涨停/跌停/炸板家数、封板率、连板高度、晋级率、昨日涨停今日溢价 | 东方财富涨停池；拿不到时用全市场快照估算 |
| 指数与期货 | 中证500 尾盘涨跌、IF/IC/IM 升贴水、行业板块强弱 | 新浪 / 东方财富 |
| 消息事件 | 当日风险公告（立案、问询、减持等）、业绩预亏、30 日内解禁、近期减持 | 东方财富 |
| 次日竞价 | 高开/低开、竞价额占昨日成交额、昨日炸板股弱转强、跌停股是否继续被砸 | `python3 main.py auction` |

- **综合精选**：三个模型的候选股合并后，按上面的信号加减分。有风险公告、业绩预亏、减持，或放量尾盘跳水的股票会被剔除。
- **情绪温度**：情绪为“冰点/低迷”时，程序会提示观望。
- **权重未经检验**：加减分的权重目前是经验值。根据本项目对日线的检验，“尾盘急拉、收在高位”的股票次日反而偏弱，所以尾盘急拉算扣分项。
- **数据归档**：所有原始数据每天存到 `data/archive/日期/`，积累几周后可以检验哪些信号真的有效，再调整权重。
- **数据源限制**：东方财富不可用时，资金流、涨停池、龙虎榜席位、公告事件会缺失，程序会在“数据来源”一行里注明。

## 数据源

日线按 `config.yaml` 中 `data.sources` 的顺序依次尝试：东方财富 → 腾讯 → 新浪。前一个失败时自动换下一个，各源的数据都统一成相同的口径。

`python3 main.py check` 可以测试哪个数据源能用。

## 常见问题

- **下载慢或报错**：东方财富接口有频率限制，可以把 `data.max_workers` 调到 2。日线收盘后只下载一次，之后读 `data/cache`。
- **`universe=all` 扫描慢**：第一次需要下载约 300 只股票的日线，之后走缓存。
- **AKShare 接口变了**：`pip install -U akshare` 升级。


python3 main.py tail       # 14:30 后：三个模型 + 综合评估，输出“综合精选”10 只和情绪温度
python3 main.py market     # 只看市场情绪
python3 main.py auction    # 第二天 9:25 后：验证昨天的选股
python3 main.py tail --fast  # 想快一点，跳过扩展数据