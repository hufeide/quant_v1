在项目根目录 /home/administator/workspace/quant_v1 下执行：

# 看有哪些实验模块
python3 src/ablation/run.py --help

# 跑单个模块
python3 src/ablation/run.py universe     # universe 过滤模块（6项）
python3 src/ablation/run.py scoring      # 打分权重模块（6项）
python3 src/ablation/run.py risk         # 卖出风控模块（7项）
python3 src/ablation/run.py sentiment   # 情绪闸门模块（3项）
python3 src/ablation/run.py position     # 仓位约束模块（4项）
python3 src/ablation/run.py buy_guards   # 买入保护模块（4项）

# 一次跑多个模块
python3 src/ablation/run.py universe scoring risk

# 跑全部 30 项（耗时长，建议后台跑）
nohup python3 src/ablation/run.py all > /tmp/ablation_all.log 2>&1 &

结果存放在：
backtest/results/ablation/ablation_comparison.csv   # 对比总表
backtest/results/ablation/ablation_<实验名>/        # 每次实验的净值和成交明细