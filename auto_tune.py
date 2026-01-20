import subprocess
import re
import time
import itertools
import pandas as pd
import os

# ================= 基础配置 =================
PYTHON_PATH = r"D:\learn\Anaconda\python.exe"
SCRIPT_NAME = "run_seq.py"

# 这里填你的模型名字 (MyRec) 和数据集
BASE_CMD = "--model=ACLSRec --dataset=Amazon_Beauty"

# 目标指标 (要和日志输出一致)
TARGET_METRIC = "NDCG@10"

# 固定随机种子 (保证调参可复现)
FIXED_SEED = 2024

# ================= 🟢 参数网格 (在这里修改你想尝试的值) =================
param_grid = {
    # 学习率
    "learning_rate": [0.001],

    # 自定义参数 lmd (对应代码 self.lmd)x
    "lmd": [0.1, 0.01],

    # 自定义参数 gamma (对应代码 self.gamma)
    "gamma": [0.001, 0.0001, 0.01],

    # 自定义参数 tau (对应 InfoNCE 温度系数)
    "tau": [1],

    # 自定义参数 sim_lambda (对应原本硬编码的 50)
    "sim_lambda": [1, 2, 0.1, 0.5],

    "sim": ['cos']
}


# ======================================================================

def get_combinations(grid):
    """生成笛卡尔积组合"""
    keys = grid.keys()
    values = grid.values()
    combinations = []
    for combo in itertools.product(*values):
        combinations.append(dict(zip(keys, combo)))
    return combinations


def run_experiment(params, run_id, total_runs):
    """运行单次实验"""
    print(f"\n[进度 {run_id}/{total_runs}] 测试参数: {params}")

    # 构造命令行参数
    cmd_args = f"{BASE_CMD} --seed={FIXED_SEED}"
    for key, value in params.items():
        cmd_args += f" --{key}={value}"

    cmd = f'"{PYTHON_PATH}" {SCRIPT_NAME} {cmd_args}'

    start_time = time.time()
    try:
        # 运行并捕获输出
        result = subprocess.run(cmd, shell=True, capture_output=True, encoding='utf-8')
        output = result.stdout + result.stderr

        # 使用正则提取 test result
        match = re.search(r"test result:.*?(\{.*?\})", output, re.DOTALL | re.IGNORECASE)
        if match:
            result_dict_str = match.group(1)
            # 提取 NDCG@10
            metric_pattern = rf"'{TARGET_METRIC}':\s*(\d+\.\d+)"
            score_match = re.search(metric_pattern, result_dict_str, re.IGNORECASE)

            if score_match:
                score = float(score_match.group(1))
                elapsed = time.time() - start_time
                print(f"✅ 成功! {TARGET_METRIC}: {score:.4f} (耗时: {elapsed:.0f}s)")
                return score
            else:
                print(f"⚠️ 没找到 {TARGET_METRIC}")
        else:
            # 如果没找到结果，打印最后几行日志看报错
            print("❌ 运行失败或未进入测试阶段")
            print(output[-300:])

    except Exception as e:
        print(f"❌ 系统错误: {e}")

    return 0.0


def main():
    # 1. 准备参数组合
    combos = get_combinations(param_grid)
    total_runs = len(combos)

    print(f"🚀 开始 MyRec 自动调优")
    print(f"📊 预计总运行次数: {total_runs}")
    print(f"🎯 优化目标: {TARGET_METRIC}")
    print("=" * 50)

    results = []

    # 2. 循环运行
    for i, params in enumerate(combos):
        score = run_experiment(params, i + 1, total_runs)

        # 记录结果
        res_entry = params.copy()
        res_entry['score'] = score
        results.append(res_entry)

    # 3. 结果处理
    if not results:
        print("❌ 没有获得任何有效结果。")
        return

    df = pd.DataFrame(results)
    df = df.sort_values(by='score', ascending=False)  # 按分数降序

    # 获取最好的一条
    best_run = df.iloc[0]
    best_score = best_run['score']
    best_params = best_run.drop('score').to_dict()

    # 4. 输出并保存文件
    print("\n" + "=" * 50)
    print(f"🏆 最佳结果: {TARGET_METRIC} = {best_score:.4f}")
    print(f"🔑 最佳参数: {best_params}")
    print("=" * 50)

    # 保存 csv 表格
    csv_file = "myrec_tuning_log.csv"
    df.to_csv(csv_file, index=False)
    print(f"📄 完整日志已保存至: {csv_file}")

    # 保存最好的结果到 txt (方便直接查看)
    txt_file = "best_result.txt"
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write("========== MyRec Best Result ==========\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Best Score ({TARGET_METRIC}): {best_score:.6f}\n\n")
        f.write("Best Parameters:\n")
        for k, v in best_params.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nReproduce Command:\n")
        cmd_str = f"python {SCRIPT_NAME} {BASE_CMD} --seed={FIXED_SEED}"
        for k, v in best_params.items():
            cmd_str += f" --{k}={v}"
        f.write(cmd_str)

    print(f"💾 最佳配置已单独保存至: {txt_file}")


if __name__ == "__main__":
    main()
