import subprocess
import re
import time
import itertools
import pandas as pd
import os

PYTHON_PATH = r"D:\learn\Anaconda\python.exe"
SCRIPT_NAME = "run_seq.py"

BASE_CMD = "--model=ACLSRec --dataset=Amazon_Beauty"

TARGET_METRIC = "NDCG@10"

FIXED_SEED = 2024

param_grid = {
    "learning_rate": [0.001],

    "lmd": [0.1, 0.01],

    "gamma": [0.001, 0.0001, 0.01],

    "tau": [1],

    "sim_lambda": [1, 2, 0.1, 0.5],

    "sim": ['cos']
}


def get_combinations(grid):
    keys = grid.keys()
    values = grid.values()
    combinations = []
    for combo in itertools.product(*values):
        combinations.append(dict(zip(keys, combo)))
    return combinations


def run_experiment(params, run_id, total_runs):
    print(f"\n[{run_id}/{total_runs}] params: {params}")

    cmd_args = f"{BASE_CMD} --seed={FIXED_SEED}"
    for key, value in params.items():
        cmd_args += f" --{key}={value}"

    cmd = f'"{PYTHON_PATH}" {SCRIPT_NAME} {cmd_args}'

    start_time = time.time()
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, encoding='utf-8')
        output = result.stdout + result.stderr

        match = re.search(r"test result:.*?(\{.*?\})", output, re.DOTALL | re.IGNORECASE)
        if match:
            result_dict_str = match.group(1)
            metric_pattern = rf"'{TARGET_METRIC}':\s*(\d+\.\d+)"
            score_match = re.search(metric_pattern, result_dict_str, re.IGNORECASE)

            if score_match:
                score = float(score_match.group(1))
                elapsed = time.time() - start_time
                print(f" {TARGET_METRIC}: {score:.4f} (time: {elapsed:.0f}s)")
                return score
            else:
                print(f" {TARGET_METRIC}")
        else:
            print(output[-300:])

    except Exception as e:
        print(f"error: {e}")

    return 0.0


def main():
    combos = get_combinations(param_grid)
    total_runs = len(combos)

    print(f"total: {total_runs}")
    print(f"target: {TARGET_METRIC}")
    print("=" * 50)

    results = []

    for i, params in enumerate(combos):
        score = run_experiment(params, i + 1, total_runs)

        res_entry = params.copy()
        res_entry['score'] = score
        results.append(res_entry)

    if not results:
        return

    df = pd.DataFrame(results)
    df = df.sort_values(by='score', ascending=False)  # 按分数降序

    best_run = df.iloc[0]
    best_score = best_run['score']
    best_params = best_run.drop('score').to_dict()

    print("\n" + "=" * 50)
    print(f"best_result: {TARGET_METRIC} = {best_score:.4f}")
    print(f"best_params: {best_params}")
    print("=" * 50)

    csv_file = "myrec_tuning_log.csv"
    df.to_csv(csv_file, index=False)
    print(f"📄 save_in: {csv_file}")

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

    print(f" save_result: {txt_file}")


if __name__ == "__main__":
    main()
