import os.path
import sys
import pandas as pd
import matplotlib.pyplot as plt
import yaml

EVAL_FILENAME = 'eval.csv'
HYDRA_DIR = '.hydra'
CONFIG_FILE = 'config.yaml'
OVERRIDE_FILE = 'overrides.yaml'


def read_metrics(filename):
    metrics = pd.read_csv(filename).to_dict()
    return metrics


def parse_kv(kv_pair):
    key, value = kv_pair.split('=')
    key = key.split('.')[-1]
    return key, value


def read_metadata(run_path, config_filename, override_filename):
    run_name = os.path.split(run_path)[-1]
    metadata = run_name.split(",")
    metadata = dict(parse_kv(kv_pairs) for kv_pairs in metadata)
    metadata["run_path"] = run_path
    return metadata


def read_run_metric(run_path):
    hydra_dir = os.path.join(run_path, HYDRA_DIR)
    return {
        "metrics": read_metrics(os.path.join(run_path, EVAL_FILENAME)),
        "metadata": read_metadata(run_path, os.path.join(hydra_dir, CONFIG_FILE),
                                  os.path.join(hydra_dir, OVERRIDE_FILE))
    }


def read_runs_metrics(run_root_dir):
    runs = [dir for dir in os.listdir(run_root_dir) if os.path.isdir(os.path.join(run_root_dir, dir))]
    print("Found {} runs.".format(len(runs)))
    print(runs)
    metrics = [read_run_metric(os.path.join(run_root_dir, run)) for run in runs]
    return metrics


def get_label_from_metadata(metadata):
    algos = [k for k, v in metadata.items() if v.strip().lower() == 'true']
    if "drq_k" in metadata or "drq_m" in metadata:
        algos.append("drq_" + metadata["aug_type"])
    return '+'.join(algos) if algos else 'baseline_dqn'


def plot_data(runs, OUT_DIR):
    import matplotlib.pyplot as plt  # <shrugs>, don't hate the player, hate the game
    envs = {
        "Pong": plt.subplots(),
        "Breakout": plt.subplots(),
        "SpaceInvaders": plt.subplots()
    }

    # Store all plots in dict
    for run in runs:
        env = run['metadata']['env']
        steps = list(run['metrics']['step'].values())
        rewards = list(run['metrics']['episode_reward'].values())
        fig, ax = envs[env]
        ax.plot(steps, rewards, label=get_label_from_metadata(run['metadata']))

    for env, (plt, ax) in envs.items():
        ax.set_title("Performance on "  + env)
        ax.set_xlabel("training steps")
        ax.set_ylabel("eval reward")
        ax.legend(loc='upper left', prop={'size': 6})
        ax.grid(which='both')
        plt.savefig(os.path.join(OUT_DIR, env + ".png"), dpi=300)


if __name__ == '__main__':
    directory_path = sys.argv[1]
    OUT_DIR = directory_path if len(sys.argv) <= 2 else sys.argv[2]
    data = read_runs_metrics(directory_path)
    plot_data(data, OUT_DIR)
