import argparse

# PyTorch >= 2.6 compatibility: torch.load defaults to weights_only=True,
# which breaks old RecBole checkpoint saving/loading (the saved files embed
# a RecBole Config object). Force the legacy permissive behavior.
import torch

_orig_torch_load = torch.load


def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_compat

from recbole.quick_start import run_recbole


def _convert_value(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', 'yes'):
        return True
    if value.lower() in ('false', 'no'):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def parse_config_dict(unknown_args):
    config_dict = {}
    i = 0
    while i < len(unknown_args):
        arg = unknown_args[i]
        if not arg.startswith('--'):
            raise argparse.ArgumentTypeError(f"Unexpected argument: {arg}")

        if '=' in arg:
            key, value = arg[2:].split('=', 1)
            i += 1
        else:
            key = arg[2:]
            i += 1
            if i >= len(unknown_args) or unknown_args[i].startswith('--'):
                raise argparse.ArgumentTypeError(f"Missing value for argument: --{key}")
            value = unknown_args[i]
            i += 1

        config_dict[key] = _convert_value(value)

    return config_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='ACLSRec', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='Amazon_Beauty', help='name of datasets')
    parser.add_argument('--config_files', type=str, default='seq.yaml', help='config files')

    args, unknown_args = parser.parse_known_args()
    config_dict = parse_config_dict(unknown_args)

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    run_recbole(
        model=args.model,
        dataset=args.dataset,
        config_file_list=config_file_list,
        config_dict=config_dict or None,
    )
