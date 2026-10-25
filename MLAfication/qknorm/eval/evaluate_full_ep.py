"""Evaluate a completed portable EP export using the original benchmark paths."""
import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Evaluation JSON with dataset/task paths')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    checkpoint = Path(config['checkpoint'])
    if not (checkpoint / 'export_complete.json').is_file():
        raise RuntimeError('Portable export is incomplete')
    output = Path(config['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    environment = os.environ.copy()
    environment['MHA2MLA_DATA_PATH'] = config['data_root']
    environment['LONGBENCH_DATASET'] = config['longbench_source']
    environment['LOCAL_WORLD_SIZE'] = str(config.get('world_size', 8))
    environment['WANDB_MODE'] = 'disabled'
    world = str(config.get('world_size', 8))
    # Keep the LightEval-generated path component below filesystem name limits.
    short_path = output / 'full-model'
    if not short_path.exists():
        short_path.symlink_to(checkpoint, target_is_directory=True)

    def run(name, command, single_gpu=False):
        env = environment.copy()
        if single_gpu:
            env['CUDA_VISIBLE_DEVICES'] = '0'
            env.pop('LOCAL_WORLD_SIZE', None)
        with (output / f'{name}.log').open('a') as log:
            subprocess.run(command, cwd=here.parent, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)

    run('commonsense', [sys.executable, '-m', 'accelerate.commands.launch', '--multi_gpu',
        '--num_processes', world, '--num_machines', '1', '--mixed_precision', 'bf16',
        '--dynamo_backend', 'no', '--main_process_port', '29551', str(here / 'eval_v2.py'), 'accelerate',
        '--model_args', f'pretrained={short_path},revision=main,dtype=bfloat16,max_length=4096,model_parallel=false',
        '--override_batch_size', str(config.get('commonsense_batch_size', 8)),
        '--custom_tasks', config['custom_tasks'], '--tasks', config['tasks_file'],
        '--output_dir', str(output / 'commonsense')])
    common_results = sorted((output / 'commonsense').rglob('results_*.json'))
    if not common_results:
        raise RuntimeError('Commonsense did not produce a result file')
    results = json.loads(common_results[-1].read_text())['results']
    task_keys = {'ARC': 'arc:_average', 'HellaSwag': 'hellaswag', 'MMLU': 'mmlu',
                 'OBQA': 'openbook_qa', 'PIQA': 'piqa', 'WinoGrande': 'winogrande'}
    common = {name: results[f'custom|{key}|0']['acc_norm'] * 100 for name, key in task_keys.items()}
    common['Avg@CS'] = sum(common.values()) / len(common)
    (output / 'paper_commonsense.json').write_text(json.dumps(common, indent=2, allow_nan=False) + '\n')
    longbench = {
        'seed': 42, 'report_to': 'none', 'output_dir': str(output / 'longbench'),
        'per_device_eval_batch_size': 1, 'max_steps': 0,
        'model_name_or_path': config['model_name_or_path'],
        'mla_tensor_path': str(checkpoint / 'mla_tensors.pt'), 'is_baseline': False,
        'ckpt_path': str(checkpoint), 'longbench': True, 'lb_max_tokens': 4096,
        'lb_batch_size': config.get('longbench_batch_size', 4),
    }
    lb_config = output / 'longbench.yaml'
    lb_config.write_text(yaml.safe_dump(longbench, sort_keys=False))
    run('longbench', [sys.executable, '-m', 'torch.distributed.run', '--standalone',
        '--nproc_per_node', world, str(here / 'longbench.py'), '--cfg_file', str(lb_config)])
    scores = defaultdict(list)
    for line in (output / 'longbench/lbv1_4k.jsonl').read_text().splitlines():
        row = json.loads(line)
        scores[row['dataset']].append(row['score'])
    excluded = {'multifieldqa_zh', 'dureader', 'vcsum', 'lsht', 'passage_retrieval_zh'}
    longbench_scores = {name: 100 * sum(values) / len(values) for name, values in scores.items()}
    english = [value for name, value in longbench_scores.items() if name not in excluded]
    if len(english) != 16 or len(scores) != 21:
        raise RuntimeError('LongBench did not evaluate every expected dataset')
    longbench_scores['Avg@LB'] = sum(english) / len(english)
    (output / 'paper_longbench.json').write_text(json.dumps(longbench_scores, indent=2, allow_nan=False) + '\n')
    vio_output = output / 'vio.tsv'
    run('vio', [sys.executable, str(here / 'collect_moe_vio.py'),
        '--model_name_or_path', config['model_name_or_path'], '--ckpt', f'full={checkpoint}',
        '--ckpt', f"peft={config['peft_checkpoint']}", '--seq_length', '4096', '--num_samples', '128',
        '--batch_size', '1', '--data_seed', '42', '--sample_offset', '512000',
        '--data_path', config['data_root'], '--output_file', str(vio_output)], single_gpu=True)
    maxima = defaultdict(list)
    with (output / 'vio_by_layer.tsv').open() as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            if int(row['tokens']) <= 0:
                raise RuntimeError('Vio contains an unobserved layer')
            maxima[row['label']].append(float(row['max_vio']))
    summary = {label: {'max_vio': max(values), 'mean_vio': sum(values) / len(values),
                       'num_layers': len(values)} for label, values in maxima.items()}
    (output / 'paper_vio.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    (output / 'evaluation_complete.json').write_text(json.dumps({'checkpoint': str(checkpoint)}) + '\n')


if __name__ == '__main__':
    main()
