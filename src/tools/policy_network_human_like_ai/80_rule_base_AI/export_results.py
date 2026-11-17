"""Export the small trained policy and path-independent measured results to Git."""
import argparse
from pathlib import Path
import subprocess
import time
import numpy as np
from common import HERE, REPO, read, save, file_sha256


def export(directory):
    agreement=read(directory/'agreement.json')
    model=read(directory/'selected_model.json')
    if file_sha256(directory/'selected_model.json')!=agreement['model_sha256']:
        raise ValueError('Selected model hash differs')
    exported={key:model[key] for key in ('kind','features','phases','weights','phase_formula','policy_formula','tie_rule','selection')}
    exported['training']=dict(dataset='Othello Quest [800,1000)',counts=agreement['identity']['counts'],
                              seed=agreement['identity']['seed'],selection=agreement['identity']['selection'],
                              manifest_sha256=agreement['identity']['manifest_sha256'],
                              source_hashes=agreement['identity']['source_hashes'],
                              original_model_sha256=agreement['model_sha256'])
    save(HERE/'trained_policy.json',exported)
    summary={key:agreement[key] for key in ('selected_validation','test','paired_game_bootstrap','model_sha256','predictions_sha256','records_sha256')}
    summary['training']=exported['training']
    summary['agreement_definition']='legal Top1; board-preserving D4 symmetries accepted; ties ascending bit index'
    loss=directory/'loss_l21/result.json'
    if loss.exists():
        result=read(loss)
        if result['status']!='complete' or result['identity']['partial']:
            raise ValueError('L21 loss must cover the full test')
        if result['identity']['model_sha256']!=agreement['model_sha256']:
            raise ValueError('L21 loss model differs')
        if result['identity']['prediction_sha256']!=agreement['predictions_sha256']:
            raise ValueError('L21 saved moves differ')
        for filename,key in (('scores.npy','scores_sha256'),('per_position_loss.npy','per_position_loss_sha256')):
            if file_sha256(directory/'loss_l21'/filename)!=result[key]:
                raise ValueError('L21 score file changed: '+filename)
        losses=np.load(directory/'loss_l21/per_position_loss.npy',allow_pickle=False)
        measured=result['metrics']
        if (len(losses)!=agreement['identity']['counts']['test'] or np.any(losses<0)
            or int(losses.sum(dtype=np.int64))!=measured['total_disc_loss']
            or float(losses.mean())!=measured['mean_disc_loss']):
            raise ValueError('Full-test L21 mean does not reproduce from per-position losses')
        summary['disc_loss_l21']=result
        runs=directory/'loss_l21/execution_runs.json'
        if runs.exists():
            summary['disc_loss_execution_runs']=read(runs)
    else:
        summary['disc_loss_l21']=dict(status='pending')
    save(HERE/'results_summary.json',summary)
    print('Exported trained_policy.json and results_summary.json')


def wait_for_loss(directory):
    expected=read(directory/'agreement.json')['model_sha256']
    started=time.monotonic()
    last=-60
    while not (directory/'loss_l21/result.json').exists():
        if read(directory/'agreement.json')['model_sha256']!=expected:
            raise ValueError('Model changed while awaiting loss; export cancelled')
        status=read(directory/'loss_l21/status.json')
        if status['status']=='failed':
            raise RuntimeError('L21 evaluation failed: '+status.get('error','unknown error'))
        elapsed=time.monotonic()-started
        if elapsed>6*3600:
            raise TimeoutError('Timed out waiting for L21 evaluation')
        if elapsed-last>=60:
            print(f'Waiting for full L21 result: {status["completed_pairs"]}/{status["total_pairs"]}; '
                  f'workers={status["workers"]}; ETA={status.get("eta_sec")} sec',flush=True)
            last=elapsed
        time.sleep(5)
    if read(directory/'agreement.json')['model_sha256']!=expected:
        raise ValueError('Model changed while awaiting loss; export cancelled')


def commit_results():
    # Explicit paths and --only keep unrelated staged work out of this commit.
    paths=[str((HERE/name).relative_to(REPO)) for name in ('trained_policy.json','results_summary.json')]
    result=subprocess.run(['git','diff','--quiet','HEAD','--',*paths],cwd=REPO)
    if result.returncode==0:
        print('Results already committed',flush=True)
    elif result.returncode==1:
        subprocess.run(['git','diff','--check','--',*paths],cwd=REPO,check=True)
        subprocess.run(['git','commit','--only','-m','Record full L21 disc loss for the Quest rule policy','--',*paths],cwd=REPO,check=True)
    else:
        raise RuntimeError('Could not inspect result diff')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=HERE/'output/quest_800-1000')
    parser.add_argument('--wait-for-loss',action='store_true')
    parser.add_argument('--commit',action='store_true')
    args=parser.parse_args()
    if args.wait_for_loss:
        wait_for_loss(args.input)
    export(args.input)
    if args.commit:
        commit_results()
