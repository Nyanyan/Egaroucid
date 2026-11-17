"""Export the small trained policy and path-independent measured results to Git."""
import argparse
from common import HERE, read, save, file_sha256


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
        summary['disc_loss_l21']=result
    else:
        summary['disc_loss_l21']=dict(status='pending')
    save(HERE/'results_summary.json',summary)
    print('Exported trained_policy.json and results_summary.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=__import__('pathlib').Path,default=HERE/'output/quest_800-1000')
    export(parser.parse_args().input)
