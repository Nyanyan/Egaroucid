"""Print the probability and additive rule contributions for every legal move."""
import argparse
import json
from pathlib import Path
import numpy as np
from rule_policy import RulePolicy, FEATURE_NAMES, extract, phase_weights


def run(args):
    policy=RulePolicy.load(args.model)
    p,o=np.uint64(args.player),np.uint64(args.opponent)
    if p&o:
        raise ValueError('Player and opponent discs overlap')
    probabilities=policy.predict(p,o)
    moves,features=extract(p,o)
    phase=np.asarray(phase_weights(p,o))
    rows=[]
    for move,vector in zip(moves,features):
        index=63-int(move)
        contributions=vector*(policy.weights@phase)
        rows.append(dict(move=chr(97+index%8)+str(index//8+1),policy_index=int(move),
                         probability=float(probabilities[move]),logit=float(contributions.sum()),
                         contributions={name:float(v) for name,v in zip(FEATURE_NAMES,contributions) if abs(v)>1e-12}))
    rows.sort(key=lambda row:(-row['probability'],row['policy_index']))
    print(json.dumps(dict(phase=phase.tolist(),moves=rows),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,default=Path(__file__).with_name('trained_policy.json'))
    parser.add_argument('--player',type=lambda s:int(s,0),required=True)
    parser.add_argument('--opponent',type=lambda s:int(s,0),required=True)
    run(parser.parse_args())
