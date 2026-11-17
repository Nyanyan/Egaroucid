"""Coordinate, flip, symmetry, and probability checks for the actual rules."""
import unittest
import numpy as np
from common import default_data, load_split
from rule_policy import (extract, legal_moves, flips_for_move, RulePolicy,
                         FEATURE_NAMES, manual_weights)
from policy_data import calc_legal_batch
from policy_accuracy import TRANSFORM_MAPS, transform_bitboards, equivalent_policy_mask


def reference_flips(player,opponent,policy):
    """Independent coordinate walk, without the production bitboard shifts."""
    x,y=policy%8,policy//8
    result=0
    for dx,dy in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
        nx,ny=x+dx,y+dy
        ray=0
        while 0<=nx<8 and 0<=ny<8 and (opponent>>(8*ny+nx))&1:
            ray|=1<<(8*ny+nx)
            nx,ny=nx+dx,ny+dy
        if 0<=nx<8 and 0<=ny<8 and (player>>(8*ny+nx))&1:
            result|=ray
    return result


class RulesTest(unittest.TestCase):
    def test_initial_board_coordinates(self):
        p=np.uint64((1<<35)|(1<<28))
        o=np.uint64((1<<36)|(1<<27))
        expected=sum(1<<i for i in (19,26,37,44))
        self.assertEqual(int(legal_moves(p,o)),expected)
        flips,_,_=flips_for_move(p,o,19)
        self.assertEqual(int(flips),1<<27)
        prob=RulePolicy(manual_weights()).predict(p,o)
        self.assertTrue(np.allclose(prob[[19,26,37,44]],.25))
        equiv=equivalent_policy_mask(np.array([p]),np.array([o]),np.array([19]))
        self.assertEqual(set(np.flatnonzero(equiv[0])),{19,26,37,44})

    def test_full_data_legal_and_symmetries(self):
        records,_,legal=load_split(default_data(),'val')
        rng=np.random.default_rng(613)
        sample=records[rng.choice(len(records),size=128,replace=False)]
        policy=RulePolicy(rng.normal(size=(len(FEATURE_NAMES),3)))
        for row in sample:
            p,o=np.uint64(row['player']),np.uint64(row['opponent'])
            self.assertEqual(int(legal_moves(p,o)),int(calc_legal_batch(np.array([p]),np.array([o]))[0]))
            moves,f=extract(p,o)
            self.assertTrue(np.isfinite(f).all())
            for move in moves:
                self.assertEqual(int(flips_for_move(p,o,int(move))[0]),reference_flips(int(p),int(o),int(move)))
            prob=policy.predict(p,o)
            self.assertAlmostEqual(prob.sum(),1.0,places=12)
            for mapping in TRANSFORM_MAPS:
                tp=transform_bitboards(np.array([p]),mapping)[0]
                to=transform_bitboards(np.array([o]),mapping)[0]
                tm,tf=extract(tp,to)
                lookup={int(m):j for j,m in enumerate(tm)}
                for j,m in enumerate(moves):
                    np.testing.assert_array_equal(f[j],tf[lookup[int(mapping[m])]])
                np.testing.assert_allclose(policy.predict(tp,to)[mapping],prob,atol=1e-12)

    def test_missing_loss_is_rejected(self):
        from evaluate_loss import summarize
        records=np.zeros(2,dtype=[('player','<u8'),('opponent','<u8')])
        with self.assertRaises(ValueError):
            summarize(records,np.array([0,-32768]))
        stats=summarize(records,np.array([0,4]))
        self.assertEqual(stats['positions'],2)
        self.assertEqual(stats['mean_disc_loss'],2)


if __name__=='__main__':
    unittest.main()
