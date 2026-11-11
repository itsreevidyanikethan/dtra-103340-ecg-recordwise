"""Regression tests use synthetic records and temporary outputs only."""
from pathlib import Path
import sys,tempfile,unittest,json
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch,yaml
from sklearn.metrics import f1_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import reconstruct as r
from beats import extract_record


class ReconstructionTests(unittest.TestCase):
    def test_rr_intervals_ignore_nonbeat_annotations(self):
        rec=SimpleNamespace(sig_name=['MLII'],p_signal=np.arange(100,dtype=float).reshape(-1,1),fs=10)
        ann=SimpleNamespace(sample=np.array([10,20,30,40]),symbol=['N','V','+','A'])
        with patch('beats.wfdb.rdrecord',return_value=rec),patch('beats.wfdb.rdann',return_value=ann):
            x,y,rr,meta=extract_record('synthetic',return_metadata=True)
        self.assertEqual(meta['sample_positions'],[10,20,40]);self.assertEqual(x.shape,(3,256))
        self.assertEqual(rr[1,0],1.0);self.assertEqual(rr[1,1],2.0)

    def test_bootstrap_equals_explicit_group_resampling(self):
        y=np.array([0,1,2,0,1,2,0,1,2]);pred=np.array([0,1,1,0,0,2,2,1,2]);groups=np.array(['a']*3+['b']*3+['c']*3)
        _,actual=r.bootstrap(y,pred,groups,n_rep=15,seed=9);rng=np.random.default_rng(9);expected=[]
        for _ in range(15):
            chosen=rng.integers(0,3,3);rows=np.concatenate([np.flatnonzero(groups==np.unique(groups)[j]) for j in chosen])
            expected.append(f1_score(y[rows],pred[rows],labels=[0,1,2],average='macro',zero_division=0))
        np.testing.assert_allclose(actual,expected)

    def test_internal_training_freeze_and_predictions(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'outputs';cfg_path=Path(d)/'protocol.yaml'
            cfg=yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/reconstruction.yaml').read_text())
            cfg['split']['train_records']=[101,102,201,202,105];cfg['split']['test_records']=[200,203]
            cfg['models']['cnn'].update(seeds=[1],max_epochs=2,early_stop_patience=1,batch_size=16)
            cfg['models']['extra_trees']['n_estimators']=5;cfg['evaluation']['bootstrap']['replicates']=10
            cfg_path.write_text(yaml.safe_dump(cfg))
            def records(data_root,db,names):
                rng=np.random.default_rng(4);n=len(names)*12
                x=rng.normal(size=(n,256)).astype(np.float32);y=np.tile([0,1,2],n//3)
                rr=rng.random((n,3)).astype(np.float32);rec=np.repeat(list(map(str,names)),12)
                return [x,y,rr,rec,np.arange(n)],{name:{'usable_beats':12} for name in map(str,names)}
            with patch.object(r,'load_records',side_effect=records):r.run_internal(cfg_path,'synthetic',out)
            result=json.loads((out/'mitdb_results.json').read_text())
            self.assertIn('extra_trees',result);self.assertIn('waveform_rr_seed1',result)
            self.assertIn('f1',result['extra_trees']['Normal'])
            split=json.loads((out/'split_manifest.json').read_text())
            self.assertFalse(split['known_subject_overlap'])
            a=np.load(out/'mitdb_waveform_seed1_predictions.npz');self.assertEqual(a['probabilities'].shape,(24,3))
            (out/'training_statistics.json').write_text('{"mean":0,"std":1}')
            with self.assertRaisesRegex(ValueError,'Frozen artifact changed'):
                r.evaluate_database('mitdb','synthetic',['200','203'],cfg,out)


if __name__=='__main__':
    torch.set_num_threads(1);unittest.main()
