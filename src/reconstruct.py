"""Run an explicitly new subject-grouped ECG benchmark with saved evidence."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import time
from importlib.metadata import version

import joblib
import numpy as np
import torch
import wfdb
import yaml
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from beats import extract_record, waveform_stats
from models import BeatCNN, make_extra_trees, tree_features
from experiment import per_class_metrics


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def write_json(path,obj):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def subject(record):
    return 'subject_201_202' if str(record) in ['201','202'] else str(record)


def load_records(data_root, db, record_names):
    chunks=[]; audit={}
    for record in record_names:
        path=Path(data_root)/db/str(record)
        x,y,rr,meta=extract_record(str(path),return_metadata=True)
        if not len(y):
            audit[str(record)]={**meta,'usable_beats':0};continue
        positions=np.asarray(meta.pop('sample_positions'),dtype=np.int64)
        audit[str(record)]={**meta,'usable_beats':len(y),'class_counts':np.bincount(y,minlength=3).tolist()}
        chunks.append((x,y,rr,np.full(len(y),str(record)),positions))
    if not chunks:raise ValueError('No usable labelled beats')
    arrays=[np.concatenate([c[i] for c in chunks]) for i in range(5)]
    return arrays,audit


def bootstrap(y,pred,groups,n_rep=2000,seed=1):
    """Whole-group bootstrap via summed group confusion matrices.

    Exactly equivalent to resampling all rows of selected groups, including
    multiplicities; avoids copying large external beat arrays every replicate.
    """
    unique=np.unique(groups)
    cms=np.stack([confusion_matrix(y[groups==g],pred[groups==g],labels=[0,1,2]) for g in unique])
    rng=np.random.default_rng(seed);values=[]
    for _ in range(n_rep):
        cm=cms[rng.integers(0,len(unique),len(unique))].sum(0)
        denominator=cm.sum(0)+cm.sum(1)
        f=np.divide(2*np.diag(cm),denominator,out=np.zeros(3),where=denominator!=0)
        values.append(float(f.mean()))
    return {'replicates':n_rep,'unit':'group','ci95':np.percentile(values,[2.5,97.5]).tolist()},np.asarray(values)


@torch.no_grad()
def probabilities(model,x,rr,batch=512):
    model.eval();parts=[]
    for start in range(0,len(x),batch):
        xb=torch.from_numpy(x[start:start+batch]);rb=torch.from_numpy(np.nan_to_num(rr[start:start+batch]))
        parts.append(torch.softmax(model(xb,rb),dim=1).numpy())
    return np.concatenate(parts)


def train_model(name,rr_branch,seed,cfg,x,y,rr,xv,yv,rrv,out):
    checkpoint=out/f'{name}.pt';complete=out/f'{name}_complete.json';resume=out/f'{name}_resume.pt'
    if complete.exists():return
    torch.manual_seed(seed);np.random.seed(seed)
    model=BeatCNN(rr_branch=rr_branch);opt=torch.optim.Adam(model.parameters(),lr=cfg['lr'])
    criterion=torch.nn.CrossEntropyLoss();xt=torch.from_numpy(x);yt=torch.from_numpy(y)
    rrt=torch.from_numpy(np.nan_to_num(rr));best=-1.;best_state=None;best_epoch=0;stale=0;history=[];first=1
    if resume.exists():
        saved=torch.load(resume,weights_only=False,map_location='cpu')
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
        best=saved['best'];best_state=saved['best_state'];best_epoch=saved['best_epoch']
        stale=saved['stale'];history=saved['history'];first=saved['epoch']+1
        np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng'])
    for epoch in range(first,cfg['max_epochs']+1):
        if stale>=cfg['early_stop_patience']:break
        began=time.monotonic();model.train();order=np.random.permutation(len(y))
        for start in range(0,len(y),cfg['batch_size']):
            idx=order[start:start+cfg['batch_size']];opt.zero_grad()
            loss=criterion(model(xt[idx],rrt[idx]),yt[idx]);loss.backward();opt.step()
        score=float(f1_score(yv,probabilities(model,xv,rrv).argmax(1),labels=[0,1,2],average='macro',zero_division=0))
        history.append({'epoch':epoch,'validation_macro_f1':score,'seconds':time.monotonic()-began})
        if score>best:best=score;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch;stale=0
        else:stale+=1
        state={'epoch':epoch,'model':model.state_dict(),'optimizer':opt.state_dict(),
               'best':best,'best_state':best_state,'best_epoch':best_epoch,'stale':stale,'history':history,
               'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state()}
        tmp=resume.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(resume)
        print(name,epoch,round(score,6),flush=True)
    torch.save(best_state,checkpoint)
    write_json(out/f'{name}_history.json',{'selected_epoch':best_epoch,'best_validation_macro_f1':best,'history':history})
    write_json(complete,{'complete':True,'new_reconstruction':True})


def run_internal(cfg_path,data_root,out):
    cfg=yaml.safe_load(Path(cfg_path).read_text());out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'mitdb_complete.json').exists():print('Internal evaluation already complete.');return
    development=list(map(str,cfg['split']['train_records']));test=list(map(str,cfg['split']['test_records']))
    if {subject(r) for r in development}&{subject(r) for r in test}:
        raise ValueError('Known subject crosses development/test partitions')
    cache=out/'development_cache.npz'
    if cache.exists():
        a=np.load(cache,allow_pickle=False);x,y,rr,records,positions=(a[k] for k in ['x','y','rr','records','positions'])
    else:
        (x,y,rr,records,positions),audit=load_records(data_root,'mitdb',development)
        np.savez_compressed(cache,x=x,y=y,rr=rr,records=records,positions=positions)
        write_json(out/'development_record_audit.json',audit)
    group=np.array([subject(r) for r in records])
    splitter=GroupShuffleSplit(n_splits=1,test_size=cfg['split']['validation_fraction'],random_state=cfg['split']['validation_seed'])
    tr,va=next(splitter.split(x,y,group));mean=float(x[tr].mean());std=float(x[tr].std())
    x=((x-mean)/std).astype(np.float32)
    split_info={'train_records':sorted(set(records[tr])),'validation_records':sorted(set(records[va])),
                'test_records':test,'known_subject_overlap':False,'seed':cfg['split']['validation_seed']}
    write_json(out/'split_manifest.json',split_info);write_json(out/'training_statistics.json',{'mean':mean,'std':std})
    write_json(out/'run_metadata.json',{'run_kind':'new_reconstruction','original_manuscript_not_available':True,
        'protocol_sha256':digest(cfg_path),'python':platform.python_version(),
        'packages':{p:version(p) for p in ['torch','numpy','scikit-learn','wfdb']},
        'subject_split_change':'201 and 202 co-assigned to development; distinct from supplied record-only split',
        'validation_rows':len(va),'training_rows':len(tr),'threads':torch.get_num_threads()})
    models=[]
    for variant in cfg['models']['cnn']['variants']:
        for seed in cfg['models']['cnn']['seeds']:
            name=f'{variant}_seed{seed}';models.append((name,variant=='waveform_rr'))
            train_model(name,variant=='waveform_rr',seed,cfg['models']['cnn'],x[tr],y[tr],rr[tr],x[va],y[va],rr[va],out)
    forest=out/'extra_trees.joblib'
    if not forest.exists():
        tree=make_extra_trees(cfg['models']['extra_trees']['n_estimators'],cfg['models']['extra_trees']['seed'])
        tree.set_params(n_jobs=cfg['models']['extra_trees'].get('n_jobs',2))
        tree.fit(tree_features(rr[tr],waveform_stats(x[tr])),y[tr]);joblib.dump(tree,forest,compress=3)
    frozen={p.name:digest(p) for p in [out/f'{name}.pt' for name,_ in models]+[forest,out/'training_statistics.json',out/'split_manifest.json']}
    write_json(out/'freeze.json',{'protocol_sha256':digest(cfg_path),'artifacts':frozen})
    evaluate_database('mitdb',data_root,test,cfg,out,subject_groups=True)
    write_json(out/'mitdb_complete.json',{'complete':True})


def evaluate_database(db,data_root,record_names,cfg,out,subject_groups=False):
    out=Path(out);stats=json.loads((out/'training_statistics.json').read_text());frozen=json.loads((out/'freeze.json').read_text())
    for name,h in frozen['artifacts'].items():
        if digest(out/name)!=h:raise ValueError(f'Frozen artifact changed: {name}')
    (x,y,rr,records,positions),audit=load_records(data_root,db,record_names)
    x=((x-stats['mean'])/stats['std']).astype(np.float32)
    group=np.array([subject(r) for r in records]) if subject_groups else records
    write_json(out/f'{db}_record_audit.json',audit)
    result={'database':db,'new_reconstruction':True,'beats':len(y),'records':len(set(records)),
            'class_order':['Normal','VEB','Other'],'cluster_unit':'known_subject' if subject_groups else 'record',
            'external_limit':'Record clusters may not represent independent Holter subjects; no survival or heart-attack prediction claim.'}
    for variant in cfg['models']['cnn']['variants']:
        for seed in cfg['models']['cnn']['seeds']:
            name=f'{variant}_seed{seed}';model=BeatCNN(rr_branch=variant=='waveform_rr')
            model.load_state_dict(torch.load(out/f'{name}.pt',weights_only=True,map_location='cpu'))
            p=probabilities(model,x,rr);pred=p.argmax(1)
            result[name]=per_class_metrics(y,pred)
            interval,draws=bootstrap(y,pred,group,cfg['evaluation']['bootstrap']['replicates'],seed=1)
            result[name]['macro_f1_bootstrap']=interval
            np.savez_compressed(out/f'{db}_{name}_predictions.npz',labels=y,probabilities=p,records=records,sample_positions=positions)
            np.save(out/f'{db}_{name}_bootstrap.npy',draws)
    tree=joblib.load(out/'extra_trees.joblib');p=tree.predict_proba(tree_features(rr,waveform_stats(x)))
    pred=tree.classes_[p.argmax(1)];result['extra_trees']=per_class_metrics(y,pred)
    interval,draws=bootstrap(y,pred,group,cfg['evaluation']['bootstrap']['replicates'],seed=1)
    result['extra_trees']['macro_f1_bootstrap']=interval
    np.savez_compressed(out/f'{db}_extra_trees_predictions.npz',labels=y,probabilities=p,records=records,sample_positions=positions,class_order=tree.classes_)
    np.save(out/f'{db}_extra_trees_bootstrap.npy',draws)
    write_json(out/f'{db}_results.json',result)
    print(db,'evaluation complete:',len(y),'beats',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config',default='configs/reconstruction.yaml')
    ap.add_argument('--data-root',default='data');ap.add_argument('--out',default='outputs');ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--external',choices=['incartdb','svdb','sddb']);a=ap.parse_args();torch.set_num_threads(a.threads)
    if a.external:
        cfg=yaml.safe_load(Path(a.config).read_text())
        if digest(a.config)!=json.loads((Path(a.out)/'freeze.json').read_text())['protocol_sha256']:
            raise ValueError('Protocol changed since model freezing')
        records=sorted(p.stem for p in (Path(a.data_root)/a.external).glob('*.hea'))
        if a.external == 'sddb':
            included=[r for r in records if (Path(a.data_root)/a.external/(r+'.atr')).is_file()]
            write_json(Path(a.out)/'sddb_annotation_policy.json', {
                'annotation':'audited atr only; unaudited ari excluded',
                'included_records':included, 'excluded_local_records':[r for r in records if r not in included]})
            records=included
        evaluate_database(a.external,a.data_root,records,cfg,a.out)
    else:run_internal(a.config,a.data_root,a.out)


if __name__=='__main__':main()
