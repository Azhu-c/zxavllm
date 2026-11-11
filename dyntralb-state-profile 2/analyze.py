"""Chronological holdout: predict the NEXT pre-schedule running load (not a reward)."""
import argparse
import glob
import json
import math
from collections import Counter

FEATURES = ('running_count', 'waiting_count', 'running_blocks', 'waiting_blocks',
            'running_max_blocks', 'waiting_max_blocks')
ACTIONS = ('scheduled_requests', 'scheduled_tokens', 'scheduled_from_waiting', 'known_scheduled_blocks_pre')

def solve(a, b):
    n = len(b)
    for i in range(n):
        j = max(range(i, n), key=lambda k: abs(a[k][i]))
        a[i], a[j] = a[j], a[i]; b[i], b[j] = b[j], b[i]
        if abs(a[i][i]) < 1e-12: raise ValueError('singular model')
        z = a[i][i]
        for k in range(i, n): a[i][k] /= z
        b[i] /= z
        for j in range(n):
            if j == i: continue
            z = a[j][i]
            for k in range(i, n): a[j][k] -= z * a[i][k]
            b[j] -= z * b[i]
    return b

def fit_evaluate(train, test, use_actions):
    xs = [x[0] + (x[1] if use_actions else []) for x in train]
    ys = [x[2] for x in train]
    dim = len(xs[0]); mean = [sum(x[j] for x in xs)/len(xs) for j in range(dim)]
    std = [max(math.sqrt(sum((x[j]-mean[j])**2 for x in xs)/len(xs)), 1e-9) for j in range(dim)]
    def transform(x): return [1.] + [(x[j]-mean[j])/std[j] for j in range(dim)]
    z = [transform(x) for x in xs]; d = dim + 1
    matrix = [[sum(v[i]*v[j] for v in z) + (1.0 if i == j and i else 0.0) for j in range(d)] for i in range(d)]
    vec = [sum(v[i]*y for v,y in zip(z,ys)) for i in range(d)]
    weights = solve(matrix, vec)
    errors = []
    for s,a,y in test:
        pred = sum(w*v for w,v in zip(weights, transform(s + (a if use_actions else []))))
        errors.append(abs(pred-y))
    return sum(errors)/len(errors)

def main():
    p = argparse.ArgumentParser(); p.add_argument('directory'); p.add_argument('--output', default=None)
    args = p.parse_args(); paths = sorted(glob.glob(args.directory + '/rank-*-pid-*.jsonl'))
    if not paths: raise SystemExit('没有发现 rank-*-pid-*.jsonl；请先运行真实采集。')
    train=[]; test=[]; actions=Counter(); errors=[]; rows_count=0
    for path in paths:
        rows=[]
        with open(path, encoding='utf-8') as f:
            for lineno,line in enumerate(f, 1):
                try:
                    r=json.loads(line); assert r['schema_version']==1
                    rows.append(r)
                except Exception as exc: errors.append(f'{path}:{lineno}: {exc}')
        rows_count += len(rows)
        for r in rows: actions[r['action_observed']['scheduled_requests']] += 1
        split = int(len(rows)*0.7)
        for i in range(len(rows)-1):
            x,y=rows[i],rows[i+1]
            if x['rank'] != y['rank'] or x['pid'] != y['pid'] or y['step'] != x['step'] + 1: continue
            if i < split <= i+1: continue
            try:
                item=([float(x['state_pre'][k]) for k in FEATURES],
                      [float(x['action_observed'][k]) for k in ACTIONS],
                      float(y['state_pre']['running_blocks']))
            except (KeyError, TypeError, ValueError): continue
            (train if i+1 < split else test).append(item)
    report={'rows': rows_count, 'files': len(paths), 'valid_transitions_train':len(train),
            'valid_transitions_test':len(test), 'scheduled_requests_histogram':dict(sorted(actions.items())),
            'parse_errors':errors[:20], 'target':'next pre-schedule running_blocks; not a reward',
            'warning':'Observational association only; actions are policy-selected; not an offline policy evaluation.'}
    if len(train) >= 20 and len(test) >= 5:
        mean=sum(x[2] for x in train)/len(train)
        report['test_mae_train_mean']=sum(abs(x[2]-mean) for x in test)/len(test)
        report['test_mae_state']=fit_evaluate(train,test,False)
        report['test_mae_state_action']=fit_evaluate(train,test,True)
    else: report['status']='样本不足：至少需要 20 个训练和 5 个测试转移；实际研究需要更多。'
    result=json.dumps(report,ensure_ascii=False,indent=2)
    if args.output:
        with open(args.output,'x',encoding='utf-8') as f: f.write(result+'\n')
    print(result)

if __name__=='__main__': main()
