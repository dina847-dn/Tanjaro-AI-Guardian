"""Synthetic proof of concept; not a calibrated plant model."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
COLS=['inlet_ntu','outlet_ntu','flow_lpm']
FEATURES=COLS+['reduction_pct','flow_ratio']
LABELS=['Normal','Possible blockage','Poor filtration','Mixed abnormality']
EXAMPLES={'Normal':(120,24,2.5),'Possible blockage':(120,26,.85),'Poor filtration':(120,95,2.4),'Mixed abnormality':(120,95,.7),'Sensor fault':(120,-1,2.5)}
ACTIONS={'Normal':'Continue monitoring; confirm water quality independently.','Possible blockage':'Inspect filter, valves, pump and flow sensor. Low flow alone cannot identify the cause.','Poor filtration':'Inspect filter media, sedimentation and turbidity calibration.','Mixed abnormality':'Investigate both hydraulic and filtration conditions.','Sensor fault':'Check wiring, calibration, missing values and sensor range.','Uncertain':'Request operator review; do not automate any action.'}
def synthetic(seed,n):
    rng=np.random.default_rng(seed); rows=[]
    for label in LABELS:
        bad='filtration' in label or label=='Mixed abnormality'; low=label in ['Possible blockage','Mixed abnormality']
        for _ in range(n):
            a=rng.uniform(60,240); ratio=np.clip(rng.normal(.76 if bad else .22,.12),.02,1.15); q=np.clip(rng.normal(.85 if low else 2.5,.38),.05,4)
            rows.append([a,a*ratio,q,label])
    return pd.DataFrame(rows,columns=COLS+['label']).sample(frac=1,random_state=seed).reset_index(drop=True)
def features(df):
    x=df[COLS].astype(float).copy(); x['reduction_pct']=100*(1-x.outlet_ntu/x.inlet_ntu); x['flow_ratio']=x.flow_lpm/2.5; return x

def baseline(a,b,q):
    low=q<1.5; poor=(1-b/a)<.5
    return 'Mixed abnormality' if low and poor else 'Possible blockage' if low else 'Poor filtration' if poor else 'Normal'
def train():
    tr=synthetic(42,500); te=synthetic(2026,150)
    m=RandomForestClassifier(n_estimators=80,min_samples_leaf=5,random_state=42,n_jobs=1).fit(features(tr),tr.label)
    pred=m.predict(features(te)); bp=[baseline(*r) for r in te[COLS].itertuples(index=False,name=None)]
    return m,tr,te,pred,bp

def diagnose(m,values):
    try: a,b,q=map(float,values)
    except (ValueError,TypeError): a,b,q=np.nan,np.nan,np.nan
    out={'inlet_ntu':a,'outlet_ntu':b,'flow_lpm':q,'reduction_pct':np.nan,'diagnosis':'Sensor fault','model_vote':np.nan,'baseline':'Not evaluated','reason':'Missing, non-finite or negative reading; or outside configured demonstration instrument limits.'}
    if not np.isfinite([a,b,q]).all() or min(a,b,q)<0 or a>1000 or b>1000 or q>10: return out
    if a>0: out['reduction_pct']=100*(1-b/a)
    if not 60<=a<=240 or not .05<=q<=4 or b/a>1.15:
        out.update(diagnosis='Uncertain',reason='Outside synthetic model operating envelope; needs operator review.'); return out
    x=features(pd.DataFrame([[a,b,q]],columns=COLS)); p=m.predict_proba(x)[0]; k=int(p.argmax()); score=float(p[k]); label=str(m.classes_[k]); out.update(diagnosis=label if score>=.65 else 'Uncertain',model_vote=score,baseline=baseline(a,b,q),reason='Random Forest estimated class support; not a calibrated probability or confirmed cause.')
    return out
