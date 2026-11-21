from model import *
from sklearn.metrics import accuracy_score
m,tr,te,pred,bp=train()
for label,v in EXAMPLES.items():
    r=diagnose(m,v); assert r['diagnosis']==label,(label,r); print(label,'PASS')
for v in [(None,2,3),(float('nan'),2,3),(120,24,-1),(120,1001,2)]: assert diagnose(m,v)['diagnosis']=='Sensor fault'
for v in [(0,0,2),(30,10,2),(120,20,0),(120,180,2)]: assert diagnose(m,v)['diagnosis']=='Uncertain'
print('Invalid and out-of-envelope checks PASS')
print('RF accuracy',accuracy_score(te.label,pred),'baseline',accuracy_score(te.label,bp))
