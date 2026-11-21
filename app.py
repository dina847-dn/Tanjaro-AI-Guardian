import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from model import *
from datetime import datetime,timezone
st.set_page_config(page_title='Tanjaro AI Guardian', page_icon='Screenshot 2026-11-22 005652.png', layout='wide')
st.markdown('''<style>.stApp{background:#f5f8fb;color:#15324b} [data-testid="stMetric"]{background:white;border:1px solid #d9e4eb;padding:16px;border-radius:10px} h1,h2,h3{color:#123b54}</style>''',unsafe_allow_html=True)
st.title('TANJARO AI GUARDIAN')
st.caption('AI-assisted wastewater monitoring | Diya Shirwan & Dina Shirwan | Civil Engineering · KUST')
st.warning('Research prototype — no live Tanjaro measurements or validated plant control. Turbidity does not establish water safety.')
@st.cache_resource
def load(): return train()
m,tr,te,pred,bp=load()
if 'history' not in st.session_state: st.session_state.history=[]
page=st.sidebar.radio('Navigation',['Monitoring dashboard','Batch data analysis','Model evidence & limitations'])
st.sidebar.caption('Reference flow: 2.5 L/min (synthetic setup). Valid only within the declared demonstration envelope.')
def record(v,source):
    r=diagnose(m,v); r.update(source=source,timestamp=datetime.now(timezone.utc).isoformat()); st.session_state.history.append(r); return r

def display(r):
    c=st.columns(4)
    for col,key,title,unit in zip(c,COLS+['reduction_pct'],['Inlet turbidity','Outlet turbidity','Flow rate','Turbidity reduction'],['NTU','NTU','L/min','%']):
        val=r[key]; col.metric(title,f'{val:.2f} {unit}' if pd.notna(val) else 'Unavailable')
    label=r['diagnosis']; (st.success if label=='Normal' else st.warning if label in ['Uncertain','Possible blockage'] else st.error)(f'AI diagnosis: {label}')
    st.write(ACTIONS[label]); st.caption(r['reason']); st.write('Model vote share:',f"{r['model_vote']:.1%}" if pd.notna(r['model_vote']) else 'Not evaluated',' | Threshold baseline:',r['baseline'])
if page=='Monitoring dashboard':
    mode=st.radio('Data source',['Simulated scenario','Manual experimental readings'],horizontal=True)
    if mode=='Simulated scenario':
        scenario=st.selectbox('Operating condition',list(EXAMPLES)); vals=EXAMPLES[scenario]; st.info('SIMULATED DATA — selected scenario, not measured water.'); r=diagnose(m,vals); display(r)
        if st.button('Record this sample'): record(vals,'Simulated scenario')
        if st.button('Run 20 simulated readings'):
            import numpy as np
            rng=np.random.default_rng(12)
            for i in range(20): record(np.array(vals)*rng.uniform(.94,1.06,3),'Simulated sequence')
    else:
        st.info('Enter calibrated sensor measurements. Manual entry does not verify their origin. Use matched inlet/outlet samples accounting for residence time.')
        with st.form('manual'):
            c=st.columns(3); vals=[c[i].number_input(n,value=float(v),format='%.2f') for i,(n,v) in enumerate(zip(['Inlet NTU','Outlet NTU','Flow L/min'],EXAMPLES['Normal']))]; go=st.form_submit_button('Analyze and record')
        if go: st.session_state.last_manual=record(vals,'User-entered measurements')
        if 'last_manual' in st.session_state: display(st.session_state.last_manual)
    st.subheader('Session history')
    if st.session_state.history:
        df=pd.DataFrame(st.session_state.history); st.plotly_chart(px.line(df.reset_index(),x='index',y=['inlet_ntu','outlet_ntu'],color_discrete_sequence=['#17699c','#159781'],labels={'index':'Sample number','value':'Turbidity (NTU)'}),use_container_width=True); st.plotly_chart(px.line(df.reset_index(),x='index',y='flow_lpm',labels={'index':'Sample number','flow_lpm':'Flow (L/min)'}),use_container_width=True); st.dataframe(df,use_container_width=True); st.download_button('Export history CSV',df.to_csv(index=False),'guardian_history.csv');
        if st.button('Clear history'): st.session_state.history=[]; st.rerun()
    else: st.caption('Record samples to populate trends.')
elif page=='Batch data analysis':
    st.subheader('Analyze a sensor CSV'); st.caption('Required columns: inlet_ntu, outlet_ntu, flow_lpm. Rows are assumed to be time-aligned.'); st.download_button('Download example (synthetic)',pd.DataFrame(EXAMPLES.values(),columns=COLS).to_csv(index=False),'example_simulated.csv'); source=st.radio('Declare dataset origin',['Simulated / demonstration','User-provided experimental measurements']); f=st.file_uploader('CSV file',type=['csv'])
    if f is not None:
        try:
            df=pd.read_csv(f,nrows=10001)
            if len(df)>10000 or not set(COLS).issubset(df.columns) or df.empty: raise ValueError('Use 1–10,000 rows and all three required columns.')
            result=pd.DataFrame([diagnose(m,row) for row in df[COLS].itertuples(index=False,name=None)]); result.insert(0,'source',source); st.dataframe(result,use_container_width=True); st.plotly_chart(px.histogram(result,x='diagnosis'),use_container_width=True); st.download_button('Export diagnoses',result.to_csv(index=False),'guardian_batch_results.csv')
        except Exception as e: st.error(f'Cannot analyze CSV: {e}')
else:
    st.subheader('Reproducible synthetic benchmark'); st.info('Independent random seeds, same assumed data-generating distributions. This tests implementation, not generalization to real plants or superiority over threshold alarms.')
    a,b,c=st.columns(3); a.metric('Training samples',len(tr)); b.metric('Held-out synthetic samples',len(te)); c.metric('Random Forest trees',m.n_estimators)
    a,b=st.columns(2); a.metric('RF synthetic accuracy',f'{accuracy_score(te.label,pred):.1%}'); b.metric('Baseline synthetic accuracy',f'{accuracy_score(te.label,bp):.1%}')
    matrix=pd.DataFrame(confusion_matrix(te.label,pred,labels=LABELS),index=LABELS,columns=LABELS); st.write('Confusion matrix — rows actual, columns predicted'); st.dataframe(matrix); st.dataframe(pd.DataFrame(classification_report(te.label,pred,output_dict=True,zero_division=0)).T); st.write('Feature importance (impurity-based; not causal explanation)'); st.bar_chart(pd.Series(m.feature_importances_,index=FEATURES)); st.download_button('Training dataset',tr.to_csv(index=False),'synthetic_train.csv'); st.download_button('Held-out dataset',te.to_csv(index=False),'synthetic_test.csv'); st.markdown('**Next steps:** calibrate sensors, collect labeled normal and induced-fault runs with safe simulated wastewater, split by experimental run, compare false alarms and missed faults against baseline, and validate on independent runs. No automatic discharge or plant actuation is implemented.'); st.caption('Model vote share is not calibrated confidence. Sensor fault checks detect invalid values, not all sensor failures. Low flow may be caused by several mechanisms.')
