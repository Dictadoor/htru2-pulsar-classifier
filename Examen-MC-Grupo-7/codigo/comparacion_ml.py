"""Amplía HTRU2. Ejecutar: python comparacion_autonomo.py.
Selección por Balanced Accuracy de validación, desempate F1 y orden de candidatos.
La prueba ya fue observada en la práctica previa; no se usa para seleccionar.
"""
import hashlib
import json
import platform
import time
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, RocCurveDisplay, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from modelo_random_forest import ROOT, DATA_FILE, FEATURES, TARGET, RANDOM_STATE, validate_data

OUT = ROOT / 'resultados' / 'comparacion_ml'
REFERENCE = ROOT / 'modelo_historico'

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def protected_files():
    return [ROOT/'app.py', ROOT/'modelo_random_forest.py', DATA_FILE, *sorted(REFERENCE.glob('*'))]

def prepare_data():
    """Comprensión e ingeniería de datos: reutiliza esquema y controles."""
    data = pd.read_csv(DATA_FILE)
    quality = validate_data(data)
    x, y = data[FEATURES].astype(float), data[TARGET].astype(int)
    xt,xh,yt,yh = train_test_split(x,y,test_size=.30,stratify=y,random_state=RANDOM_STATE)
    xv,xp,yv,yp = train_test_split(xh,yh,test_size=.50,stratify=yh,random_state=RANDOM_STATE)
    splits = {'entrenamiento':(xt,yt),'validación':(xv,yv),'prueba':(xp,yp)}
    assert not (set(xt.index)&set(xv.index) or set(xt.index)&set(xp.index) or set(xv.index)&set(xp.index))
    assert len(xt)+len(xv)+len(xp)==len(data)
    for part,(xx,yy) in splits.items():
        pd.DataFrame({'fila_csv':xx.index,'class':yy.to_numpy()}).to_csv(OUT/f'indices_{part}.csv',index=False)
    return quality,splits

def metrics(y,pred,prob):
    """Calidad: métricas positivas para clase 1 y especificidad de clase 0."""
    tn,fp,fn,tp = confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(exactitud=accuracy_score(y,pred),exactitud_balanceada=balanced_accuracy_score(y,pred),
        precision=precision_score(y,pred,zero_division=0),recall=recall_score(y,pred,zero_division=0),
        f1=f1_score(y,pred,zero_division=0),especificidad=float(tn/(tn+fp)),roc_auc=roc_auc_score(y,prob),
        matriz_confusion=[[int(tn),int(fp)],[int(fn),int(tp)]])

def predict(model,x):
    return model.predict(x),model.predict_proba(x)[:,list(model.classes_).index(1)]

def select_on_validation(candidates,splits,experiment):
    """Modelado: entrenamiento ajusta; validación selecciona; prueba no participa."""
    xt,yt=splits['entrenamiento']; xv,yv=splits['validación']
    ranked=[]
    for config,model in candidates:
        start=time.perf_counter(); model.fit(xt,yt); seconds=time.perf_counter()-start
        pred,prob=predict(model,xv); result=metrics(yv,pred,prob)
        ranked.append((result['exactitud_balanceada'],result['f1'],model,config,seconds,result))
    best=max(ranked,key=lambda item:(item[0],item[1]))
    trials=[dict(experimento=experiment,parametros=i[3],segundos_fit=i[4],seleccionado=i is best,**i[5]) for i in ranked]
    return best[2],best[3],best[4],trials

def complexity(model):
    """Complejidad observada: nodos y profundidad, o coeficientes e iteraciones."""
    if hasattr(model,'estimators_'):
        return dict(arboles=len(model.estimators_),nodos=sum(t.tree_.node_count for t in model.estimators_),
                    profundidad_maxima=max(t.tree_.max_depth for t in model.estimators_))
    lr=model.named_steps['logistica']
    return dict(coeficientes=int(lr.coef_.size+lr.intercept_.size),iteraciones=int(lr.n_iter_.max()))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    before={str(p.relative_to(ROOT)):sha256(p) for p in protected_files() if p.is_file()}
    quality,splits=prepare_data()
    # Referencia histórica: cargar; no reentrenar ni aplicar escalado posterior.
    baseline=joblib.load(REFERENCE/'modelo_random_forest.joblib')
    assert isinstance(baseline,RandomForestClassifier) and baseline.n_estimators==300
    saved_params=json.loads((REFERENCE/'metricas.json').read_text(encoding='utf-8'))['parametros_modelo']
    assert baseline.get_params()==saved_params
    # Verificar correspondencia con predicciones y probabilidades archivadas.
    old=pd.read_csv(REFERENCE/'predicciones_prueba.csv').set_index('fila_csv')
    xp,yp=splits['prueba']; pred,prob=predict(baseline,xp); aligned=old.loc[xp.index]
    assert np.array_equal(pred,aligned['clase_predicha'])
    assert np.array_equal(yp.to_numpy(),aligned['clase_real'])
    assert np.allclose(prob,aligned['probabilidad_candidato_pulsar'],rtol=0,atol=1e-14)
    rf_candidates=[]
    for depth in [5,10,15]:
        config={**saved_params,'n_estimators':150,'max_depth':depth}
        rf_candidates.append((config,RandomForestClassifier(**config)))
    lr_candidates=[]
    for c in [.01,.1,1.,10.]:
        lr=LogisticRegression(C=c,solver='lbfgs',max_iter=2000,tol=1e-4,class_weight='balanced',random_state=42)
        lr_candidates.append((lr.get_params(),Pipeline([('estandarizador',StandardScaler()),('logistica',lr)])))
    rf,rf_config,rf_time,rf_trials=select_on_validation(rf_candidates,splits,'RF modificado')
    lr,lr_config,lr_time,lr_trials=select_on_validation(lr_candidates,splits,'Regresión logística')
    selected={'RF original':saved_params,'RF modificado':rf_config,'Regresión logística':lr_config}
    # Congelar la elección antes de calcular métricas de prueba de alternativas.
    (OUT/'seleccion_validacion.json').write_text(json.dumps(dict(criterio='Balanced Accuracy; desempate F1; orden declarado',candidatos=rf_trials+lr_trials,seleccion=selected),ensure_ascii=False,indent=2),encoding='utf-8')
    models={'RF original':baseline,'RF modificado':rf,'Regresión logística':lr}
    rows=[]; structures=[]
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    cmfig,cmax=plt.subplots(1,3,figsize=(12,3.6))
    for i,(name,model) in enumerate(models.items()):
        model_dir=OUT/f'experimento_{i+1}'; model_dir.mkdir(exist_ok=True)
        if i: joblib.dump(model,model_dir/'modelo.joblib')
        structures.append(dict(modelo=name,segundos_fit=[None,rf_time,lr_time][i],**complexity(model)))
        individual,iaxes=plt.subplots(1,2,figsize=(9,3.8))
        for part,(xx,yy) in splits.items():
            pred,prob=predict(model,xx); result=metrics(yy,pred,prob)
            rows.append(dict(modelo=name,particion=part,n=len(yy),**result))
            pd.DataFrame({'fila_csv':xx.index,'clase_real':yy.to_numpy(),'clase_predicha':pred,'probabilidad_clase_1':prob}).to_csv(model_dir/f'predicciones_{part}.csv',index=False)
            pd.DataFrame(result['matriz_confusion'],index=['Real 0','Real 1'],columns=['Predicho 0','Predicho 1']).to_csv(model_dir/f'matriz_{part}.csv')
            if part in ('entrenamiento','prueba'):
                j=0 if part=='entrenamiento' else 1
                RocCurveDisplay.from_predictions(yy,prob,name=name,ax=axes[j])
                RocCurveDisplay.from_predictions(yy,prob,name=part,ax=iaxes[j])
            if part=='prueba':
                ConfusionMatrixDisplay(np.array(result['matriz_confusion']),display_labels=['0','1']).plot(ax=cmax[i],colorbar=False,cmap='Blues')
                cmax[i].set_title(name)
        for ax,title in zip(iaxes,['Entrenamiento','Prueba']):
            ax.plot([0,1],[0,1],'--',color='gray'); ax.set(title=title,xlabel='Tasa de falsos positivos',ylabel='Tasa de verdaderos positivos')
        individual.tight_layout(); individual.savefig(model_dir/'curvas_roc.png',dpi=220); plt.close(individual)
    for ax,title in zip(axes,['Entrenamiento','Prueba']):
        ax.plot([0,1],[0,1],'--',color='gray'); ax.set(title=title,xlabel='Tasa de falsos positivos',ylabel='Tasa de verdaderos positivos'); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT/'curvas_roc.png',dpi=220); plt.close(fig)
    cmfig.tight_layout(); cmfig.savefig(OUT/'matrices_confusion.png',dpi=220); plt.close(cmfig)
    pd.DataFrame(rows).drop(columns='matriz_confusion').to_csv(OUT/'metricas_comparacion.csv',index=False,encoding='utf-8-sig')
    after={str(p.relative_to(ROOT)):sha256(p) for p in protected_files() if p.is_file()}
    assert before==after,'Se modificó un archivo protegido'
    result=dict(python=platform.python_version(),sklearn=sklearn.__version__,auditoria=quality,dataset_sha256=sha256(DATA_FILE),
        seleccion=selected,resultados=rows,complejidad=structures,integridad_referencia=before,
        limitacion='Prueba consultada en prácticas anteriores; selección actual solo mediante validación.')
    (OUT/'resultados.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).drop(columns='matriz_confusion').to_string(index=False))
    print('Referencia y dashboard preservados; salidas:',OUT)

if __name__=='__main__':
    main()

