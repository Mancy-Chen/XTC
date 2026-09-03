from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import t, spearmanr

from config import PROJECT_ROOT, OUTPUT_DIR
root=PROJECT_ROOT
old=OUTPUT_DIR
tables={}; notes={}; titles={}
def fmt(x,n=3): return f'{float(x):.{n}f}'.replace('-','−')
def dec(x,n=3):
 s=fmt(x,n);return s.replace('−0.','−.').replace('0.','.',1) if abs(x)<1 else s
def prob(x):return '<.001' if x<.001 else dec(x)
def small(x):return f'{x:.3e}'.replace('-','−') if 0<abs(x)<.0005 else fmt(x)
def read(p):
 d=pd.read_csv(p)
 for c in d:
  if c not in ['subject_id','xtc_group']:d[c]=pd.to_numeric(d[c].astype(str).str.replace(',','.'),errors='coerce')
 return d
def fit(d,pc,poly=True):
 d=d.copy();d.age=d.age.fillna(d.age.mean())
 cols=[pc+'_delta',pc+'_pre','vwrec_pre','aseg+DKT_BrainSegVol_pre','log1p_xtc','age','iq']
 if poly:cols+=['lca1jt','lsigpw','lalupw','ls1jht','lco1jt']
 names=['delta','baseline','memory','brain','dose','age','iq']+(['cannabis','tobacco','alcohol','amphetamine','cocaine'] if poly else [])
 a=d[cols].to_numpy();a-=a.mean(0);X=np.column_stack([np.ones(len(d)),a,(d.sex==2).astype(float)]);names=['intercept']+names+['sex']
 y=(d.vwrec_pre+d.vwrec_delta).to_numpy();assert np.isfinite(X).all() and np.isfinite(y).all()
 b=np.linalg.lstsq(X,y,rcond=None)[0];res=y-X@b;df=len(y)-np.linalg.matrix_rank(X);se=np.sqrt(np.diag(np.linalg.inv(X.T@X))*(res@res)/df);stat=b/se;p=2*t.sf(abs(stat),df);lo=b-t.ppf(.975,df)*se;hi=b+t.ppf(.975,df)*se
 std=b*X.std(0,ddof=1)/y.std(ddof=1);r2=1-(res@res)/sum((y-y.mean())**2)
 result={k:dict(B=b[i],SE=se[i],Beta=std[i],t=stat[i],p=p[i],lo=lo[i],hi=hi[i]) for i,k in enumerate(names)}
 return result,r2,1-(1-r2)*(len(y)-1)/df

direct=pd.read_csv(old/'LMM/predefined_roi_voxelvolume/table_direct_group_change_difference.csv')
tables[3]=[['Outcome','Users − naive difference in change (cm³)','95% CI','p','FDR q']]+[[r.outcome,fmt(r.estimate_cm3),f'{fmt(r.ci_low)} to {fmt(r.ci_high)}',prob(r.p),prob(r.FDR_q)] for r in direct.itertuples()]
v=pd.read_csv(root/'Output/PCA/predefined_roi/predefined_roi_pca_explained_variance.csv')
tables[5]=[['Feature set','Component','Explained variance','Cumulative variance']]
for name,label in [('Combined_shape_only','Combined shape-only'),('Combined_shape_firstorder','Combined shape + first-order')]:
 for r in v[v.analysis.eq(name)].itertuples():tables[5].append([label,r.component,fmt(r.explained_variance_ratio),fmt(r.cumulative_explained_variance_ratio)])
terms=[('time','time'),('log(1 + XTC dose)','log_dose_c'),('time × log(1 + XTC dose)','time:log_dose_c'),('age','age_c'),('sex: female vs male','C(sex)[T.2]'),('Baseline BrainSegVol (cm³)','BrainSegVol_c')]
sub=[('cannabis','cannabis_c'),('alcohol','alcohol_c'),('tobacco','tobacco_c'),('amphetamine','amphetamine_c'),('cocaine','cocaine_c')]
for num,name,poly in [(6,'Combined_shape_only',False),(7,'Combined_shape_only',True),(8,'Combined_shape_firstorder',False),(9,'Combined_shape_firstorder',True)]:
 d=pd.read_csv(old/('LMM/predefined_roi_pca/predefined_roi_pca_lmm_'+('polysubstance' if poly else 'primary')+'_all_terms.csv'));d=d[d.analysis.eq(name)].set_index('term')
 tables[num]=[['Term','B','SE','z','p']]
 for label,term in terms+(sub if poly else []):
  r=d.loc[term];tables[num].append([label,fmt(r.beta),fmt(r.std_error),fmt(r.statistic),prob(r.p)])

d=read(root/'Output/PCA/predefined_roi/predefined_roi_pca_scores_wide.csv')
labels=[('ΔPC1','delta'),('Baseline PC1','baseline'),('Baseline delayed recall','memory'),('Baseline BrainSegVol (cm³)','brain'),('ln(1 + XTC dose)','dose'),('Age','age'),('Sex: female vs male','sex'),('IQ','iq'),('Cannabis','cannabis'),('Alcohol','alcohol'),('Tobacco','tobacco'),('Amphetamine','amphetamine'),('Cocaine','cocaine')]
tables[11]=[['Radiomics PCA','Model','N','ΔPC1 B','SE','t','p','R²','Adjusted R²']]
for pref,label,num in [('Combined_shape_only','Combined shape-only',12),('Combined_shape_firstorder','Combined shape + first-order',13)]:
 for poly in [False,True]:
  coef,r2,ar2=fit(d,pref+'_PC1',poly);r=coef['delta'];tables[11].append([label,('With' if poly else 'Without')+' polysubstance covariates','95',fmt(r['B']),fmt(r['SE']),fmt(r['t']),prob(r['p']),dec(r2),dec(ar2)])
  if poly:
   tables[num]=[['Predictor','B','SE','Beta','t','p']]+[[lab,small(coef[key]['B']),small(coef[key]['SE']),fmt(coef[key]['Beta']),fmt(coef[key]['t']),prob(coef[key]['p'])] for lab,key in labels]
   notes[num]=f'N = 95. R² = {dec(r2)}; adjusted R² = {dec(ar2)}. The outcome was follow-up delayed recall. B is unstandardized; Beta is fully standardized. Continuous predictors were mean-centered. BrainSegVol was measured in cm³; substance-use covariates were ln(1 + x)-transformed.'
titles[11]='Supplementary Table S11. Regression summary for follow-up delayed recall using predefined-ROI radiomics PC1.'
titles[12]='Supplementary Table S12. Fully adjusted follow-up delayed-recall regression using shape-only radiomics PC1.'
titles[13]='Supplementary Table S13. Fully adjusted follow-up delayed-recall regression using shape-plus-first-order radiomics PC1.'

w=read(root/'Output/PCA/whole_brain/whole_brain_pca_scores_wide.csv')
v=pd.read_csv(root/'Output/PCA/whole_brain/whole_brain_pca_explained_variance.csv')
tables[14]=[['Component','Explained variance','Cumulative variance']]+[[r.component,f'{100*r.explained_variance_ratio:.2f}%',f'{100*r.cumulative_explained_variance_ratio:.2f}%'] for r in v.head(5).itertuples()]
lm=pd.read_csv(old/'LMM/whole_brain_pca/whole_brain_pca_lmm_all_terms.csv');tables[15]=[['Model','Predictor','B','SE','95% CI','p']]
for model,label in [('primary','Base model'),('polysubstance_adjusted','Polysubstance-adjusted model')]:
 for i,(lab,term) in enumerate([('Intercept','Intercept')]+terms+(sub if model!='primary' else [])):
  r=lm[lm.model_type.eq(model)&lm.term.eq(term)].iloc[0];tables[15].append([label if i==0 else '',lab,fmt(r.beta),fmt(r.std_error),f'{fmt(r.ci_low)} to {fmt(r.ci_high)}',prob(r.p)])
raw=w.PC1_delta.to_numpy();X=np.column_stack([np.ones(len(w)),w[['PC1_pre','aseg+DKT_BrainSegVol_pre']]]);adj=raw-X@np.linalg.lstsq(X,raw,rcond=None)[0]
tables[16]=[['Group','N','Spearman ρ','p']]
for label,mask in [('XTC-naive',w.xlttot_sessie3.eq(0)),('XTC users',w.xlttot_sessie3.gt(0)),('Overall',np.ones(len(w),bool))]:
 vals=[spearmanr(a[mask],w.loc[mask,'vwrec_delta']) for a in [raw,adj]]
 tables[16].append([label,str(sum(mask)),f'Raw: {dec(vals[0].statistic)}\nAdjusted: {dec(vals[1].statistic)}',f'Raw: {prob(vals[0].pvalue)}\nAdjusted: {prob(vals[1].pvalue)}'])
notes[16]='Raw and adjusted correlations are reported for each group. Adjusted PC1 change was residualized for baseline PC1 and baseline BrainSegVol across the full sample. PCA was fitted at baseline, and follow-up observations were projected into the baseline-defined space.'
coef,r2,ar2=fit(w,'PC1');tables[17]=[['Predictor','B','SE','95% CI','Conventional p']]+[[lab,small(coef[key]['B']),small(coef[key]['SE']),f"{fmt(coef[key]['lo'])} to {fmt(coef[key]['hi'])}",prob(coef[key]['p'])] for lab,key in [('Intercept','intercept')]+labels]
notes[17]=f'N = 95; R² = {dec(r2)}; adjusted R² = {dec(ar2)}; overall model p < .001. The outcome was follow-up delayed recall. B values are unstandardized. Continuous predictors were mean-centered; sex was categorical. BrainSegVol was measured in cm³; substance-use covariates were ln(1 + x)-transformed.'
titles[17]='Supplementary Table S17. Fully adjusted regression predicting follow-up delayed recall from whole-brain PC1.'

b=pd.read_csv(root/'Output/bootstrap/whole_brain_pca/whole_brain_pca_bootstrap_summary.csv').set_index('measure')
tables[18]=[['Measure','Original','Bootstrap median','95% percentile interval']]
for metric,label,kind in [('PC1_explained_variance','PC1 explained variance','percent'),('loading_cosine_similarity','Loading-vector cosine similarity','corr'),('baseline_PC1_score_correlation','Baseline PC1-score correlation','corr'),('delta_PC1_correlation','ΔPC1 correlation','corr'),('original_top20_overlap','Original top-20 overlap','count'),('time_by_dose_coefficient','Unadjusted XTC dose–ΔPC1 slope','slope')]:
 r=b.loc[metric]
 def show(v):return f'{100*v:.2f}%' if kind=='percent' else (f'{v:.0f}' if kind=='count' else fmt(v,4))
 tables[18].append([label,show(r.original)+('/20' if kind=='count' else ''),show(r.bootstrap_median)+('/20' if kind=='count' else ''),f'{show(r.percentile_2_5)} to {show(r.percentile_97_5)}'])
tables[19]=[['Analysis','Original ρ','Bootstrap median','PCA-bootstrap interval']]
for group,label in [('overall','Overall'),('naive','XTC-naive'),('users','XTC users')]:
 r=b.loc['raw_rho_'+group];tables[19].append([label,dec(r.original),dec(r.bootstrap_median),f'{dec(r.percentile_2_5)} to {dec(r.percentile_97_5)}'])
notes[18]='Intervals summarize PCA-refitting variability across 1,000 baseline bootstrap resamples after sign alignment; they are not confidence intervals for population exposure effects. The dose–ΔPC1 slope was estimated by regressing PC1 change on ln(1 + XTC dose) without covariates.'
notes[19]='Intervals reflect variation from refitting PCA across baseline bootstrap samples; they are not confidence intervals for population correlations.'

# Normality and correlation tables retain the current manuscript numbering.
normal=pd.read_csv(old/'correlations/predefined_roi_voxelvolume/normality_tests.csv')
tables[4]=[['Variable','N','Shapiro–Wilk W','p','Interpretation']]
for r in normal.itertuples():
 if r.column=='vwrec_delta' or r.column.startswith('adjusted_'):
  tables[4].append([r.variable,str(r.N),fmt(r.Shapiro_W),prob(r.p),'Evidence of non-normality' if r.p<.05 else 'No evidence of non-normality'])
c=pd.read_csv(old/'correlations/predefined_roi_pca/table_combined_pc1_raw_adjusted_group_spearman.csv')
tables[10]=[['Feature set','Group','N','Raw rho','Raw p','Raw FDR q','Adjusted rho','Adjusted p','Adjusted FDR q']]
for name,label in [('Combined_shape_only','Shape-only'),('Combined_shape_firstorder','Shape + first-order')]:
 for group in ['XTC-naive','XTC users']:
  rr=c[c.analysis.eq(name)&c['sample'].eq(group)&c.adjustment.eq('raw')].iloc[0]
  aa=c[c.analysis.eq(name)&c['sample'].eq(group)&c.adjustment.eq('adjusted')].iloc[0]
  tables[10].append([label,group,str(rr.N),dec(rr.Spearman_rho),prob(rr.p),prob(rr.FDR_q_across_2_tests),dec(aa.Spearman_rho),prob(aa.p),prob(aa.FDR_q_across_2_tests)])
titles.update({3:'Direct group differences in volume change',4:'Normality of memory and adjusted volume changes',5:'Predefined-ROI PCA explained variance',6:'Shape-only primary PC1 mixed model',7:'Shape-only sensitivity PC1 mixed model',8:'Shape-plus-first-order primary PC1 mixed model',9:'Shape-plus-first-order sensitivity PC1 mixed model',10:'Predefined-ROI PC1 correlations with delayed-recall change',14:'Whole-brain PCA explained variance',15:'Whole-brain PC1 mixed models',16:'Whole-brain PC1 change and delayed-recall change',18:'Whole-brain PCA stability',19:'Stability of raw whole-brain PC1 memory correlations'})
notes[10]='BH FDR correction was applied across the two predefined-ROI PCA feature sets, separately within each sample and adjustment type. Adjusted PC1 changes were residualized for baseline PC1 and baseline BrainSegVol.'
for num in [6,7,8,9,15]:
 notes[num]='Participant-specific random-intercept mixed model. BrainSegVol is the baseline whole-brain volume covariate (cm³). Continuous predictors were mean-centered; XTC dose and other substance-use covariates were ln(1 + x)-transformed.'
dest=old/'manuscript_tables';dest.mkdir(parents=True,exist_ok=True)
manifest=pd.read_csv(old/'PCA/predefined_roi/predefined_roi_pca_feature_manifest.csv')
included=manifest.loc[manifest.included.eq(True)]
shape=int(included.analysis.eq('Combined_shape_only').sum())
combined=int(included.analysis.eq('Combined_shape_firstorder').sum())
(dest/'Table_S01_note.md').write_text(
 f'The shape-only PCA included {shape} features across the four ROIs. '
 f'The combined shape-plus-first-order PCA included {combined} features: '
 f'{shape} shape and {combined-shape} first-order features. '
 'The left-hippocampal first-order 10th-percentile feature was excluded because its baseline counterpart was unavailable.\n'
)
for num,rows in sorted(tables.items()):
 pd.DataFrame(rows[1:],columns=rows[0]).to_csv(dest/f'Table_S{num:02d}.csv',index=False)
 heading=titles.get(num,f'Supplementary Table S{num}')
 clean=lambda row:'| '+' | '.join(str(v).replace('\n','<br>') for v in row)+' |'
 md=heading+'\n\n'+clean(rows[0])+'\n'+clean(['---']*len(rows[0]))+'\n'+'\n'.join(clean(r) for r in rows[1:])+'\n'
 if num in notes:md+='\n'+notes[num]+'\n'
 (dest/f'Table_S{num:02d}.md').write_text(md)
import shutil
for ext in ['csv','md']:
 shutil.copyfile(old/f'demographics/Supplementary_Table_S2_full_demographics.{ext}',dest/f'Table_S02.{ext}')
json.dump(dict(tables=tables,notes=notes,titles=titles),open(dest/'table_data.json','w'),ensure_ascii=False,indent=2)
print('Current supplementary tables S2–S19 exported; S1 is the feature-class description in the manuscript.')
