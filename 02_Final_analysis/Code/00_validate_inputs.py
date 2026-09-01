"""Validate deidentified inputs without original participant IDs or SPSS files."""
import hashlib
import json
import numpy as np
import pandas as pd
from config import (BEHAVIOR_INPUT, BRAINSEGVOL_INPUT, IMAGING_COVARIATES_INPUT,
                    PREDEFINED_ROI_INPUT, WHOLE_BRAIN_INPUT, DEMOGRAPHICS_INPUT, LOG_DIR)
from utils import ensure_dirs, read_csv_numeric, validate_unique_subjects


def validate():
    paths = {'behavior': BEHAVIOR_INPUT, 'covariates': IMAGING_COVARIATES_INPUT,
             'brainsegvol': BRAINSEGVOL_INPUT, 'radiomics': PREDEFINED_ROI_INPUT,
             'whole_brain': WHOLE_BRAIN_INPUT, 'demographics': DEMOGRAPHICS_INPUT}
    data, report = {}, {'files': {}, 'checks': [], 'warnings': []}
    def same(name, a, b):
        if not np.allclose(a, b, equal_nan=True, rtol=1e-9, atol=1e-9):
            raise ValueError(name)
        report['checks'].append(name)
    reference = None
    for name, path in paths.items():
        df = read_csv_numeric(path).rename(columns={'studnr': 'subject_id'})
        validate_unique_subjects(df)
        if len(df) != 95 or not df.subject_id.str.fullmatch(r'XTC\d{3}').all():
            raise ValueError(f'{name}: expected 95 deidentified XTCnnn IDs')
        ids = set(df.subject_id)
        if reference is None: reference = ids
        if ids != reference: raise ValueError(f'{name}: participant IDs do not align')
        data[name] = df.set_index('subject_id').sort_index()
        report['files'][name] = {'file': path.name, 'rows': len(df), 'columns': len(df.columns),
                                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    b, cov, brain, dem = (data[k] for k in ['behavior','covariates','brainsegvol','demographics'])
    if b[['xpillen3','sex','vwrec','bvwrec']].isna().any().any():
        raise ValueError('Missing required behavioral values')
    if (b.xpillen3 < 0).any(): raise ValueError('Negative dose')
    if not np.array_equal(b.xtc_group, np.where(b.xpillen3 > 0,'XTC users','XTC-naive')):
        raise ValueError('Exposure group mismatch')
    same('Exposure indicator', b.xtc_exposure, (b.xpillen3 > 0).astype(int))
    same('Behavioral log dose', b.log1p_xpillen3, np.log1p(b.xpillen3))
    for outcome, pre, post in [('immediate','vwsom','bvwsom'),('delayed','vwrec','bvwrec'),('recognition','vwherk','bvwherk')]:
        same(outcome+' change arithmetic', b[f'ravlt_{outcome}_change'], b[post]-b[pre])
    same('Recognition decline', b.ravlt_recognition_decline, (b.bvwherk<b.vwherk).astype(int))
    pairs = [('sex','sex'),('dart_iq','iq'),('xpillen3','xlttot_sessie3'),('vwrec','vwrec_pre'),
             ('ravlt_delayed_change','vwrec_delta'),('log1p_cannabis','lca1jt'),
             ('log1p_tobacco','lsigpw'),('log1p_alcohol','lalupw'),
             ('log1p_amphetamine','ls1jht'),('log1p_cocaine','lco1jt')]
    for name in ['covariates','radiomics','whole_brain']:
        d = data[name]
        for x,y in pairs: same(name+': '+y+' agrees with behavior', d[y], b[x])
        same(name+': log dose', d.log1p_xtc, np.log1p(d.xlttot_sessie3))
        same(name+': imaging age consistency', d.age, cov.age)
        same(name+': BrainSegVol mm3/cm3', d['aseg+DKT_BrainSegVol_pre'], brain['aseg+DKT_BrainSegVol_pre']/1000)
    same('Demographic sex', dem.sex_baseline, b.sex)
    same('Demographic dose', dem.XTC_cumulative_tablets_followup, b.xpillen3)
    if not np.array_equal(dem.xtc_group,b.xtc_group): raise ValueError('Demographic group mismatch')
    report['warnings'].append(f'Imaging age missing in {int(cov.age.isna().sum())} rows; model-stage imputation unchanged.')
    report['warnings'].append(f'Behavioral/imaging ages differ in {int((~np.isclose(b.age,cov.age,equal_nan=True)).sum())} rows including missingness; source values retained.')
    for name in ['radiomics','whole_brain']:
        d = data[name]
        columns = [c for c in d if c.endswith('_pre') and c not in ['vwrec_pre','aseg+DKT_BrainSegVol_pre']]
        for c in columns:
            partner = c[:-4]+'_delta'
            if partner not in d: raise ValueError(f'{name}: missing {partner}')
            if not np.isfinite(d[[c,partner]].to_numpy(dtype=float)).all():
                raise ValueError(f'{name}: non-finite paired feature {c}')
        unpaired = [c for c in d if c.endswith('_delta') and c!='vwrec_delta' and c[:-6]+'_pre' not in d]
        if unpaired: report['warnings'].append(f'{name}: unpaired features excluded by existing PCA pairing: {unpaired}')
        report['files'][name]['paired_features'] = len(columns)
    report['status'] = 'passed'
    return report


def main():
    report = validate()
    ensure_dirs([LOG_DIR])
    (LOG_DIR/'input_validation.json').write_text(json.dumps(report,indent=2))
    pd.DataFrame(report['files']).T.to_csv(LOG_DIR/'input_manifest.csv')
    print('Validation passed: six CSV inputs, 95 aligned participants; no original IDs or SPSS needed.')
    for warning in report['warnings']: print('NOTE:',warning)


if __name__ == '__main__': main()
