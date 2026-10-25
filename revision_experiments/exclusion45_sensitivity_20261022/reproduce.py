"""Reproduce the post-audit sensitivities from frozen, released case labels."""
import argparse
import csv
import json
import math
from pathlib import Path
import re

import numpy as np
from stats_core import MODELS, contrast, scaffold_simultaneous, adjust_holm

HERE = Path(__file__).resolve().parent
BOOLS = ('retained_after_exclusion','original_pm100_success','original_strict_sequence_success',
         'projected_pm100_success','projected_strict_sequence_success',
         'projected_strict_executor_filtered_success','full_procedure_success',
         'projection_supported','phase_scaffold_success','target_scaffold_success')
CONTRASTS = (
    ('projected_pm100_success','full_procedure_success','Projected PM100 versus Full','projection_pm100'),
    ('projected_strict_sequence_success','full_procedure_success','Projected strict versus Full','projection_strict'),
    ('projected_strict_executor_filtered_success','full_procedure_success','Executor-filtered strict versus Full','projection_filtered'),
    ('original_pm100_success','full_procedure_success','Independent original PM100 versus Full','original_pm100'),
    ('original_strict_sequence_success','full_procedure_success','Independent original strict versus Full','original_strict'),
    ('original_pm100_success','projected_pm100_success','Original PM100 versus projected PM100','original_projected_pm100'),
    ('original_strict_sequence_success','projected_strict_sequence_success','Original strict versus projected strict','original_projected_strict'),
    ('phase_scaffold_success','full_procedure_success','Phase Scaffold versus Full','scaffold_six_contrasts'),
    ('target_scaffold_success','full_procedure_success','Target Scaffold versus Full','scaffold_six_contrasts'),
)


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def save_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def truth(value):
    assert value in ('True','False'),value
    return value=='True'


def verify_expected(name, rows):
    expected=read_csv(HERE/'expected'/name)
    assert len(rows)==len(expected),name
    for index,(actual,reference) in enumerate(zip(rows,expected)):
        for key,wanted in reference.items():
            value=actual.get(key)
            if wanted=='':
                assert value is None or value=='',(name,index,key,value)
                continue
            try:
                wanted_number=float(wanted)
                tolerance=1e-300 if key.endswith('_p') else 1e-10
                assert math.isclose(float(value),wanted_number,rel_tol=1e-8,abs_tol=tolerance),(name,index,key,value,wanted)
            except ValueError:
                assert str(value)==wanted,(name,index,key,value,wanted)


def verify_exclusions():
    audit=json.loads((HERE/'inputs/instruction_goal_audit.json').read_text(encoding='utf-8'))
    excluded={}
    density_order=['wood','stone','metal']
    for row in audit['density']:
        material=density_order[row['semantic_rank']]
        button=next(item['button'] for item in row['associations'] if item['material']==material)
        assert button==row['semantic_expected_button']
        if button!=row['saved_target_button']:
            excluded[row['case_id']]='density_instruction_vs_saved_button'
    for row in audit['billiards']:
        requested=re.findall(r'container_[a-z]+_[12]',row['instruction'])
        assert requested==row['requested_pocket_ids']
        if row['target_hole'] not in requested:
            excluded[row['case_id']]='billiards_instruction_vs_backend_pocket'
    expected={r['case_id']:r['reason'] for r in read_csv(HERE/'inputs/excluded_cases.csv')}
    assert excluded==expected and len(excluded)==45
    assert sum(v.startswith('density') for v in excluded.values())==5
    return excluded


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,default=HERE.parents[1])
    parser.add_argument('--output-dir',type=Path,default=HERE/'reproduced')
    args=parser.parse_args()
    out=args.output_dir.resolve()
    assert not out.exists(),'Choose a new output directory; expected and prior results are never overwritten.'
    out.mkdir(parents=True)
    exclusions=verify_exclusions()
    frozen=read_csv(HERE/'inputs/frozen_case_labels.csv')
    for row in frozen:
        for name in BOOLS:
            row[name]=truth(row[name])
        assert row['retained_after_exclusion']==(row['case_id'] not in exclusions)
    assert len(frozen)==len({(r['model'],r['case_id']) for r in frozen})==6399
    tables=[]
    for population in ('original_2133','exclude_45'):
        selected=[r for r in frozen if population=='original_2133' or r['retained_after_exclusion']]
        for model in list(MODELS.values())+['Pooled fixed-model strata']:
            rows=selected if model=='Pooled fixed-model strata' else [r for r in selected if r['model']==model]
            for left,right,name,family in CONTRASTS:
                tables.append(contrast(rows,left,right,name,model,population,family))
        scaffold_simultaneous(selected,population,tables)
    adjust_holm(tables)
    verify_expected('paired_statistics.csv',tables)
    save_csv(out/'paired_statistics.csv',tables)

    components=[]
    for key,model in MODELS.items():
        source=read_csv(args.repo_root.resolve()/'analysis_data'/f'{key}_A.csv')
        assert len(source)==len({r['case_id'] for r in source})==2133
        for population in ('original_2133','exclude_45'):
            chosen=[r for r in source if population=='original_2133' or r['case_id'] not in exclusions]
            for is_normalized,prefix in ((False,''),(True,'type_normalized_')):
                means=[np.mean([float(r[prefix+name]) for r in chosen]) for name in
                       ('skill_match_score','entity_match_score','skill_with_entity_match_score','exact_match_score')]
                components.append({'population':population,'model':model,'type_normalized':is_normalized,'n':len(chosen),
                    'SR':means[0],'PR':means[1],'SPR':means[2],'PM':means[3],
                    'released_code_total_0_80':.4*means[0]+.4*means[1],
                    'recomputed_four_metric_score_0_100':.4*means[0]+.4*means[1]+.1*means[2]+.1*means[3]})
    verify_expected('released_evaluator_metric_sensitivity.csv',components)
    save_csv(out/'released_evaluator_metric_sensitivity.csv',components)

    schema=read_csv(HERE/'inputs/schema_control_case_labels.csv')
    for row in schema:
        for key in ('raw_json_schema_valid','schema_valid','phase_target_valid','native_complete','success'):
            row[key]=truth(row[key])
    schema_contrasts=[]
    schema_counts=[]
    for population in ('original_322','exclude_overlap_45'):
        for model in MODELS.values():
            index={(r['case_id'],r['condition']):r for r in schema
                   if r['model']==model and (population=='original_322' or r['case_id'] not in exclusions)}
            ids=sorted({key[0] for key in index})
            paired=[{'task_cluster':index[cid,'json_object']['task_cluster'],
                     'object_success':index[cid,'json_object']['success'],
                     'schema_success':index[cid,'json_schema']['success']} for cid in ids]
            schema_contrasts.append(contrast(paired,'schema_success','object_success','JSON Schema versus JSON object',
                                             model,population,'schema_final_success',seed=20261021))
            for condition in ('json_object','json_schema'):
                chosen=[index[cid,condition] for cid in ids]
                schema_counts.append({'population':population,'model':model,'condition':condition,'n':len(chosen),
                    'task_clusters':len({r['task_cluster'] for r in chosen}),
                    **{key:sum(r[key] for r in chosen) for key in ('schema_valid','phase_target_valid','native_complete','success')}})
    adjust_holm(schema_contrasts)
    for name,rows in (('schema_control_paired_sensitivity.csv',schema_contrasts),('schema_control_gate_sensitivity.csv',schema_counts)):
        verify_expected(name,rows)
        save_csv(out/name,rows)
    receipt={'all_expected_tables_reproduced':True,'excluded_cases':45,'retained_cases':2088,
             'task_clusters':34,'filtered_decoding_cases':309,'new_inference_calls':0,'new_native_calls':0,
             'bootstrap_replicates':20000,'main_seed':20260711,'schema_seed':20261021}
    (out/'VERIFIED.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt))


if __name__=='__main__':
    main()
