"""Recheck every saved relaxed-contract trace without a model or simulator."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from trace_audit import audit, project_steps

HERE=Path(__file__).resolve().parent


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=HERE/'reproduced')
    args=parser.parse_args()
    out=args.output_dir.resolve()
    assert not out.exists(),'Choose a new output directory; archived records are never overwritten.'
    out.mkdir(parents=True)
    protocol=load(HERE/'frozen_protocol.json')
    counts=Counter(selected=len(protocol['cases']))
    checked=[]
    for case in protocol['cases']:
        sample=case['audit_id']
        saved=load(HERE/'case_results'/f'{sample}.json')
        if saved['dispatch_category']=='unsupported_dispatch':
            assert not saved['native_attempted'] and not saved['verified_goal']
            counts['unsupported']+=1
            continue
        assert saved['native_attempted']
        counts['native_attempted']+=1
        arms={}
        for name in ('initial_no_action','canonical_control','alternative'):
            directory=HERE/'traces'/sample/name
            command=load(directory/'native_command.json')
            raw=load(directory/'captured_result.json')
            expected=command['executable_steps']
            if name=='alternative':
                assert project_steps(expected)==case['original_steps']
            result=audit(raw,expected)
            reference=saved['arms'][name]
            for field,value in result.items():
                assert reference[field]==value,(sample,name,field)
            arms[name]=result
            counts['native_calls']+=1
        initial=arms['initial_no_action']
        assert initial['complete_without_exception'] and not initial['raw_final_goal']
        goal=arms['alternative']['goal_after_complete_execution'] and not initial['raw_final_goal']
        all_stage=goal and arms['alternative']['all_stages_success']
        assert goal==saved['verified_goal']
        assert all_stage==saved['verified_goal_all_stages']
        assert (goal and arms['alternative']['stage_false_count']>0)==saved['verified_goal_with_stage_failure']
        canonical=arms['canonical_control']['goal_after_complete_execution']
        assert canonical==saved['canonical_goal']
        counts['goal']+=int(goal)
        counts['all_stage_goal']+=int(all_stage)
        counts['goal_with_false_stage']+=int(goal and arms['alternative']['stage_false_count']>0)
        counts['reference_goal']+=int(canonical)
        counts['alternative_complete']+=int(arms['alternative']['complete_without_exception'])
        checked.append({'audit_id':sample,'case_id':case['case_id'],'model':case['model'],
                        'goal':goal,'all_stage_goal':all_stage,'canonical_goal':canonical})
    expected={'selected':86,'unsupported':5,'native_attempted':81,'native_calls':243,
              'goal':5,'all_stage_goal':3,'goal_with_false_stage':2,'reference_goal':74,'alternative_complete':66}
    assert dict(counts)==expected,(dict(counts),expected)
    with (out/'case_recheck.csv').open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(checked[0]))
        writer.writeheader()
        writer.writerows(checked)
    receipt={'counts':dict(counts),'original_model_choices_preserved':True,
             'new_inference_calls':0,'new_native_calls':0,'historical_B_labels_changed':False}
    (out/'VERIFIED.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt))


if __name__=='__main__':
    main()
