"""Original no-repair trace auditor; no native environment import is required."""
def project_steps(executable):
    return [{"skill":step["skill"], **({"target":step["target"]["number"]}
        if isinstance(step.get("target"),dict) else {})} for step in executable]

def audit(result, expected):
    trace = result.get('trace', [])
    observed = project_steps(trace)
    wanted = project_steps(expected)
    identity = observed == wanted
    prefix = observed == wanted[:len(observed)]
    stage_exception = bool(result.get('stage_exception_policy', {}).get('stage_exception_caught'))
    complete = bool(result.get('execution_completed')) and identity and (not stage_exception) and (not result.get('native_timeout')) and (not result.get('infrastructure_error'))
    flags = [step.get('stage_success') for step in trace]
    all_stages = bool(expected) and complete and (len(flags) == len(expected)) and all((flag is True for flag in flags))
    goal = bool(result.get('condition_success'))
    return {'raw_final_goal': goal, 'complete_without_exception': complete, 'full_trace_identity': identity, 'trace_is_original_prefix': prefix, 'observed_steps': observed, 'expected_steps': wanted, 'stage_success_flags': flags, 'all_stages_success': all_stages, 'stage_false_count': sum((flag is False for flag in flags)), 'goal_after_complete_execution': complete and goal, 'goal_and_all_stages_success': all_stages and goal, 'goal_with_stage_failure': complete and goal and any((flag is False for flag in flags)), 'stage_exception': stage_exception, 'native_timeout': bool(result.get('native_timeout')), 'infrastructure_error': bool(result.get('infrastructure_error')), 'pre_action_load_failure': bool(result.get('native_exception') and 'trace' not in result), 'native_exception': result.get('exception', ''), 'stage_exception_detail': result.get('stage_exception_policy', {}).get('exception', '')}
