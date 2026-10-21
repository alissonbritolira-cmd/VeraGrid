# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. https://mozilla.org/MPL/2.0/
# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations
import copy
import numpy as np
from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
from VeraGridEngine.enumerations import GeneratorControlMode
def solve_with_controls(grid, options, controls, max_outer=12, voltage_tolerance=1e-7, telemetry=None,
                        cancel_callback=None, sensitivity_update='finite_difference', jacobian_refresh_outer=3,
                        sensitivity_cache=None, numerical_polish_pu=None, reuse_branch_compilation=True,
                        compilation_cache=None):
    # The controller already owns its private numerical cache and voltage
    # history. General snapshot reuse must not introduce a second history.
    options = copy.copy(options)
    options.initialize_with_existing_solution = False
    if sensitivity_update not in ('finite_difference', 'broyden', 'analytic'):
        raise ValueError('Método de atualização de sensibilidades não reconhecido.')
    if not isinstance(jacobian_refresh_outer, int) or jacobian_refresh_outer < 1:
        raise ValueError('Intervalo de renovação da Jacobiana inválido.')
    use_broyden = sensitivity_update in ('broyden', 'analytic')
    use_analytic = use_broyden and reuse_branch_compilation
    requested_pf_tolerance = float(options.tolerance)
    requested_control_tolerance = float(voltage_tolerance)
    refinement_reason = None
    if numerical_polish_pu is None and controls and options.tolerance > voltage_tolerance:
        # O Newton pode aceitar zero iterações após uma pequena correção de Q,
        # mantendo a tensão anterior e tornando a derivada externa singular.
        # Reutiliza o refinamento existente, sem relaxar o alvo do controle.
        numerical_polish_pu = float(voltage_tolerance)
        refinement_reason = 'Precisão AC necessária ao ajuste externo de tensão.'
    if numerical_polish_pu is not None:
        numerical_polish_pu = float(numerical_polish_pu)
        if not np.isfinite(numerical_polish_pu) or not 0 < numerical_polish_pu <= min(voltage_tolerance, options.tolerance):
            raise ValueError('Precisão extra deve ser positiva e mais estrita que as tolerâncias do estudo.')
        # Critérios de aprovação permanecem no estudo. Esta precisão adicional
        # limita a dispersão numérica entre inicializações distintas da rede.
        voltage_tolerance = numerical_polish_pu
        options.tolerance = numerical_polish_pu
    if telemetry is not None:
        telemetry['sensitivity_update'] = sensitivity_update
        telemetry['pf_requested_tolerance_pu'] = requested_pf_tolerance
        telemetry['pf_effective_tolerance_pu'] = float(options.tolerance)
        telemetry['control_requested_tolerance_pu'] = requested_control_tolerance
        telemetry['control_effective_tolerance_pu'] = float(voltage_tolerance)
        if refinement_reason is not None:
            telemetry['precision_refinement_reason'] = refinement_reason
        if numerical_polish_pu is not None:
            telemetry['numerical_polish_pu'] = numerical_polish_pu
    def count(key):
        if telemetry is not None:
            telemetry[key] = telemetry.get(key, 0)+1
    indices={b.code:i for i,b in enumerate(grid.buses)}
    trace=[]
    from VeraGridEngine.IO.ons_ems.control_equations import ControlEquations
    equations = ControlEquations(grid, controls)
    enforce_local_q=options.control_Q
    options.control_Q=False
    from VeraGridEngine.IO.ons_ems.control_solver import ControlSolver
    cached_solver = ControlSolver(grid, options, telemetry, fixed_network=True,
                                   warm_start=use_analytic, compilation_cache=compilation_cache) if reuse_branch_compilation else None
    if not reuse_branch_compilation and compilation_cache is not None:
        compilation_cache.clear()
    # O solver compara a injeção líquida da barra com limites do gerador.
    # Aplicar limites à geração calculada, incluindo cargas na mesma barra.
    local={}
    gen_idx={g:i for i,g in enumerate(grid.generators)}
    for g in grid.generators:
        if g.active and g.bus.active and not g.bus.is_slack and g.control_mode==GeneratorControlMode.V:
            local.setdefault(g.bus,[]).append(g)
    def finish_local_audit(result):
        if getattr(options, 'local_q_allocation', 'capability_range') != 'capability_range':
            from VeraGridEngine.Simulations.PowerFlow.local_reactive_allocation import finalize_local_control_audit
            return finalize_local_control_audit(result, local)
        return result
    def set_control(c,value):
        c['value']=float(np.clip(value,c['min'],c['max']))
        if c['kind']=='svc': c['device'].B=c['value']
        else:
            # Divisão proporcional com saturação nos limites individuais.
            from VeraGridEngine.IO.ons_ems.analytic_controls import distribute_q
            devices=c['devices'];q=distribute_q(c,c['value'])
            for g,value in zip(devices,q):
                g.Q=float(value)
                g.control_mode=GeneratorControlMode.Q
    def inner():
        if cancel_callback is not None and cancel_callback():
            raise RuntimeError("Estudo cancelado durante o ajuste dos controles.")
        for group in local.values():
            for g in group: g.control_mode=GeneratorControlMode.V
        for step in range(len(local)+1):
            driver=PowerFlowDriver(grid,options,import_adapter_internal=True) if cached_solver is None else None
            if telemetry is not None:
                import time
                started=time.perf_counter()
                telemetry['pf_calls']=telemetry.get('pf_calls',0)+1
            try:
                if cached_solver is not None:
                    r = cached_solver.run()
                else:
                    driver.run();r=driver.results
            finally:
                if telemetry is not None:
                    telemetry['pf_wall_s']=telemetry.get('pf_wall_s',0)+time.perf_counter()-started
            if telemetry is not None:
                telemetry['sum_iterations_reported']=telemetry.get('sum_iterations_reported',0)+float(r.iterations)
            if not r.converged:
                raise RuntimeError('Fluxo interno não convergiu')
            changed=False
            if enforce_local_q:
                for bus,group in local.items():
                    if group[0].control_mode!=GeneratorControlMode.V: continue
                    q=sum(r.gen_q[gen_idx[g]] for g in group)
                    qmin=sum(g.Qmin for g in group);qmax=sum(g.Qmax for g in group)
                    if q<qmin-1e-6 or q>qmax+1e-6:
                        for g in group:
                            g.Q=g.Qmin if q<qmin else g.Qmax
                            g.control_mode=GeneratorControlMode.Q
                        changed=True
            if not changed: return finish_local_audit(r)
        raise RuntimeError('Limites locais de Q não estabilizaram')
    def residual(r):
        return equations.residual(r)
    def active_signature(free):
        local_modes = tuple(g.control_mode for group in local.values() for g in group)
        remote_saturation = tuple(-1 if g.Q <= g.Qmin+1e-7 else 1 if g.Q >= g.Qmax-1e-7 else 0
                                  for c in controls if c['kind'] == 'generator' for g in c['devices'])
        control_bounds = tuple(-1 if c['value'] <= c['min']+1e-7 else
                               1 if c['value'] >= c['max']-1e-7 else 0 for c in controls)
        return tuple(free), local_modes, remote_saturation, control_bounds
    def finite_jacobian(free, x, err):
        count('jacobian_builds')
        matrix = np.empty((len(free), len(free)))
        for col,k in enumerate(free):
            c=controls[k];step=max(0.05,abs(x[k])*1e-4)
            if x[k]+step>c['max']: step=-step
            set_control(c,x[k]+step)
            actual=c['value']-x[k]
            if actual==0: raise RuntimeError('Controle sem faixa de perturbação')
            trial=inner();matrix[:,col]=(residual(trial)[free]-err[free])/actual
            set_control(c,x[k])
        return matrix
    def line_search(free, delta, x, effective):
        for alpha in [1.,0.5,0.25,0.125,0.0625]:
            count('line_search_trials')
            for k,d in zip(free,delta): set_control(controls[k],x[k]+alpha*d)
            trial=inner();rawerr=residual(trial)
            bound=equations.limited(trial,rawerr)
            effective_new=rawerr.copy();effective_new[bound]=0.
            if np.max(np.abs(effective_new)) < np.max(np.abs(effective)):
                return trial, rawerr, bound
        return None, None, None
    def solve_correction(matrix, free, error):
        try:
            return np.linalg.solve(matrix, -error[free])
        except np.linalg.LinAlgError as exc:
            # Keep the physical equations and the actual controller identities.
            # A singular sensitivity is not evidence of AC non-convergence.
            singular_values = np.linalg.svd(matrix, compute_uv=False)
            diagnostic = dict(schema_version=1, stage='remote_voltage_sensitivity',
                iteration=iteration, shape=list(matrix.shape),
                rank=int(np.linalg.matrix_rank(matrix)),
                singular_values=singular_values.tolist(),
                controls=[dict(index=int(k), name=controls[k]['name'],
                    kind=controls[k]['kind'], bus=controls[k]['bus'],
                    target=controls[k]['target'], value=controls[k]['value'],
                    minimum=controls[k]['min'], maximum=controls[k]['max'],
                    error_pu=float(error[k]),
                    devices=[g.code for g in controls[k].get('devices', [])]) for k in free],
                sensitivity=matrix.tolist(), trace=trace)
            if telemetry is not None:
                telemetry['control_failure'] = diagnostic
            affected = ', '.join(f"{controls[k]['name']}@{controls[k]['bus']}" for k in free)
            raise RuntimeError('Sensibilidade singular dos controles de tensão '
                f"(posto {diagnostic['rank']}/{len(free)}, iteração {iteration}); "
                'não é classificação de instabilidade. Controles: '+affected) from exc
    for c in controls: set_control(c,c['value'])
    if not controls: return inner(),trace
    J = None
    signature = None
    last_full_iteration = -jacobian_refresh_outer
    if use_broyden and sensitivity_cache is not None and sensitivity_cache.get('jacobian') is not None:
        J = sensitivity_cache['jacobian'].copy()
        signature = sensitivity_cache.get('signature')
        last_full_iteration = 0
        count('scenario_sensitivity_cache_loaded')
    accepted_result = None
    for iteration in range(max_outer):
        if use_broyden and cancel_callback is not None and cancel_callback():
            raise RuntimeError("Estudo cancelado durante o ajuste dos controles.")
        if accepted_result is not None:
            r=accepted_result;accepted_result=None
            count('accepted_result_cache_hits')
        else:
            r=inner()
        err=residual(r)
        x=np.array([c['value'] for c in controls])
        # Alvo relaxado apenas quando o recurso esgota a faixa documentada.
        limited=equations.limited(r,err)
        effective=err.copy();effective[limited]=0.
        records=equations.records(r,err,limited)
        record={'iteration':iteration,'max_equation_error_pu':float(np.max(np.abs(effective))),
            'max_voltage_error_pu':max((abs(row['voltage_error_pu']) for row in records if not row['limited']),default=0.),
            'controls':records}
        trace.append(record)
        free=np.where(~limited)[0]
        current_signature = active_signature(free)
        if np.max(np.abs(effective))<voltage_tolerance:
            if use_broyden and sensitivity_cache is not None:
                sensitivity_cache['jacobian'] = J.copy() if J is not None and signature == current_signature else None
                sensitivity_cache['signature'] = current_signature
            return r,trace
        approximate = bool(use_broyden and J is not None and signature == current_signature and
                           iteration-last_full_iteration < jacobian_refresh_outer and np.isfinite(J).all())
        if not approximate:
            if use_analytic and cached_solver is not None:
                try:
                    from VeraGridEngine.IO.ons_ems.analytic_controls import control_jacobian
                    J = control_jacobian(cached_solver, r, controls, free)
                    count('analytic_jacobian_builds')
                except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                    count('analytic_fallbacks')
                    if telemetry is not None:
                        telemetry['analytic_fallback_reason'] = str(exc)
                    J = finite_jacobian(free,x,err)
            else:
                J=finite_jacobian(free,x,err)
            last_full_iteration=iteration
        else:
            count('jacobian_reused')
        trial = None
        try:
            delta=solve_correction(J,free,err)
            trial,newerr,bound=line_search(free,delta,x,effective)
        except np.linalg.LinAlgError:
            if not approximate and not use_analytic:
                raise
        except RuntimeError:
            if (not approximate and not use_analytic) or (cancel_callback is not None and cancel_callback()):
                raise
        if trial is None and (approximate or use_analytic):
            # A aproximação nunca é motivo suficiente para declarar falha.
            # Retorne ao mesmo estado e calcule todas as sensibilidades por FD.
            count('finite_difference_fallbacks')
            for c,value in zip(controls,x): set_control(c,value)
            J=finite_jacobian(free,x,err)
            last_full_iteration=iteration
            delta=solve_correction(J,free,err)
            trial,newerr,bound=line_search(free,delta,x,effective)
        if trial is None: raise RuntimeError('Iteração externa não reduziu o erro; consultar trace')
        if use_broyden:
            new_signature=active_signature(np.where(~bound)[0])
            actual_step=np.array([c['value'] for c in controls])[free]-x[free]
            denominator=float(actual_step @ actual_step)
            if new_signature == current_signature and denominator > 1e-20:
                # Atualização secante com o deslocamento efetivo, após limites.
                J=J+np.outer(newerr[free]-err[free]-J @ actual_step,actual_step)/denominator
                signature=new_signature
                count('broyden_updates')
            else:
                J=None;signature=None
                count('active_set_refreshes')
        # A busca de passo já resolveu o estado aceito. Não repetir o mesmo
        # fluxo na próxima iteração, inclusive com Jacobiana por diferenças finitas.
        accepted_result=trial
    raise RuntimeError('Controles remotos não convergiram')
