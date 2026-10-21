# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. https://mozilla.org/MPL/2.0/
# SPDX-License-Identifier: MPL-2.0
"""Developed PF discrete-control regressions with isolated temporary fixtures."""
from pathlib import Path
import copy,unittest,tempfile
from unittest.mock import patch
import numpy as np
from VeraGridEngine import api as vg
from VeraGridEngine.IO.ons_ems.network_converter import open_stm
from VeraGridEngine.IO.ons_ems.discrete_controls import active_requests,validate_discrete_controls,solve_discrete
from VeraGridEngine.IO.ons_ems.policy import pf_options
from VeraGridEngine.IO.ons_ems.stm_controls import prepare_control_grid

def discrete_stm(kind='tap', p=40., q=30., band=(.98,1.02), count=4, size=10., initial=0., remote=False, svc=False):
    buses=["1,'Rede',230,3,0,0,0,1,1,1,0,1.15,.85,1.2,.8,1,1,0",
           "2,'Consumo',230,0,0,0,0,1,1,1,0,1.15,.85,1.2,.8,1,1,0"]
    if remote:buses += ["3,'Remota',230,0,0,0,0,1,1,1,0,1.15,.85,1.2,.8,1,1,0"]
    target=3 if remote else 2
    loads=[f"{target},'01',1,{p},{q},0,0,0,0,1,0,0,'Carga'"]
    generators=["1,'01',50,0,500,-500,1,1,500,0,0,1,1,100,1000,0,1,0,1,1,0,0,0,0,0,0,'Rede'"]
    lines=[];transformers=[];shunts=[]
    if kind in ('tap','both'):
        transformers=[f"1,2,'01',2,.005,.2,150,180,200,1,0,{target},2,1.2,.8,{band[1]},{band[0]},.01,1,1,1,1,1,11,'Regulador'"]
    else:lines=["1,2,'01',.005,.2,0,150,180,200,0,0,0,0,0,0,1,1,10,1,1,0,0,'Conexão'"]
    if remote:lines += ["2,3,'01',.005,.06,0,150,180,200,0,0,0,0,0,0,1,1,10,1,1,0,0,'Ramo remoto'"]
    if kind in ('shunt','both'):
        shunts += [f"2,1,{band[1]},{band[0]},{target},{initial},1,1,{count},{size},0"]
    if svc:shunts += [f"{target},2,1.,0,{target},0.,1,1,1,10.,0"]
    sections=[buses,loads,generators,shunts,lines,[],transformers,[],[],[],["1,0,0,'Area'"],["1,'Zona','Zona'"],["1,'Empresa'"],["1,'Subestacao','0.0.0','0.0.0'"]]
    return '5\n10 2026 100\nControles discretos\n'+'\n'.join('\n'.join([*s,'0 /']) for s in sections)+'\n'


class DiscreteControlTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix='ons-ems-controls-')
        self.addCleanup(temporary.cleanup)
        self.folder=Path(temporary.name)

    def grid(self,kind='tap',**kwargs):
        path=self.folder/(kind+'.stm');path.write_text(discrete_stm(kind,**kwargs),encoding='utf-8')
        grid=open_stm(path)
        grid.import_metadata['discrete_control_policy']={'tap_method':'discrete_steps'}
        grid.import_metadata['parameters'].update(PFBLKTAP=kind not in ('tap','both'),PFBLKSHT=kind not in ('shunt','both'),PFBLKSHFT=True)
        return grid

    def options(self,grid):
        return pf_options(grid)

    def run_flow(self,grid):
        options=self.options(grid);options.tolerance=1e-10;options.max_iter=60
        working,controls=prepare_control_grid(grid)
        result,trace=solve_discrete(working,options,controls,voltage_tolerance=1e-9,sensitivity_update='analytic')
        self.assertTrue(result.converged);self.assertLess(result.error,1e-9)
        return working,result,trace

    def test_oltc_band_is_reached_on_physical_lattice_with_new_ac_solutions(self):
        grid=self.grid();working,result,trace=self.run_flow(grid)
        state=working.import_metadata['discrete_last_solution']
        self.assertEqual(state['status'],'band_satisfied');self.assertTrue(state['complete'])
        vm=abs(result.voltage[1]);self.assertGreaterEqual(vm,.98-1e-9);self.assertLessEqual(vm,1.02+1e-9)
        tap=working.transformers2w[0].tap_module
        self.assertAlmostEqual((tap-.8)/.01,round((tap-.8)/.01),places=9)
        self.assertGreater(state['evaluations'],state['steps']);self.assertEqual(grid.transformers2w[0].tap_module,1.)
        direct=copy.deepcopy(working);direct.import_metadata['format']='synthetic static';direct.import_metadata['pf_adapter_required']=False
        direct.import_metadata['parameters']={}
        ordinary=vg.power_flow(direct,vg.PowerFlowOptions(tolerance=1e-10,control_taps_modules=False,control_taps_phase=False))
        np.testing.assert_allclose(result.voltage,ordinary.voltage,atol=1e-10,rtol=0)
        np.testing.assert_allclose(result.Sf,ordinary.Sf,atol=1e-8,rtol=0)
        np.testing.assert_allclose(result.St,ordinary.St,atol=1e-8,rtol=0)

    def test_shunt_steps_have_voltage_squared_mvar_and_group_counts(self):
        grid=self.grid('shunt',band=(.985,1.015));working,result,_=self.run_flow(grid)
        control=working.import_metadata['discrete_last_solution']['controls'][0]
        self.assertEqual(control['kind'],'shunt');self.assertEqual(control['counts'],[3])
        self.assertAlmostEqual(working.shunts[0].B,30.)
        self.assertGreaterEqual(abs(result.voltage[1]),.985-1e-9)
        # Q recebido pela rede = carga Q - B|V|²; B não é injeção fixa.
        net_q=grid.loads[0].Q-working.shunts[0].B*abs(result.voltage[1])**2
        self.assertAlmostEqual(result.St[0].imag,-net_q,places=7)
        self.assertEqual(grid.shunts[0].B,0.)

    def test_continuous_svc_remains_analytic_and_its_trace_is_retained(self):
        grid=self.grid('tap',svc=True);settings=self.options(grid)
        working,controls=prepare_control_grid(grid);telemetry={}
        result,trace=solve_discrete(working,settings,controls,sensitivity_update='analytic',telemetry=telemetry)
        self.assertTrue(result.converged);self.assertEqual(working.import_metadata['discrete_last_solution']['status'],'band_satisfied')
        self.assertGreater(telemetry.get('analytic_jacobian_builds',0),0)
        self.assertIn('SVC 2',[c['name'] for c in trace[-1]['controls']])
        self.assertIn('Regulador',[c['name'] for c in trace[-1]['controls']])

    def test_step_limit_does_not_qualify_unfinished_control(self):
        grid=self.grid();settings=self.options(grid);working,controls=prepare_control_grid(grid)
        result,_=solve_discrete(working,settings,controls,max_steps=1,sensitivity_update='analytic')
        self.assertTrue(result.converged);self.assertEqual(working.import_metadata['discrete_last_solution']['status'],'step_limit')
        self.assertFalse(working.import_metadata['discrete_last_solution']['complete'])

    def test_cancellation_rolls_back_and_preserves_last_valid_diagnostics(self):
        grid=self.grid();settings=self.options(grid);working,controls=prepare_control_grid(grid)
        working.import_metadata['lcc_last_solution']={'marker':'last valid'}
        working.import_metadata['discrete_last_solution']={'marker':'last valid'}
        calls=[0]
        def cancel():calls[0]+=1;return calls[0]>=3
        with self.assertRaisesRegex(RuntimeError,'cancelado'):solve_discrete(working,settings,controls,cancel_callback=cancel,sensitivity_update='analytic')
        self.assertEqual(working.transformers2w[0].tap_module,1.)
        self.assertEqual(working.import_metadata['lcc_last_solution'],{'marker':'last valid'})
        # The latest developed solver clears stale per-attempt diagnostics
        # before snapshotting; physical positions and the valid LCC state stay.
        self.assertNotIn('discrete_last_solution',working.import_metadata)

    def test_failed_probes_restore_the_ac_state_and_remain_inconclusive(self):
        import VeraGridEngine.IO.ons_ems.lcc as lcc
        grid=self.grid();settings=self.options(grid);working,controls=prepare_control_grid(grid)
        original=lcc.solve_coupled
        def unstable(g,o,c,**kw):
            if abs(g.transformers2w[0].tap_module-1.)>1e-10:
                g.generators[0].Q=99999.;g.import_metadata['lcc_last_solution']={'incomplete':True}
                raise RuntimeError('Falha numérica proposital')
            return original(g,o,c,**kw)
        with patch.object(lcc,'solve_coupled',unstable):result,_=solve_discrete(working,settings,controls,sensitivity_update='analytic')
        self.assertTrue(result.converged);self.assertEqual(working.transformers2w[0].tap_module,1.)
        self.assertNotEqual(working.generators[0].Q,99999.)
        self.assertNotIn('lcc_last_solution',working.import_metadata)
        state=working.import_metadata['discrete_last_solution'];self.assertEqual(state['status'],'inconclusive_probes')
        self.assertFalse(state['complete']);self.assertTrue(all(p['classification']=='numerical_inconclusive' for p in state['failed_probes']))

    def test_blocked_requests_preserve_positions_and_ignore_nonexecuted_invalid_steps(self):
        grid=self.grid('both');grid.import_metadata['parameters'].update(PFBLKTAP=True,PFBLKSHT=True)
        grid.import_metadata['source_records']['transformers'][0][17]=0
        self.assertEqual(validate_discrete_controls(grid,self.options(grid)),[])
        working,result,_=self.run_flow(grid);self.assertEqual(working.transformers2w[0].tap_module,1.);self.assertEqual(working.shunts[0].B,0.)
        self.assertEqual(working.import_metadata['discrete_last_solution']['status'],'not_requested')

    def test_continuous_oltc_selects_default_without_rounding_nonlattice_tap(self):
        grid=self.grid();grid.import_metadata['discrete_control_policy']={}
        grid.transformers2w[0].tap_module=.9973
        grid.import_metadata['source_records']['transformers'][0][17]=0.
        working,result,_=self.run_flow(grid)
        state=working.import_metadata['discrete_last_solution']
        self.assertEqual(state['status'],'band_satisfied')
        self.assertEqual(state['controls'][0]['tap_method'],'continuous_band')
        self.assertLessEqual(state['steps'],8)
        self.assertGreaterEqual(abs(result.voltage[1]),.98-1e-9)
        self.assertEqual(grid.transformers2w[0].tap_module,.9973)

    def phase_grid(self):
        folder=Path(__file__).resolve().parent/'data'/'ons_ems'
        return open_stm(folder/'phase_control_developed.stm',folder/'phase_control_developed.prm')

    def test_phase_mw_control_uses_from_terminal_and_matches_reference_target_policy(self):
        working,result,_=self.run_flow(self.phase_grid())
        state=working.import_metadata['discrete_last_solution'];control=state['controls'][0]
        self.assertEqual(state['status'],'band_satisfied');self.assertEqual(control['kind'],'phase')
        self.assertEqual(control['measurement_terminal'],'from')
        self.assertGreaterEqual(control['power_mw'],19.-1e-8);self.assertLessEqual(control['power_mw'],21.+1e-8)
        self.assertLess(abs(control['power_mw']-19.2),1e-4)
        self.assertGreater(control['phase_deg'],0.)
        self.assertLess(abs(control['phase_deg']-6.397135254501),.001)
        self.assertAlmostEqual(control['power_mw'],result.Sf[2].real,places=7)

    def test_phase_same_session_two_driver_runs_and_internal_native_fallback(self):
        grid=self.phase_grid();options=vg.PowerFlowOptions(tolerance=1e-10,max_iter=100,control_taps_modules=False,control_taps_phase=True)
        first=vg.power_flow(grid,options);second=vg.power_flow(grid,options)
        self.assertTrue(first.converged);self.assertTrue(second.converged)
        np.testing.assert_allclose(first.voltage,second.voltage,atol=1e-12,rtol=0)
        self.assertEqual(grid.transformers2w[0].tap_phase,0.)
        working,controls=prepare_control_grid(grid)
        result,_=solve_discrete(working,options,controls,sensitivity_update='finite_difference',reuse_branch_compilation=False)
        self.assertTrue(result.converged)
        np.testing.assert_allclose(first.voltage,result.voltage,atol=1e-7,rtol=0)
        self.assertNotIn('stm_control_inner',working.import_metadata)

    def test_shunt_internal_fallback_has_no_adapter_recursion_or_double_normalization(self):
        grid=self.grid('shunt');settings=self.options(grid);working,controls=prepare_control_grid(grid)
        result,_=solve_discrete(working,settings,controls,reuse_branch_compilation=False,sensitivity_update='finite_difference')
        self.assertTrue(result.converged);self.assertEqual(working.import_metadata['discrete_last_solution']['status'],'band_satisfied')
        direct=copy.deepcopy(working);direct.import_metadata['format']='synthetic static';direct.import_metadata['pf_adapter_required']=False
        direct.import_metadata.pop('effective_discrete',None)
        ordinary=vg.power_flow(direct,vg.PowerFlowOptions(tolerance=settings.tolerance,control_taps_modules=False,control_taps_phase=False))
        np.testing.assert_allclose(result.Sf,ordinary.Sf,atol=1e-7,rtol=0)
        np.testing.assert_allclose(result.St,ordinary.St,atol=1e-7,rtol=0)
