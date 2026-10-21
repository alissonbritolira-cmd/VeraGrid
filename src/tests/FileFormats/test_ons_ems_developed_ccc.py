"""CCC: KVL independente, período, potência e contratos de controle explícitos."""
import copy
import math
import unittest
from pathlib import Path
import uuid
import numpy as np
from scipy.integrate import solve_ivp
from VeraGridEngine.IO.ons_ems.ccc import bridge, operating_point, CommutationError


def record(mode=1):
    return dict(code='7055|7054|DC6',pole='6',active=True,mode=mode,resistance=.6,
        order=329.702 if mode in (1,3) else 3296.294815140669,voltage=100.022,
        threshold=0.,margin=0.,compounding=0.,
        rectifier=['7055','2','18','-9.5','0','.959364','500','.0756','1','1.1375','.8625','.011957','1.902857','R'],
        inverter=['7054','2','30','22','0','.959364','230','.164348','1','1.1375','.8875','.011905','4.753732','I'])


class CCCBridgeTests(unittest.TestCase):
    def test_kvl_and_capacitor_charges_independent_integration(self):
        for angle,cap in [(0.,1.902857),(math.radians(168),4.753732),(math.radians(15),.959364)]:
            s=bridge(angle,3.29629,38.85,.959364,cap)
            def rhs(x,y):
                j,vo,vi=y
                return [(-math.sqrt(2)*s.ac_voltage*math.sin(angle+x)-vo+vi)/(2*s.reactance),
                        cap*j,cap*(s.current-j)]
            initial=[s.current,s.capacitor_peak-s.capacitor_outgoing_charge,-s.capacitor_peak]
            reference=solve_ivp(rhs,[0,s.overlap],initial,rtol=2e-12,atol=2e-13,dense_output=True)
            points=np.linspace(0,s.overlap,51)
            np.testing.assert_allclose(reference.sol(points)[0],s.outgoing(points),rtol=0,atol=1e-9)
            self.assertAlmostEqual(reference.y[0,-1],0.,delta=1e-9)
            self.assertAlmostEqual(reference.y[1,-1],s.capacitor_peak,delta=1e-9)
            self.assertAlmostEqual(reference.y[2,-1],-s.capacitor_peak+s.capacitor_incoming_charge,delta=1e-9)
            self.assertLess(s.periodic_residual,1e-8)

    def test_zero_capacitor_limit_matches_lcc_bridge_voltage(self):
        for alpha in [math.radians(15),math.radians(150)]:
            s=bridge(alpha,.5,41.4,1.,0.)
            expected=3*math.sqrt(2)/math.pi*41.4*math.cos(alpha)-3/math.pi*.5
            self.assertAlmostEqual(s.voltage,expected,delta=1e-9)

    def test_valve_margin_differs_from_network_angle(self):
        s=bridge(math.radians(168),3.29629,38.85,.959364,4.753732)
        self.assertGreater(s.gamma_valve,s.gamma_network)
        end=-math.sqrt(2)*s.ac_voltage*math.sin(s.alpha+s.overlap+s.gamma_valve)-2*s.capacitor_peak
        end+=s.capacitor_incoming_charge+s.current*s.capacitor*s.gamma_valve
        self.assertAlmostEqual(end,0.,delta=1e-9)

    def test_invalid_and_unfinished_commutation_are_not_safe_states(self):
        with self.assertRaises(ValueError):bridge(0.,1.,10.,0.,1.)
        with self.assertRaises(ValueError):bridge(0.,float('nan'),10.,1.,1.)
        with self.assertRaises(CommutationError):bridge(math.radians(186),3.29629,38.85,.959364,4.753732)


class CCCControlTests(unittest.TestCase):
    def test_four_order_modes_preserve_energy_and_specified_terminal(self):
        for mode in (1,2,3,4):
            r=record(mode);s=operating_point(r,1.02767,1.028)
            self.assertAlmostEqual(s['pf_mw']+s['pt_mw'],s['losses_mw'],delta=1e-8)
            self.assertAlmostEqual(s['losses_mw'],.6*s['current_ka']**2,delta=1e-10)
            self.assertLess(s['control_residual_mw'],1e-8)
            if mode==1:self.assertAlmostEqual(-s['pt_mw'],r['order'],delta=1e-8)
            if mode==3:self.assertAlmostEqual(s['pf_mw'],r['order'],delta=1e-8)
            if mode in (2,4):self.assertAlmostEqual(s['current_ka'],r['order']/1000,delta=1e-10)
            self.assertEqual(s['tap_policy'],'continuous')

    def test_voltage_changes_recalculate_reactive_and_taps_without_mutating_record(self):
        r=record();original=copy.deepcopy(r)
        a=operating_point(r,1.02767,1.028);b=operating_point(r,1.01,1.02)
        self.assertNotEqual(a['inverter_tap'],b['inverter_tap'])
        self.assertNotEqual(a['qf_mvar'],b['qf_mvar']);self.assertEqual(r,original)

    def test_discrete_taps_do_not_silently_revert_to_continuous(self):
        r=record();r['ccc_tap_policy']='discrete'
        try:s=operating_point(r,1.02767,1.028)
        except CommutationError:return # a regulação exata pode ser inviável nesse domínio
        for row,key in [(r['rectifier'],'rectifier_tap'),(r['inverter'],'inverter_tap')]:
            steps=(s[key]-float(row[8]))/float(row[11])
            self.assertAlmostEqual(steps,round(steps),delta=1e-8)
        self.assertTrue(s['discrete_taps'])

    def test_continuous_interior_taps_are_neither_limited_nor_quantized(self):
        state=operating_point(record(),1.02767,1.028)
        for side in ('rectifier','inverter'):
            self.assertFalse(state[side+'_tap_limited'])
            self.assertIsNone(state[side+'_tap_limit'])
            self.assertFalse(state[side+'_tap_quantized'])
        self.assertAlmostEqual(state['pf_mw'],336.221335704999,delta=1e-8)
        self.assertAlmostEqual(state['qt_mvar'],49.98120423611705,delta=1e-8)

    def test_voltage90_continuous_minimum_tap_is_limited_without_quantization(self):
        rec=record();original=copy.deepcopy(rec)
        state=operating_point(rec,.924903,.9252)
        self.assertAlmostEqual(state['rectifier_tap'],float(rec['rectifier'][10]),delta=1e-10)
        self.assertTrue(state['rectifier_tap_limited'])
        self.assertEqual(state['rectifier_tap_limit'],'minimum')
        self.assertFalse(state['rectifier_tap_quantized'])
        self.assertFalse(state['inverter_tap_limited'])
        self.assertIsNone(state['inverter_tap_limit'])
        self.assertFalse(state['inverter_tap_quantized'])
        self.assertAlmostEqual(state['qf_mvar'],129.6242214666033,delta=1e-8)
        self.assertAlmostEqual(-state['pt_mw'],rec['order'],delta=1e-8)
        self.assertLess(state['alpha_rad'],math.radians(float(rec['rectifier'][2])))
        self.assertEqual(rec,original)

    def test_discrete_interior_rectifier_tap_is_quantized_without_physical_saturation(self):
        from VeraGridEngine.IO.ons_ems.ccc import _converter
        rec=record();current=rec['order']/rec['voltage']
        target=rec['voltage']+rec['resistance']*current
        state,tap,residual,diagnostics=_converter(rec['rectifier'],1.02767,current,target,tap_policy='discrete')
        row=rec['rectifier'];steps=(tap-float(row[8]))/float(row[11])
        self.assertAlmostEqual(steps,round(steps),delta=1e-8)
        self.assertTrue(diagnostics['quantized'])
        self.assertFalse(diagnostics['limited'])
        self.assertIsNone(diagnostics['limit'])
        self.assertLess(residual,1e-8)
        self.assertAlmostEqual(int(row[1])*state.power,current*target,delta=1e-8)

    def test_failure_does_not_mutate_record_or_prior_state(self):
        r=record();s=operating_point(r,1.02767,1.028);old=copy.deepcopy(s)
        with self.assertRaises(CommutationError):operating_point(r,.05,.05)
        self.assertEqual(s,old)
        r['compounding']=.6
        self.assertEqual(operating_point(r,1.02767,1.028)['contract_pending'],['compounding'])


class CCCIntegrationTests(unittest.TestCase):
    def test_coupled_pf_pole_outage_and_roundtrip_keep_model_and_open_case(self):
        from VeraGridEngine import api as vg
        from test_ons_ems_developed_pf import text_stm
        from VeraGridEngine.IO.ons_ems.stm_parser import open_stm
        root=Path(__file__).resolve().parents[2]
        folder=root/'Temporarios'/'CCC'/uuid.uuid4().hex;folder.mkdir(parents=True)
        content=text_stm();header='\n'.join(content.splitlines()[:3])+'\n'
        sections=content[len(header):].split('0 /')[:14]
        # Variante controlada: uma linha AC liga barras da mesma tensão nominal.
        # A tensão de válvula da ponte inversora é preservada pelo produto kV*ratio.
        sections[0]="\n1,'R',500,3,0,0,0,1,1,1.02767,0,1.1,.9,1.12,.88,1,1,0\n2.1,'I',500,2,0,0,0,1,1,1.028,0,1.1,.9,1.12,.88,1,1,0\n"
        sections[2]="\n1,'01',50,0,500,-500,1.02767,1,100,0,0,1,1,1000,-1000,0,1,0,1,1,0,0,0,0,0,0,'G1'\n2.1,'01',20,0,500,-500,1.028,2.1,100,0,0,1,1,1000,-1000,0,2,0,1,1,0,0,0,0,0,0,'G2'\n"
        rec=record();rec['rectifier'][0]='1';rec['inverter'][0]='2.1'
        rec['inverter'][6]='500';rec['inverter'][7]=str(230*.164348/500)
        sections[8]='\n'+','.join(['6','1','1','1','.6','329.702','100.022','0','0','0','102','400.860','CCC'])+'\n'+','.join(rec['rectifier'])+'\n'+','.join(rec['inverter'])+'\n'
        source=folder/'Caso CCC.stm';source.write_text(header+'0 /'.join(sections)+'0 /\n',encoding='utf-8')
        grid=open_stm(source);original=[(g.P,g.Q) for g in grid.generators]
        options=vg.PowerFlowOptions(tolerance=1e-9,control_q=True,control_remote_voltage=True,use_stored_guess=True)
        result=vg.power_flow(grid,options)
        self.assertTrue(result.converged);self.assertLess(result.error,1e-9)
        state=grid.import_metadata['lcc_last_solution']['states']['1|2.1|DC6']
        self.assertEqual(state['converter_type'],'CCC');self.assertLess(state['periodic_residual'],1e-8)
        self.assertEqual([(g.P,g.Q) for g in grid.generators],original)
        self.assertFalse(grid.import_metadata['lcc_qualification']['qualified'])
        destination=folder/'Estudo.veragrid';vg.save_file(grid,str(destination))
        reopened=vg.open_file(str(destination));again=vg.power_flow(reopened,options)
        np.testing.assert_allclose(again.voltage,result.voltage,rtol=0,atol=1e-6)
        np.testing.assert_allclose(again.Pf_hvdc,result.Pf_hvdc,rtol=0,atol=1e-6)
        self.assertEqual(reopened.import_metadata['source_records'],grid.import_metadata['source_records'])
        reopened.hvdc_lines[0].active=False
        outage=vg.power_flow(reopened,options)
        self.assertTrue(outage.converged);np.testing.assert_array_equal(outage.Pf_hvdc,[0.])
        self.assertTrue(grid.hvdc_lines[0].active)
