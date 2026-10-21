# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. https://mozilla.org/MPL/2.0/
# SPDX-License-Identifier: MPL-2.0
"""Existing developed electrical tests, ported without changing their equations."""
import copy,math,json,tempfile,unittest
from pathlib import Path
import numpy as np
from VeraGridEngine import api as vg
from VeraGridEngine.IO.ons_ems.network_converter import open_stm,read_prm
from VeraGridEngine.IO.ons_ems.association import read_auxiliaries
from VeraGridEngine.IO.ons_ems.lcc import operating_point,records
from VeraGridEngine.IO.ons_ems.policy import pf_options
from VeraGridEngine.IO.ons_ems.stm_controls import prepare_control_grid

def text_stm(missing_voltage=False):
    sections=[
        ["1, 'Geração / norte', 230, 3, 0, 0, 0, 1, 1, 1, 0, 1.1, .9, 1.12, .88, 1, 1, 0",
         f"2.1, 'Carga', 230, 0, 0, 0, 0, 1, 1, {0 if missing_voltage else 1}, 0, 1.1, .9, 1.12, .88, 1, 1, 0"],
        ["2.1,'01',1,50,10,0,0,0,0,1,0,0,'Carga A'"],
        ["1,'01',50,10,100,-100,1,1,100,0,0,1,1,100,100,0,1,0,1,1,0,0,0,0,0,0,'G1'"],
        [],["1,2.1,'01',.01,.1,0,200,220,240,0,0,0,0,0,0,1,1,10,1,1,0,0,'Linha'"],
        [],[],[],[],[],["1,'Norte'"],["1,'Zona'"],["1,'Empresa'"],["1,'Subestação'"]]
    return '5\n10 2026 100\nTeste de importação\n'+'\n'.join('\n'.join(s)+'\n0 /' for s in sections)+'\n'


def network():
    grid = vg.MultiCircuit(name="Rede de ensaio", Sbase=100., fbase=60.)
    buses = []
    for index in range(4):
        bus = vg.Bus(name=f"Barra {index}", code=f"B{index}", Vnom=230.,
                     vmin=.90, vmax=1.10, is_slack=index == 0)
        grid.add_bus(bus)
        buses.append(bus)
    for index, bus in enumerate(buses):
        grid.add_generator(bus, vg.Generator(name=f"Gerador {index}", code=f"G{index}",
            P=0. if index == 0 else 20., Pmin=-100. if index == 0 else 0.,
            Pmax=100. if index == 0 else 60., Qmin=-100., Qmax=100., vset=1.))
    grid.add_load(buses[0], vg.Load(name="Carga", code="C0", P=60., Q=10.))
    for f, t, code in ((0, 1, "L01"), (0, 2, "L02"), (0, 3, "L03"),
                       (1, 2, "L12"), (2, 3, "L23")):
        grid.add_line(vg.Line(bus_from=buses[f], bus_to=buses[t], code=code,
                              name=code, r=.001, x=.05, rate=150.))
    config = {"schema_version": 1,
              "groups": [{"name": f"Grupo {index}", "generator_codes": [f"G{index}"]}
                         for index in range(1, 4)],
              "axes": {"directions": 4, "step_mw": 5., "max_radius_mw": 10.,
                       "boundary_tolerance_mw": .1},
              "contingencies": [], "tolerances": {"pf_pu": 1e-7}}
    return grid, config


def link_record(mode=1):
    return dict(code='1|2.1|DC1',pole='1',mode=mode,resistance=5.,order=50. if mode in (1,3) else 500.,
        voltage=100.,threshold=0.,margin=.1,compounding=0.,nominal_voltage=100.,rate=200.,
        name='Elo',active=True,
        rectifier=['1','2','30','5','0','1','230','.18','1','1.3','.7','.01','0','Retificador'],
        inverter=['2.1','2','30','17','0','1','230','.18','1','1.3','.7','.01','0','Inversor'])


class SageEquipmentTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.folder=Path(self.temp.name)
        self.stm=self.folder/'Rede.stm';self.stm.write_text(text_stm(),encoding='utf-8')

    def tearDown(self):self.temp.cleanup()

    def test_selected_input_voltage_bounds_preserve_bus_seed_and_trace(self):
        from VeraGridEngine.IO.ons_ems.policy import parameter_inventory
        original=open_stm(self.stm)
        # Restrição de entrada acima da tensão cadastrada: muda o controle,
        # sem mascarar o chute original da rede nem alterar o STM.
        prm=self.folder/'Limites.prm'
        prm.write_text('ORGUNDERVOLT = 1.10\nORGOVERVOLT = 1.30\n',encoding='utf-8')
        converted=open_stm(self.stm,prm)
        self.assertEqual(converted.buses[0].Vm0,original.buses[0].Vm0)
        self.assertEqual(converted.generators[0].Vset,1.10)
        change=next(x for x in converted.import_metadata['adjustments'] if x['action']=='voltage_setting_input_bounds')
        self.assertEqual(change['original_vset_pu'],original.generators[0].Vset)
        self.assertEqual(converted.import_metadata['source_records'],original.import_metadata['source_records'])
        self.assertTrue(all(x['status']=='supported' for x in parameter_inventory(converted.import_metadata['parameters'])))
        prm.write_text('ORGUNDERVOLT = 1.30\nORGOVERVOLT = 1.10\n',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'Limites PRM'):open_stm(self.stm,prm)

    def test_lcc_four_orders_and_energy_balance(self):
        for mode in (1,2,3,4):
            r=link_record(mode);s=operating_point(r,1.,1.)
            self.assertAlmostEqual(s['pf_mw']+s['pt_mw'],s['losses_mw'],places=11)
            self.assertAlmostEqual(s['losses_mw'],r['resistance']*s['current_ka']**2,places=11)
            self.assertGreater(s['qf_mvar'],0);self.assertGreater(s['qt_mvar'],0)
            if mode==1:self.assertAlmostEqual(-s['pt_mw'],r['order'],places=10)
            elif mode==3:self.assertAlmostEqual(s['pf_mw'],r['order'],places=10)
            else:self.assertAlmostEqual(s['current_ka'],r['order']/1000,places=12)
            for row,side,angle,voltage in [(r['rectifier'],'rectifier','alpha_rad','rectifier_voltage_kv'),
                                           (r['inverter'],'inverter','gamma_rad','inverter_voltage_kv')]:
                n=int(row[1]);e0=3*np.sqrt(2)/np.pi*n*float(row[6])*float(row[7])/s[side+'_tap']
                self.assertAlmostEqual(e0*np.cos(s[angle])-3/np.pi*n*float(row[5])*s['current_ka'],s[voltage],places=10)

    def test_lcc_voltage_dependence_threshold_and_limits(self):
        r=link_record();a=operating_point(r,1.,1.);b=operating_point(r,.99,.98)
        self.assertAlmostEqual(a['alpha_rad'],np.deg2rad(float(r['rectifier'][2])),places=12)
        self.assertEqual(a['alpha_rad'],b['alpha_rad'])
        self.assertNotEqual(a['rectifier_tap'],b['rectifier_tap'])
        self.assertNotEqual(a['inverter_tap'],b['inverter_tap'])
        # Com tap livre, o controle conserva o ângulo e a tensão de ponte.
        self.assertAlmostEqual(a['qf_mvar'],b['qf_mvar'],places=10)
        self.assertAlmostEqual(a['qt_mvar'],b['qt_mvar'],places=10)
        r['threshold']=110.
        with self.assertRaisesRegex(RuntimeError,'transição'):operating_point(r,1.,1.)
        with self.assertRaises(RuntimeError):operating_point(link_record(),.1,.1)

    def test_lcc_ccc_and_bad_bridge_are_explicitly_rejected(self):
        r=link_record();header=['1','1','0','1','5','50','100','0','.1','0','100','200','Elo']
        data={'dclinks':[header,r['rectifier'],r['inverter']]}
        self.assertEqual(records(data)[0]['mode'],1)
        data['dclinks'][1][12]='2'
        # CCC agora possui equações próprias; parsing não qualifica sua segurança.
        self.assertEqual(records(data)[0]['converter_type'],'CCC')
        data['dclinks'][1][4]='.1'
        with self.assertRaisesRegex(NotImplementedError,'resistivas'):records(data)

    def test_self_contained_roundtrip_with_tables_and_actual_options(self):
        table=self.folder/'EMSTable.dat';table.write_text('3\nCarga 20 1\n',encoding='utf-8')
        prm=self.folder/'Config.prm';prm.write_text('PFMVATOL=1e-8\nPFMAXITER=91\nPFFLTSTRT=N\nPFCURRATING=Y\nPFGENVARIT=0\n',encoding='utf-8')
        grid=open_stm(self.stm,prm,ems_table_path=table,association_policy='planning_sections')
        driver=vg.PowerFlowDriver(grid);driver.run();self.assertTrue(driver.results.converged)
        self.assertEqual(driver.options.max_iter,91);self.assertEqual(driver.options.tolerance,1e-8)
        self.assertEqual(grid.import_metadata['effective_thermal']['metric'],'current')
        target=self.folder/'Estudo.veragrid';vg.save_file(grid,str(target))
        table.rename(self.folder/'Tabela fora do caminho.dat');prm.rename(self.folder/'PRM fora do caminho.prm')
        loaded=vg.open_file(str(target));self.assertEqual(loaded.import_metadata,grid.import_metadata)
        reopened=vg.PowerFlowDriver(loaded);reopened.run();self.assertTrue(reopened.results.converged)
        np.testing.assert_allclose(driver.results.voltage,reopened.results.voltage,atol=1e-10)

    def test_explicit_options_precede_prm_and_boolean_dialects_agree(self):
        prm=self.folder/'Config.prm';prm.write_text('PFFLTSTRT=Y\nPFMVATOL=1e-4\nPFMAXITER=90\n',encoding='utf-8')
        self.assertTrue(read_prm(prm)['PFFLTSTRT'])
        grid=open_stm(self.stm,prm);opts=pf_options(grid,vg.PowerFlowOptions(tolerance=1e-9,max_iter=55))
        self.assertEqual(opts.tolerance,1e-9);self.assertEqual(opts.max_iter,55)
        self.assertEqual(grid.import_metadata['effective_pf']['source'],'explicit options')

    def test_current_rating_and_terminal_currents_use_same_physical_power(self):
        grid=open_stm(self.stm);grid.import_metadata['parameters'].update(PFCURRATING=True,PFOVRLDLV=2)
        driver=vg.PowerFlowDriver(grid,vg.PowerFlowOptions(tolerance=1e-10));driver.run()
        result=driver.results;self.assertTrue(result.converged)
        f,t=result.voltage
        self.assertAlmostEqual(result.loading[0],max(abs(result.Sf[0])/abs(f),abs(result.St[0])/abs(t))/220.,places=10)
        self.assertAlmostEqual(abs(result.If[0]),abs(result.Sf[0]/(100*f)),places=10)
        self.assertAlmostEqual(abs(result.It[0]),abs(result.St[0]/(100*t)),places=10)

    def test_prm_load_models_preserve_initial_power_and_do_not_mutate_source(self):
        from VeraGridEngine.IO.ons_ems.policy import apply_load_model
        grid=open_stm(self.stm);grid.buses[1].Vm0=.98
        grid.import_metadata['original_load_models'][0]['voltage_pu']=.98
        for mode in (1,2,3):
            trial=copy.deepcopy(grid);trial.import_metadata['parameters'].update(PFMWLDMDL=mode,PFMVARLDMDL=mode)
            apply_load_model(trial);l=trial.loads[0];v=l.bus.Vm0
            self.assertAlmostEqual(l.P+l.Ir*v+l.G*v*v,50.)
            self.assertAlmostEqual(l.Q-l.Ii*v-l.B*v*v,10.)
        self.assertEqual(grid.loads[0].P,50.)

    def test_gui_case_edits_are_not_replaced_by_imported_initial_values(self):
        from VeraGridEngine.IO.ons_ems.stm_controls import prepare_control_grid
        grid=open_stm(self.stm);grid.loads[0].P=65.;grid.loads[0].Q=12.
        grid.lines[0].rate=180.;grid.lines[0].contingency_factor=1.5
        grid.import_metadata['parameters']['PFMWLDMDL']=1
        candidate,_=prepare_control_grid(grid)
        l=candidate.loads[0];v=l.bus.Vm0
        self.assertAlmostEqual(l.P+l.Ir*v+l.G*v*v,65.)
        self.assertEqual(l.Q,12.);self.assertEqual(grid.loads[0].P,65.)

    def test_inactive_missing_terminal_is_preserved_but_cannot_be_energized(self):
        text=text_stm().replace("1,2.1,'01'","1,99,'01'").replace('1,1,10,1,1,0,0','0,0,10,1,1,0,0')
        self.stm.write_text(text,encoding='utf-8');grid=open_stm(self.stm)
        self.assertEqual(grid.lines[0].bus_to.code,'99');self.assertFalse(grid.buses[-1].active)
        self.assertFalse(grid.import_metadata['missing_inactive_buses'][0]['executable'])
        grid.lines[0].active=True;driver=vg.PowerFlowDriver(grid);driver.run();self.assertFalse(driver.results.converged)
        self.stm.write_text(text.replace('0,0,10,1,1,0,0','1,1,10,1,1,0,0'),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'energizado'):open_stm(self.stm)


def remote_network():
    grid,config=network()
    grid.generators[1].control_bus=grid.buses[2]
    grid.generators[1].Vset=1.005
    grid.generators[2].control_mode=vg.GeneratorControlMode.Q
    grid.import_metadata={'pf_adapter_required':True,'source_records':{
        'Sbase':100.,'generators':[],'lines':[],'transformers':[],'series_caps':[],'line_shunts':[]}}
    return grid,config


class StmControlTests(unittest.TestCase):
    def test_baseline_reference_without_source_preserves_equipment_and_load_values(self):
        from VeraGridEngine.IO.ons_ems.stm_controls import prepare_baseline_references
        grid, _ = network()
        ghost = vg.Bus(code='90000', Vnom=66., is_slack=True)
        grid.add_bus(ghost)
        grid.add_load(ghost, vg.Load(code='ghost-load', P=0., Q=.5))
        grid.import_metadata = {'format': 'ONS EMS STM v5', 'source_records': {'buses': [['90000', 'Isolada', 66., 3]]}}
        original = len(grid.buses), len(grid.loads), [(load.P, load.Q, load.active) for load in grid.loads]
        prepare_baseline_references(grid)
        self.assertFalse(ghost.active)
        self.assertFalse(ghost.is_slack)
        self.assertEqual(original, (len(grid.buses), len(grid.loads), [(load.P, load.Q, load.active) for load in grid.loads]))
        self.assertEqual(grid.import_metadata['baseline_reference_states'][0]['retained_q_mvar'], .5)

    def test_baseline_rule_keeps_connected_references_and_real_sources(self):
        from VeraGridEngine.IO.ons_ems.stm_controls import prepare_baseline_references
        grid, _ = network()
        source = vg.Bus(code='90000', Vnom=66., is_slack=True)
        grid.add_bus(source)
        grid.add_generator(source, vg.Generator(code='source', P=0., Pmax=10.))
        connected = vg.Bus(code='90001', Vnom=66., is_slack=True)
        grid.add_bus(connected)
        grid.add_line(vg.Line(bus_from=connected, bus_to=grid.buses[0], code='connection', x=.1))
        grid.import_metadata = {'format': 'ONS EMS STM v5', 'source_records': {
            'buses': [['90000', 'Fonte', 66., 3], ['90001', 'Conectada', 66., 3]]}}
        prepare_baseline_references(grid)
        self.assertTrue(source.active)
        self.assertTrue(connected.active)
        self.assertEqual(grid.import_metadata['baseline_reference_states'], [])

    def test_private_pf_converges_without_changing_open_model(self):
        grid,_=remote_network()
        original=[(g.P,g.Q,g.control_bus,g.control_mode) for g in grid.generators]
        driver=vg.PowerFlowDriver(grid,vg.PowerFlowOptions(tolerance=1e-8,control_remote_voltage=True))
        driver.run();result=driver.results
        self.assertTrue(result.converged,str(driver.logger))
        self.assertAlmostEqual(abs(result.voltage[2]),1.005,places=7)
        self.assertEqual(original,[(g.P,g.Q,g.control_bus,g.control_mode) for g in grid.generators])
        candidate,_=prepare_control_grid(grid)
        candidate.buses[1].Vm0=.5
        candidate.generators[1].P=55
        candidate.import_metadata['pf_adapter_required']=False
        self.assertNotEqual(candidate.buses[1].Vm0,grid.buses[1].Vm0)
        self.assertEqual(grid.generators[1].P,20.)
        self.assertTrue(grid.import_metadata['pf_adapter_required'])
