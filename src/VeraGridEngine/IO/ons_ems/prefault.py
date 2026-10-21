# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. https://mozilla.org/MPL/2.0/
# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations
def has_stm_control_actuators(grid):
    """Detecta cadastros ativos mesmo em decks salvos com o flag antigo falso.

    Não habilita atuação nem muda preferências: o PRM de cada fase decide
    se o atuador opera. Apenas exige o caminho que conserva seus estados.
    """
    if grid.import_metadata.get('format')!='ONS EMS STM v5':return False
    branches={str(b.code):b for b in grid.get_branches(add_hvdc=False,add_vsc=False,add_switch=True)}
    for row in grid.import_metadata.get('branch_records',[]):
        if row.get('ctrl'):
            device=branches.get(str(row['key']))
            if device is None or (device.active and device.bus_from.active and device.bus_to.active):return True
    from VeraGridEngine.IO.ons_ems.stm_parser import bus_id
    shunts={str(sh.code):sh for sh in grid.shunts}
    for row in grid.import_metadata.get('source_records',{}).get('shunts',[]):
        if int(row[1])==1 and bool(int(row[6])):
            device=shunts.get('shunt|'+bus_id(row[0]))
            if device is None or (device.active and device.bus.active):return True
    return False
