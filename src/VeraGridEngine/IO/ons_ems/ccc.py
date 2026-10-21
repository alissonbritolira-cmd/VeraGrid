# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. https://mozilla.org/MPL/2.0/
# SPDX-License-Identifier: MPL-2.0
"""Regime periódico de uma ponte CCC de seis pulsos.

A corrente de comutação e a carga dos capacitores são resolvidas juntas.
Grandezas: kV fase-fase, kA, ohms por ponte e radianos elétricos. O lado AC
recebe P/Q da fundamental da corrente, sem uma reatância equivalente de LCC.
Hipóteses explícitas: corrente CC alisada, sistema trifásico equilibrado,
perdas de válvula nulas e sobreposição inferior a 60 graus.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from scipy.optimize import brentq, least_squares, minimize_scalar

_PI = math.pi
_GX, _GW = np.polynomial.legendre.leggauss(24)


class CommutationError(RuntimeError):
    """O estado solicitado não admite a comutação periódica representada."""


def _integral(fun, a, b):
    if b <= a:
        return 0.0
    x = a + (b-a)*(_GX+1)/2
    return float((b-a)/2*np.dot(_GW, fun(x)))


@dataclass(frozen=True)
class BridgeState:
    alpha: float
    overlap: float
    current: float
    ac_voltage: float
    reactance: float
    capacitor: float
    coefficient_b: float
    capacitor_peak: float
    capacitor_incoming_charge: float
    capacitor_outgoing_charge: float
    power: float
    reactive: float
    voltage: float
    gamma_network: float
    gamma_valve: float | None
    periodic_residual: float

    def outgoing(self, x):
        return _wave(self.alpha, self.current, self.ac_voltage, self.reactance,
                     self.capacitor, self.coefficient_b, np.asarray(x))


def _parts(alpha, current, voltage, reactance, capacitor, x):
    """Solução da EDO j''+(Xcap/X)j=(Xcap/X)Id/2-E cos(theta)/(2X)."""
    e = math.sqrt(2)*voltage
    if capacitor == 0:
        return current + e/(2*reactance)*(np.cos(alpha+x)-math.cos(alpha)), np.zeros_like(x)
    k = math.sqrt(capacitor/reactance)
    if abs(k-1) < 1e-7:
        return (current/2+current/2*np.cos(x)-e/(4*reactance)*x*np.sin(alpha+x), np.sin(x))
    d = -e/(2*reactance*(k*k-1))
    a = current/2-d*math.cos(alpha)
    return current/2+a*np.cos(k*x)+d*np.cos(alpha+x), np.sin(k*x)


def _wave(alpha, current, voltage, reactance, capacitor, b, x):
    a, s = _parts(alpha, current, voltage, reactance, capacitor, x)
    return a+b*s


def _coefficient(alpha, current, voltage, reactance, capacitor, mu):
    if capacitor == 0:
        return 0.0
    # KVL: 2 X j' = -sqrt(2)Vac sin(theta) - v_out + v_in.
    # v_out(0)=Id Xcap pi/3-Xcap integral(j); v_in(0)=-Id Xcap pi/3.
    k = math.sqrt(capacitor/reactance)
    j0 = _integral(lambda x: _parts(alpha,current,voltage,reactance,capacitor,x)[0], 0,mu)
    sj = _integral(lambda x: _parts(alpha,current,voltage,reactance,capacitor,x)[1], 0,mu)
    e = math.sqrt(2)*voltage
    derivative = (-e/(4*reactance)*math.sin(alpha) if abs(k-1)<1e-7
                  else e/(2*reactance*(k*k-1))*math.sin(alpha))
    rhs = (-e*math.sin(alpha)/(2*reactance)-current*capacitor*_PI/(3*reactance)
           +capacitor*j0/(2*reactance)-derivative)
    denominator = k-capacitor*sj/(2*reactance)
    if abs(denominator) < 1e-11:
        raise CommutationError('CCC: singularidade na condição periódica dos capacitores.')
    return rhs/denominator


def bridge(alpha, current, ac_voltage, reactance, capacitor):
    """Ponto periódico da ponte; P>0 é retificação e Q>0 consumo da rede AC.

    A primeira raiz física de j(mu)=0 fecha a comutação. A integral de j
    determina as cargas inicial/final; não se impõe uma tensão capacitiva fixa.
    """
    if not all(math.isfinite(x) for x in (alpha,current,ac_voltage,reactance,capacitor)):
        raise ValueError('CCC: grandeza não finita.')
    if min(current,ac_voltage,reactance)<=0 or capacitor<0:
        raise ValueError('CCC: corrente, tensão e reatância indutiva devem ser positivas.')
    def residual(mu):
        b = _coefficient(alpha,current,ac_voltage,reactance,capacitor,mu)
        return float(_wave(alpha,current,ac_voltage,reactance,capacitor,b,np.asarray(mu)))
    previous, value = 0.0, current
    root = None
    for x in np.linspace(_PI/18000, _PI/3-1e-9, 36):
        v = residual(float(x))
        if value>0 and v<=0:
            root=brentq(residual,previous,float(x),xtol=2e-13)
            break
        previous,value=float(x),v
    if root is None:
        raise CommutationError('CCC: comutação não concluída antes de 60 graus.')
    b=_coefficient(alpha,current,ac_voltage,reactance,capacitor,root)
    wave=lambda x:_wave(alpha,current,ac_voltage,reactance,capacitor,b,x)
    samples=wave(np.linspace(0,root,65))
    if samples.min() < -1e-8*current or samples.max()>current*(1+1e-8):
        raise CommutationError('CCC: corrente fora do intervalo das válvulas durante comutação.')
    charge_out=capacitor*_integral(wave,0,root)
    charge_in=capacitor*(current*root)-charge_out
    peak=current*capacitor*_PI/3
    def harmonic(trig):
        entering=_integral(lambda x:trig(alpha+x+_PI/6)*(current-wave(x)),0,root)
        full=_integral(lambda x:trig(alpha+x+_PI/6)*current,root,2*_PI/3)
        leaving=_integral(lambda x:trig(alpha+2*_PI/3+x+_PI/6)*wave(x),0,root)
        return entering+full+leaving
    scale=3*math.sqrt(2/3)*ac_voltage/_PI
    p,q=scale*harmonic(np.sin),-scale*harmonic(np.cos)
    gamma=None
    if p<0:
        def off(x):
            return -math.sqrt(2)*ac_voltage*math.sin(alpha+root+x)-2*peak+charge_in+current*capacitor*x
        # A tensão da válvula que acabou de sair deve permanecer reversa.
        last, fv=0.,off(0.)
        if fv < 1e-7:
            for x in np.linspace(1e-9,_PI,181):
                fx=off(float(x))
                if fx>=0 and fv<0:
                    gamma=brentq(off,last,float(x),xtol=2e-13);break
                last,fv=float(x),fx
        if gamma is None:
            raise CommutationError('CCC: margem de extinção da válvula não determinada.')
    return BridgeState(alpha,root,current,ac_voltage,reactance,capacitor,b,peak,charge_in,charge_out,
                       p,q,p/current,_PI-alpha-root,gamma,abs(residual(root))/current)


def _converter(row, vm, current, voltage, inverter=False, use_minimum_gamma=False, tap_policy='continuous'):
    n=int(row[1]); amin,amax=map(math.radians,(float(row[3]),float(row[2])))
    xmin,xmax=float(row[10]),float(row[9]); tap0=float(row[8]); step=float(row[11])
    vac=float(row[6])*float(row[7])*vm
    x,cap=float(row[5]),float(row[12])
    target=-voltage if inverter else voltage
    def state(a,t):return bridge(a,current,vac/t,x,cap)
    # Primeiro localizar a solução contínua, depois testar posições discretas.
    # O tap cadastrado é uma posição conhecida; incrementos são relativos a ela.
    a_bounds=(_PI/2+1e-5,7*_PI/6-1e-5) if inverter else (amin,amax)
    angle_target=(amin if use_minimum_gamma else amax) if inverter else amax
    seed=_PI-angle_target-.15 if inverter else (amin+amax)/2
    def eq(z):
        try:
            s=state(float(z[0]),float(z[1]))
            return [(n*s.voltage-target)/max(abs(target),1.),
                    ((s.gamma_valve or 0)-angle_target) if inverter else (z[0]-angle_target)]
        except CommutationError:
            return [10.+abs(z[0]-seed),10.+abs(z[1]-tap0)]
    # Ampliar apenas o intervalo numérico de um ângulo fixo, conferido ao final.
    if a_bounds[0]==a_bounds[1]:a_bounds=(a_bounds[0]-1e-10,a_bounds[1]+1e-10)
    fit=least_squares(eq,[float(np.clip(seed,*a_bounds)),float(np.clip(tap0,xmin,xmax))],
                      bounds=([a_bounds[0],xmin],[a_bounds[1],xmax]),xtol=1e-11,ftol=1e-11,gtol=1e-11,max_nfev=100)
    continuous=float(fit.x[1])
    if tap_policy not in ('continuous','discrete'):
        raise ValueError('CCC: política de tap deve ser continuous ou discrete.')
    if step>0 and tap_policy=='discrete':
        positions=[tap0+k*step for k in range(math.ceil((xmin-tap0)/step-1e-8),math.floor((xmax-tap0)/step+1e-8)+1)]
    else:positions=[continuous]
    positions.sort(key=lambda t:abs(t-continuous))
    for tap in positions:
        # Levantamento de raízes, inclusive disparo negativo e gammaN negativo.
        points=list(np.linspace(*a_bounds,61))
        if inverter:
            valid=[]
            for a in points:
                try:state(float(a),tap);valid.append(float(a))
                except CommutationError:pass
            if len(valid)>2:
                optimum=minimize_scalar(lambda a:state(float(a),tap).voltage,
                                        bounds=(valid[0],valid[-1]),method='bounded',options={'xatol':1e-12})
                if optimum.success:points.append(float(optimum.x))
        if a_bounds[0]<float(fit.x[0])<a_bounds[1]:points.append(float(fit.x[0]))
        previous=None
        for a in sorted(points):
            try:s=state(float(a),tap); f=n*s.voltage-target
            except CommutationError:previous=None;continue
            if previous and previous[1]*f<=0:
                try:
                    alpha=brentq(lambda z:n*state(z,tap).voltage-target,previous[0],float(a),xtol=1e-12)
                    s=state(alpha,tap)
                except CommutationError:previous=None;continue
                if not inverter or (s.gamma_valve is not None and amin-1e-8<=s.gamma_valve<=amax+1e-8):
                    # Atingir a faixa física é diferente de escolher um degrau.
                    # O ajuste contínuo pode saturar sem qualquer quantização.
                    eps=1e-7*max(1.,abs(xmin),abs(xmax))
                    limit=('minimum' if abs(tap-xmin)<=eps else
                           'maximum' if abs(tap-xmax)<=eps else None)
                    diagnostics=dict(limit=limit,limited=limit is not None,
                                     quantized=tap_policy=='discrete' and step>0 and abs(tap-continuous)>1e-7)
                    return s,tap,abs(n*s.voltage-target),diagnostics
            previous=(float(a),f)
    raise CommutationError('CCC: ordem CC incompatível com limites de disparo/extinção e posições de tap.')


def operating_point(record, vm_rectifier, vm_inverter):
    """Acopla duas pontes e a linha CC, conservando o contrato de P/Q do adaptador.

    A especificação de tensão é aplicada no terminal do modo 1/2 ou 3/4.
    Compounding/margem ficam identificados como contrato ainda não qualificado.
    Não há redução silenciosa da ordem quando as posições de tap se esgotam.
    """
    if not record['active']:
        return dict(pf_mw=0.,pt_mw=0.,qf_mvar=0.,qt_mvar=0.,losses_mw=0.,current_ka=0.,converter_type='CCC')
    mode,vs,rl=record['mode'],record['voltage'],record['resistance']
    current=record['order']/1000 if mode in (2,4) else record['order']/vs
    vi=vs if mode in (1,2) else vs-rl*current
    vr=vi+rl*current
    if min(vm_rectifier,vm_inverter,vi,vr,current)<=0:
        raise CommutationError('CCC: terminal ou ordem fora do domínio positivo.')
    if record.get('threshold',0)>0 and mode in (1,3) and vr<record['threshold']:
        raise CommutationError('CCC: transição de controle por limiar de tensão não qualificada.')
    r,i=record['rectifier'],record['inverter']
    tap_policy=record.get('ccc_tap_policy','continuous')
    sr,tr,rr,dr=_converter(r,vm_rectifier,current,vr,tap_policy=tap_policy)
    si,ti,ri,di=_converter(i,vm_inverter,current,vi,True,record.get('use_inverter_minimum_gamma',False),tap_policy)
    nr,ni=int(r[1]),int(i[1])
    return dict(pf_mw=nr*sr.power,pt_mw=ni*si.power,qf_mvar=nr*sr.reactive,qt_mvar=ni*si.reactive,
                losses_mw=rl*current*current,current_ka=current,rectifier_voltage_kv=nr*sr.voltage,
                inverter_voltage_kv=-ni*si.voltage,alpha_rad=sr.alpha,gamma_rad=si.gamma_valve,
                inverter_alpha_rad=si.alpha,gamma_network_rad=si.gamma_network,
                rectifier_overlap_rad=sr.overlap,inverter_overlap_rad=si.overlap,
                rectifier_tap=tr,inverter_tap=ti,effective_mode=mode,converter_type='CCC',
                rectifier_capacitor_peak_kv=sr.capacitor_peak,inverter_capacitor_peak_kv=si.capacitor_peak,
                control_residual_mw=max(rr,ri)*current,periodic_residual=max(sr.periodic_residual,si.periodic_residual),
                tap_policy=tap_policy,discrete_taps=tap_policy=='discrete',
                rectifier_tap_limited=dr['limited'],inverter_tap_limited=di['limited'],
                rectifier_tap_limit=dr['limit'],inverter_tap_limit=di['limit'],
                rectifier_tap_quantized=dr['quantized'],inverter_tap_quantized=di['quantized'],
                contract_pending=[k for k in ('compounding','margin') if record.get(k)])
