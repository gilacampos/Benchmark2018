import numpy as np
import time
np.set_printoptions(threshold=np.inf, linewidth=400)
import math, matplotlib.pyplot as plt
from scipy.optimize import newton, minimize, Bounds, NonlinearConstraint, BFGS
from scipy.integrate import solve_ivp

def Converter(x,nDutyCycles,maxInc,timeHorizon):
    # definição dos dicionários: times2integrate & pumpStates
    timeIntPumpStates = {'timeInt': [], 'pumpState': []}

    # criação da lista de timeSteps pelo x e maxInc, coloca por ordem e retira duplicados
    def Steps(x,nDutyCycles,maxInc,timeHorizon):
        # tomando o x para todas as horas 
        steps1=[]
        for i in range (0,nDutyCycles): 
            steps1.append(x[i]) # start of dutycycle 
            steps1.append(x[i]+x[nDutyCycles+i]) # end of dutycycle        
        return sorted(set(([round(p*maxInc,10) for p in range(0, int(timeHorizon/maxInc))] + [24] + steps1)))
 
    def pumpStateM3 (x,nDutyCycles,t): #função "x to state converter" para len(x)=24
        s = 0
        for i in range(0, nDutyCycles):
            if (t >= x[i] and t < x[i]+x[nDutyCycles+i]): s = 1 #print(t,x[i],x[i]+x[nDutyCycles+i],s)
        return s

    def pumpstatesList(x,nDutyCycles,timeSteps):
        pumpStateInSteps=[]
        for valueList in timeSteps:
            pumpStateInSteps.append(pumpStateM3(x,nDutyCycles,valueList))
        return pumpStateInSteps

    timeIntPumpStates['timeInt'] = Steps(x,nDutyCycles,maxInc,timeHorizon=timeHorizon)
    timeIntPumpStates['pumpState'] = pumpstatesList(x,nDutyCycles,timeSteps=timeIntPumpStates['timeInt'])
    #print(timeIntPumpStates)
    return timeIntPumpStates

def hydraulicSimulator(timeIntPumpStates,iChart):
    #nInc = len(x)
    # definição dos dicionários
    fObjRest = {'fObj': None, 'g1': [], 'g2': []}

    # Cálculo dos consumos em cada t
    # definição do polinómio para o caudal QVC
    def Q_VC(t):
        a6 = -5.72800E-05; a5 = 3.9382E-03; a4=-9.8402E-02; a3 = 1.0477; a2 = -3.8621; a1 = -1.1695; a0 = 7.53930E+01
        QVC = a6*(t**6)+a5*(t**5)+a4*(t**4)+ a3*(t**3)+a2*(t**2)+a1*t+a0
        return QVC       

    # definição do polinómio para o caudal QR
    def Q_R(t):
        a3 = -0.004; a2 = 0.09; a1 = 0.1335; a0 = 20.0
        QR = a3*(t**3.)+a2*(t**2.)+a1*t + a0
        return QR

    # definição do tarifário usando o tempo inicial do incremento/timeStep
    def tarifario(ti):
        tarifHora = [None]*8; tarifCusto = [None]*8;
        set(tarifHora)
        tarifHora[0]= 0; tarifCusto[0]= 0.0737
        tarifHora[1]=2; tarifCusto[1]= 0.06618
        tarifHora[2]=6; tarifCusto[2]=  0.0737
        tarifHora[3]=7; tarifCusto[3]=  0.10094
        tarifHora[4]=9; tarifCusto[4]=  0.18581
        tarifHora[5]=12; tarifCusto[5]= 0.10094
        tarifHora[6]=24.0; tarifCusto[6]= 0.10094
        tarifHora[7]=24.001; tarifCusto[7]= 0.10094 #for the t=24.0
        tarifF = 0.
        for i in range(0, len(tarifHora)-1):
            if (ti >= tarifHora[i]) & (ti < tarifHora[i+1]):
                tarifF = tarifCusto[i]
                break
        if tarifF == 0.: print("Erro no tarifário",ti,i); quit()
        return tarifF

    # Dados gerais, constantes e caracteristicas da rede
    g = 9.81; densidade = 1000.0; densg = g * densidade
    hmin =  3; hmax = 7.0; hFixo = 100.0
    AF = 155.0; V0 = 620.0; hF0 = 4.; LPR = 3500; LRF = 6000
    f =  0.02; d =  0.3

    # variáveis constantes e definição das perdas de carga (função do caudal Q)
    f32gpi2d5 = 32.0*f/(g*math.pi**2.0*d**5.)
    lossesCoefPR = f32gpi2d5*LPR; lossesCoefRF = f32gpi2d5*LRF
    def hLossesPR (Q, lossesCoefPR): # Caudal em m3/s
        return lossesCoefPR * Q**2.
    def hLossesRF (Q, lossesCoefRF): # Caudal em m3/s
        return lossesCoefRF * Q**2.

    # Dados da bomba e curva hidráulica
    etaP = 0.75; pumpCoef = [280., 0, -0.0027]
    def hPumpCurve (Q,pumpCoef): # caudal em m3/h
        return pumpCoef[2]*Q**2 + pumpCoef[1]*Q + pumpCoef[0]

    # Cálculo para encontrar a raíz Qp (equilibrio da bomba vs instalação): sum(h)=0
    def BalancePump (Qp, QR, pumpCoef,lossesCoefPR,lossesCoefRF,hF,hFixo):
        return hPumpCurve(Qp,pumpCoef)- hLossesPR(Qp/3600,lossesCoefPR)-hLossesRF((Qp-QR)/3600,lossesCoefRF) - hF - hFixo        
        #
    # Inicialização dos vetores
    CostPrev = []; Qp3 = []
    CostPrev.append(0.0)

    #timeHorizon = 24; maxInc = 1
    timeSteps = timeIntPumpStates['timeInt'] #Steps(x,maxInc,timeHorizon=timeHorizon)
    pumpStateInSteps = timeIntPumpStates['pumpState'] #pumpstatesList(x,timeSteps)

    # Equações diferenciais dN/dt=(Qp-Qr.QVC)/A & dC/dt=power*tarifario
    def difFunc(t, y, pumpState , AF,pumpCoef,lossesCoefPR,lossesCoefRF,hFixo, densg, etaP):
        #global Qp3
        Qp = 0; pumpPowerCost = 0; hF=y[0];
        QR= Q_R(t); QVC = Q_VC(t)
        if pumpState == 1:
            Qp = newton(BalancePump,x0=198,args=(QR,pumpCoef,lossesCoefPR,lossesCoefRF,hF,hFixo))
            pumpPowerCost = densg/etaP * Qp/3600 * hPumpCurve(Qp,pumpCoef) * tarifario(t)/1000. #tarif euro/kWh
        #print('time=',t,'s=', pumpState (x,t), 'flow=', float(Qp),'m3/h with h=', hPumpCurve(Qp,pumpCoef),' m Level=',y[0],' PowerCost=', pumpPowerCost,' Cost=',y[1])
        Qp3.append(float(Qp))
        return [(float(Qp) - QR - QVC)/AF, pumpPowerCost]

    fObjRest['g1'] = []; timeChart=[]; levelChart=[]; costChart=[]
    costT = 0
    for i in range(len(timeSteps)-1):
        # Explicit Runge-Kutta methods (‘RK23’, ‘RK45’, ‘DOP853’) should be used for non-stiff problems and implicit methods (‘Radau’, ‘BDF’) for stiff problems
        sol1 = solve_ivp(difFunc, t_span=[timeSteps[i],timeSteps[i+1]], y0=[hF0, 0.], t_eval=None,method='RK23', args=(pumpStateInSteps[i], AF,pumpCoef,lossesCoefPR,lossesCoefRF,hFixo,densg, etaP), max_step=0.1,dense_output=False)
        hF0 = sol1.y[0][len(sol1.y[1])-1];
        fObjRest['g1'].append(hF0);
        if iChart == 1: timeChart.extend(sol1.t); levelChart.extend(sol1.y[0]); costChart.extend(sol1.y[1]/10.);             
        costT += sol1.y[1][len(sol1.y[1])-1]
        #print('time=',sol1.t,'Level=',sol1.y[0],'Cost=',sol1.y[1],'hF final=',hF0, 'Cost final=',costT)

    fObjRest['fObj']=costT
    #print('len2',len(fObjRest['g1']))

    # Construção da solução grafica
    if iChart == 1:
        x1=[];y1=[];z1=[];pp1=[];pp2=[]; pp3=[]; 
        stateChart=[]; tarifarioChart=[]; hPumpChart=[]
        fig, (ax1,ax2) =plt.subplots(2)
        for i in range(len(timeChart)): stateChart.append(np.sign(costChart[i]))
        for i in timeChart: tarifarioChart.append(tarifario(i)*30)
        for i in Qp3: hPumpChart.append(hPumpCurve (i,pumpCoef))
        ax1.plot(timeChart,tarifarioChart,timeChart,levelChart,timeChart,costChart,timeChart,stateChart);        
        ax1.set_title('Obtained solution with the Cost=%f euros' % fObjRest['fObj']); ax1.set_xlabel('Time (h)');
        ax1.set_ylabel('Tank-Level/ Pump-status/ Price (x30)'); ax1.grid();
        qPChart = np.linspace(0,pumpCoef[0], 400)
        hPChart = hPumpCurve(qPChart,pumpCoef)
        ax2.plot(Qp3,hPumpChart,'o',qPChart,hPChart)
        ax2.set_title('Pump hydraulic curve and operation points'); ax2.set_ylabel('hydraulic head (m)'); ax2.set_xlabel('Flow rate (m3/h)')
        fig.tight_layout()
        plt.show(block=True)
        
    #exit()
    return fObjRest

def predicter(x,nDutyCycles,maxInc,timeHorizon,iChart):
    timeIntPumpStates = Converter(x,nDutyCycles,maxInc,timeHorizon)
    fObjRest = hydraulicSimulator(timeIntPumpStates,iChart)
    return fObjRest

# main program (driver) for M3 formulation
nDutyCycles= 7; nInc = 1 #nInc makes more steps (warn: it can change the number of level constraints)
timeHorizon = 24; maxInc = timeHorizon/nInc; 
# Declaração de solução inicial #x = [.65 for i in range (0, nInc)];
x = [1, 4, 7, 10., 13., 17., 20.] + [2, 2.5, 2., 1., 1., 1., 1.]
print('Variable time window=',maxInc,'in a time Horizon of',timeHorizon,'h and InitialSol=',x)
print('initial times of the duty cycles',x[:nDutyCycles],' duration=',x[nDutyCycles:])

fObjRest = predicter(x, nDutyCycles, maxInc,timeHorizon,iChart=1)
st = time.time()
#exit()

def fun_obj(x,nDutyCycles,maxInc,timeHorizon): #https://stackoverflow.com/questions/63326820/sharing-objective-and-constraint-calculation-within-scipy-optimize-minimize-sl
    if any(t < 0 for t in x): print (x)
    if any(t > 24 for t in x): print (x)
    res = predicter(x,nDutyCycles,maxInc,timeHorizon,iChart=0)
    cost = res['fObj']
    #print (x, cost)
    return cost

def fun_constr_1(x,nDutyCycles,maxInc,timeHorizon):
    res = predicter(x,nDutyCycles,maxInc,timeHorizon,iChart=0)
    g1 = res['g1']
    #print(len(g1),'contr1=',g1)
    return g1

def fun_constr_2(x,nDutyCycles,timeHorizon):
    constr2 = []
    for i in range(nDutyCycles-1):
        constr2.append(x[i]+x[i+nDutyCycles] - x[i+1]) # x[i]+x[i+nDutyCycles] -x[i+1] < 0
    constr2.append(x[nDutyCycles-3]+x[2*nDutyCycles-1] - timeHorizon)
    #exit() #print('contr2=',constr2)
    return constr2

hmin =  3; hmax = 7.0;
c1 = NonlinearConstraint(lambda x: fun_constr_1(x,nDutyCycles,maxInc,timeHorizon), hmin, hmax, jac='2-point', hess=BFGS(), keep_feasible=False)
c2 = NonlinearConstraint(lambda x: fun_constr_2(x,nDutyCycles,timeHorizon), -np.inf, 0.0, jac='2-point', hess=BFGS(), keep_feasible=True)

# Nao pode ser 0 nem 1 para que as restrições sejam sempre do mesmo número (foi retirada a duplicação nos incrementos).
bounds = Bounds([0.0001 for i in range(nInc)], [23.9999 for i in range(nInc)], keep_feasible=True)
# Trust-constraint
#res = minimize(fun_obj, x, args=(), method='trust-constr', jac='2-point', hess=BFGS(), constraints=[c1, c2],
#               options={'verbose': 3}, bounds=bounds)

# Sequential Least Squares Programming (SLSQP).
res = minimize(fun_obj, x, args=(nDutyCycles,maxInc,timeHorizon), method='SLSQP', jac='2-point', constraints=[c1,c2], options={'ftol':0.01,'maxiter':50,'eps':0.01,'finite_diff_rel_step': 0.01,'iprint': 3, 'disp': True}, bounds=bounds)
#print("res=",res)
#print("Solução final: x=",[round(res.x[i], 3) for i in range(len(res.x))])
#a=input('')

# get the end time and get the execution time
et = time.time(); elapsed_time = et - st 
fObjRest = predicter(res.x,nDutyCycles, maxInc,timeHorizon,iChart=1)
print('Execution time:', elapsed_time/60, 'minutos; CustoF=',fObjRest['fObj'],'solution=',[round(res.x[i], 3) for i in range(len(res.x))], '\n')

