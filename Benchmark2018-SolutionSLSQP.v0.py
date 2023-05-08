import numpy as np
import time
np.set_printoptions(threshold=np.inf, linewidth=400)
import math, matplotlib.pyplot as plt
from scipy.optimize import newton, minimize, Bounds, NonlinearConstraint, BFGS
from scipy.integrate import solve_ivp

def predicter(x,iChart):
    nInc = len(x)
    # definição dos dicionários
    empty_timeIncrem = {
        'number': None,
        'startTime': None, 'duration': None,'endTime': None,
        'hFini': None, 'hFfin': None,'pumpFlow': None,};
    fObjRest = {'fObj': None, 'g1': [], 'g2': []}

    def pumpState (x,t): #função "x to state converter" para len(x)=24
        s = 0
        for i in range(0, nInc):
            if (t >= float(i) and t < float(i)+x[i]): s = 1
        return s

    # criação da lista de timeSteps pelo x e maxInc, coloca por ordem e retira duplicados
    def Steps(x,maxInc,timeHorizon):
        # tomando o x para todas as horas 
        steps1=[]
        for i in range (0,int(timeHorizon)): steps1.append(float(i)+x[i])
        return sorted(set(([round(p*maxInc,10) for p in range(0, int(timeHorizon/maxInc))] + [24] + steps1)))
       
    def pumpstatesList(x,timeSteps):
        pumpStateInSteps=[]
        for valueList in timeSteps:
            pumpStateInSteps.append(pumpState(x,valueList))
        return pumpStateInSteps



    # Cálculo dos consumos em cada t
    def Caudal_VC(ti, tf):
        # definição do polinómio para volume
        a6 = -5.72800E-05; a5 = 3.9382E-03; a4=-9.8402E-02; a3 = 1.0477; a2 = -3.8621; a1 = -1.1695; a0 = 7.53930E+01
        QVC = a6/7.*(tf**7.-ti**7.)+a5/6.*(tf**6.-ti**6.)+a4/5.*(tf**5.-ti**5.)+ a3/4.*(tf**4.-ti**4.)+a2/3.*(tf**3.-ti**3.)+a1/2.*(tf**2.-ti**2.)+a0*(tf-ti)
        return QVC

    def Q_VC(t):
        # definição do polinómio para o caudal
        a6 = -5.72800E-05; a5 = 3.9382E-03; a4=-9.8402E-02; a3 = 1.0477; a2 = -3.8621; a1 = -1.1695; a0 = 7.53930E+01
        QVC = a6*(t**6)+a5*(t**5)+a4*(t**4)+ a3*(t**3)+a2*(t**2)+a1*t+a0
        return QVC       

    def Caudal_R(ti, tf):
        # definição do polinómio para volume
        a3 = -0.004; a2 = 0.09; a1 = 0.1335; a0 = 20.0
        QR = a3/4.*(tf**4.-ti**4.)+a2/3.*(tf**3.-ti**3.)+a1/2.*(tf**2.-ti**2.)+a0*(tf-ti)
        return QR

    def Q_R(t):
        # definição do polinómio para o caudal
        a3 = -0.004; a2 = 0.09; a1 = 0.1335; a0 = 20.0
        QR = a3*(t**3.)+a2*(t**2.)+a1*t + a0
        return QR       

    def tarifario(ti):
        # definição do tarifário usando o tempo inicial do incremento
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



    # Dados gerais
    g = 9.81; densidade = 1000.0; densg = g * densidade;
    hmin =  3; hmax = 7.0; hFixo = 100.0
    AF = 155.0; V0 = 620.0; hF0 = 4.; deltahF = 0.
    LPR = 3500; LRF = 6000
    f =  0.02; d =  0.3
    # variáveis constantes
    f32gpi2d5 = 32.0*f/(g*math.pi**2.0*d**5.)
    lossesCoefPR = f32gpi2d5*LPR; lossesCoefRF = f32gpi2d5*LRF;
    def hLossesPR (Q, lossesCoefPR): # Caudal em m3/s
        return lossesCoefPR * Q**2.
    def hLossesRF (Q, lossesCoefRF): # Caudal em m3/s
        return lossesCoefRF * Q**2.

    # Dados da bomba e curva hidráulica
    a1 = 280.; a2 = -0.0027; etaP = 0.75; pumpCoef = [280., 0, -0.0027]
    def hPumpCurve (Q,pumpCoef): # caudal em m3/h
        return pumpCoef[2]*Q**2 + pumpCoef[1]*Q + pumpCoef[0]

    aRes = (a2*3600.**2.) - f32gpi2d5*LPR - f32gpi2d5*LRF

    # Inicialização dos vetores
    timeInc = []; CostPrev = []
    CustoT = 0; CostPrev.append(0.0)
    for i in range(0, nInc):
        # definição dos incrementos de tempo
        timeInc.append(empty_timeIncrem.copy())
        timeInc[i]['number'] = i + 1
        if i == 0:
            timeInc[i]['startTime'] = 0
            hF = hF0
            timeInc[i]['hFini'] = hF
        else:
            timeInc[i]['startTime'] = timeInc[i-1]['endTime']
            timeInc[i]['hFini'] = timeInc[i-1]['hFfin']
        timeInc[i]['duration'] = 24 / nInc
        timeInc[i]['endTime'] = timeInc[i]['startTime']+timeInc[i]['duration'];
        #print ("timeInc", timeInc[i]['number'],timeInc[i]['startTime'],timeInc[i]['duration'])
        #
        # Cálculo dos volumes bombeados no incremento i
        QVC = Caudal_VC(timeInc[i]['startTime'], timeInc[i]['endTime'])
        QR = Caudal_R(timeInc[i]['startTime'], timeInc[i]['endTime']); QRmed= QR/timeInc[i]['duration']
        #
        # Cálculo para encontrar a raíz (equilibrio da bomba vs instalação)
        def BalancePump (Qp, QR, pumpCoef,lossesCoefPR,lossesCoefRF,hF,hFixo):
            return hPumpCurve(Qp,pumpCoef)- hLossesPR(Qp/3600,lossesCoefPR)-hLossesRF((Qp-QR)/3600,lossesCoefRF) - hF - hFixo        
        #
        # Ciclo iterativo de convergência (com tolerãncia=1.E-5)
        iter = 1; hFini = hF; hFmed = hF; deltahFold = 0.; tol = 1.E-6; maxIter = 8
        bRes = 2.*f32gpi2d5*LRF*QRmed/3600.
        while iter < maxIter:
            cRes = a1-hFixo -f32gpi2d5*LRF*(QRmed/3600)**2 - hFmed
            Qp = (-bRes - math.sqrt(bRes**2 - 4 * aRes * cRes))/(2*aRes) * 3600
            #
            Qp2 = newton(BalancePump,x0=198.,args=(QRmed,pumpCoef,lossesCoefPR,lossesCoefRF,hFmed,hFixo))
            deltahFn = (Qp*x[i]*timeInc[i]['duration']-QVC-QR)/AF
            hF = hFini + deltahFn
            hFmed = hFini + deltahFn / 2
            #print("iter=",iter,cRes, Qp,Qp2, deltahFn,deltahFold, hF, deltahFn-deltahFold)
            if math.fabs(deltahFn-deltahFold) > tol:
                deltahFold = deltahFn
            else:
                break
            iter += 1
        timeInc[i]['hFfin']= hF
        #
        timeInc[i]['pumpFlow']= Qp2 #Qp
        #print("Qp2=",Qp2,Qp,'h=',hPumpCurve(Qp2,pumpCoef),hF+hFixo)
        #
        # Cálculo da energia utilizada
        WP = g*densidade/etaP*Qp/3600*(a1+a2*Qp**2.)    # in W
        tarifInc = tarifario(timeInc[i]['startTime'])*timeInc[i]['duration']/1000. # in Euro/W
        Custo =  x[i]*WP*tarifInc
        CustoT += Custo
        CostPrev.append(Custo)
        #fObjRest['g1'].append(hmin - hF); fObjRest['g2'].append(hF - hmax);
        fObjRest['g1'].append(hF); #fObjRest['g2'].append(hF);
        #print("it.= %2i, x= %5.3f, hF= %6.3f, WP= %7.3f, Tarif= %5.3f, Custo= %6.3f, %7.3f, constr= %7.3f, %7.3f, <0 ?"
        #      % (i, x[i], hF, WP, tarifario(timeInc[i]['startTime']), Custo, CustoT, fObjRest['g1'][i],fObjRest['g2'][i]))
    # Guardar valores em Arrays
    fObjRest['fObj']=CustoT;

    timeHorizon = 24
    maxInc = 1
    timeSteps = Steps(x,maxInc,timeHorizon=timeHorizon)
    pumpStateInSteps = pumpstatesList(x,timeSteps)
    #print('Steps=',Steps(x,maxInc,timeHorizon=timeHorizon))
    #print('pumpState=',pumpStateInSteps)
 

    Qp3 = []
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
        #print('time=',sol1.t,'hF inc=',hF0, 'Cost inc=',costT)
    
    #print('comparação=',costT,fObjRest['fObj'],'erro=',costT-fObjRest['fObj'])
    fObjRest['fObj']=CustoT;
    #print('len2',len(fObjRest['g1']))

    # Construção da solução grafica
    if iChart == 1:
        x1=[];y1=[];z1=[];pp1=[];pp2=[]; pp3=[]; stateChart=[]
        for i in range(0,nInc):
            x1.insert(i,timeInc[i]['startTime']);
            y1.insert(i,timeInc[i]['hFini']);
            z1.insert(i,10*tarifario(i/(nInc/24)));
            pp1.insert(i,timeInc[i]['pumpFlow']);
            pp2.insert(i,hPumpCurve(timeInc[i]['pumpFlow'],pumpCoef));
            pp3.insert(i,CostPrev[i]);

        x1.insert(nInc,timeInc[nInc-1]['endTime']); y1.insert(nInc,timeInc[nInc-1]['hFfin']); z1.insert(nInc,10*tarifario(i/(nInc/24)));pp3.insert(nInc,CostPrev[nInc]);
        fig, (ax1,ax2) =plt.subplots(2)
        for i in range(len(timeChart)): stateChart.append(np.sign(costChart[i]))
        #ax1.plot(x1,y1,x1[0:nInc],x[0:nInc],x1,z1,x1[0:nInc+1],pp3[0:nInc+1],timeChart,levelChart,timeChart,costChart,'.',timeChart,stateChart);
        ax1.plot(x1,z1,timeChart,levelChart,timeChart,costChart,timeChart,stateChart);        
        ax1.set_title('Solução Proposta, Custo=%f' % fObjRest['fObj']); ax1.set_xlabel('Tempo (h)');
        ax1.set_ylabel('Nivel/ status da bomba / Tarifario (x10)'); ax1.grid();
        qPChart = np.linspace(0,pumpCoef[0], 400)
        hPChart = hPumpCurve(qPChart,pumpCoef)
        ax2.plot(pp1,pp2,'o',qPChart,hPChart)
        ax2.set_title('Pump hydraulic curve')
        fig.tight_layout()
        plt.show()
        
    #exit()
    return fObjRest;

# main program (driver)
#----------------------
nInc = 24 #48 #96 #24
# Declaração de solução
x = [.65 for i in range (0, nInc)]; x[5] = 0.1; #x[17]=0.0; x[23]=0.0; x[20]=1.0;
fObjRest = predicter(x,1)
st = time.time()


def fun_obj(x): #https://stackoverflow.com/questions/63326820/sharing-objective-and-constraint-calculation-within-scipy-optimize-minimize-sl
    if any(t < 0 for t in x): print (x)
    if any(t > 1 for t in x): print (x)
    res = predicter(x,0)
    cost = res['fObj']
    #print (x, cost)
    return cost

def fun_constr_1(x):
    res = predicter(x,0)
    g1 = res['g1']
    return g1

hmin =  3; hmax = 7.0;
c1 = NonlinearConstraint(fun_constr_1, hmin, hmax, jac='2-point', hess=BFGS(), keep_feasible=False)
#c1 = NonlinearConstraint(fun_constr_1, -9999999, 0, jac='2-point', hess=BFGS(), keep_feasible=False)
#c2 = NonlinearConstraint(fun_constr_2, -9999999, 0, jac='2-point', hess=BFGS(), keep_feasible=False)

# Nao pode ser 0 nem 1 para que as restrições sejam sempre do mesmo número (foi retirada a duplicação nos incrementos).
bounds = Bounds([0.0001 for i in range(nInc)], [0.9999 for i in range(nInc)], keep_feasible=True)
# Trust-constraint
#res = minimize(fun_obj, x, args=(), method='trust-constr', jac='2-point', hess=BFGS(), constraints=[c1, c2],
#               options={'verbose': 3}, bounds=bounds)

# Sequential Least Squares Programming (SLSQP).
res = minimize(fun_obj, x, args=(), method='SLSQP', jac='2-point', constraints=[c1], options={'maxiter':50,'eps':0.01,'finite_diff_rel_step': 0.01,'iprint': 3, 'disp': True}, bounds=bounds)
#print("res=",res)
#print("Solução final: x=",[round(res.x[i], 3) for i in range(len(res.x))])
#a=input('')


# get the end time and get the execution time
et = time.time(); elapsed_time = et - st 
fObjRest = predicter(res.x,1)
print('Execution time:', elapsed_time/60, 'minutos; CustoF=',fObjRest['fObj'], '\n')

