import geatpy as ea
import numpy as np
import random

n = 30000
constraint1 = np.array([random.randint(0, 10000)+1000 for _ in range(n)])
constraint2 = np.array([random.randint(0, 10000)-1000 for _ in range(n)])
target1 = np.array([random.randint(0, 10000)+100 for _ in range(n)])
target2 = np.array([random.randint(0, 10000)*random.randint(0, 10000) for _ in range(n)])

def singleTarget():
    @ea.Problem.single # Vars is a one-dimensional array (by default, it is a two-dimensional array)
    def evalVars(Vars):
        f = np.sum(target1 * Vars)
        
        a = constraint1 * Vars - 100
        CV = np.array([np.sum(constraint1 * Vars) - 100, 
                       -np.sum(constraint2 * Vars) + 1000])
        
        return f, CV

    problem = ea.Problem(name='single-target test demo',
                        M=1, # target dimension
                        maxormins=[1], # 1: minimize target; -1: maximize target; you can set multiply targets
                        Dim=n, # num of variables
                        varTypes=[1 for _ in range(n)], # 0: continuous var; 1: discrete var (integer)
                        lb=[0 for _ in range(n)], # low bound
                        ub=[1 for _ in range(n)], # up bound
                        evalVars=evalVars)

    algorithm = ea.soea_SEGA_templet(problem,
                                    ea.Population(Encoding='RI', NIND=3000), # init population
                                    MAXGEN=200, # maximum generations
                                    logTras=1, # write log each logTras generation
                                    trappedValue=1e-6, # avoid local optima
                                    maxTrappedCount=10)

    res = ea.optimize(algorithm, seed=2022, verbose=True, drawing=1, outputMsg=True, drawLog=True, saveFlag=True, dirName='result')

# min(target1*X), max(target2*X)
# constraint1*X < 1000, constraint2*X > 1000
def multiTargets():
    class MyProblem(ea.Problem):
        def __init__(self):
            name = 'multi-target test demo'
            M = 2 # number of target functions
            maxormins = [1, -1] # minimize target function
            Dim = n # num of variables
            varTypes = [1] # integer - 0 or 1
            lb = [0]
            ub = [1]
            lbin = [1] # 1: include low bound; 0: exclude low bound
            ubin = [1]
            ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        
        # @ea.Problem.single
        def evalVars(self, Vars):
            # target functions: min(x^2), min(x-2)^2
            # print(Vars.shape)
            # exit(1)
            f1 = Vars * target1
            f2 = Vars * target2

            ObjV = np.hstack([f1, f2])
            tmp = (Vars * constraint1)
            print(tmp.shape)
            exit(0)
            CV = np.array(
                [Vars * constraint1 - 1000,
                 -Vars * constraint2 + 1000])

            return ObjV, CV
        
    problem = MyProblem()
    algorithm = ea.moea_NSGA2_templet(problem,
                                     ea.Population(Encoding='RI', NIND=2323214),
                                     MAXGEN=200,
                                     logTras=0)
    res = ea.optimize(algorithm, seed=2022, verbose=False, drawing=2, outputMsg=True, drawLog=False, saveFlag=True, dirName='result')


def demoOriginal():
    class MyProblem(ea.Problem):
        def __init__(self):
            name ='MyProblem'
            M = 1
            maxormins = [-1]
            Dim = 3
            varTypes = [0] * Dim # integers: 0 or 1
            lb = [0,0,0]
            ub = [1,1,2]
            lbin = [1,1,0]
            ubin = [1,1,0]
            ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb,ub, lbin, ubin)
            
        
        def aimFunc(self, pop):
            Vars = pop.Phen
            x1 = Vars[:, [0]]
            x2 = Vars[:, [1]]
            x3 = Vars[:, [2]]
            if (1):
                pop.ObjV = 4*x1 + 2*x2 + x3
                pop.CV = np.hstack([2*x1 + x2 - 1, x1 + 2*x3 - 2, np.abs(x1 + x2 + x3 - 1)])

                print(pop.ObjV.shape)
                print(pop.CV.shape)
                exit()
            else:
                f = 4*x1 + 2*x2 + x3 # objective function
                exIdx1 = np.where(2*x1 + x2 > 1)[0] # find the index of instances that violate constraints
                exIdx2 = np.where(x1 + 2*x3 > 2)[0]
                exIdx3 = np.where(x1 + x2 + x3 != 1)[0]
                exIdx = np.unique(np.hstack([exIdx1, exIdx2, exIdx3]))
                
                alpha = 2
                beta = 1
                f[exIdx] += self.maxormins[0] * alpha * (np.max(f) - np.min(f) + beta)
                pop.ObjV = f 
        # def aimFunc(self, pop):
        #     Vars = pop.Phen
        #     x1 = Vars[:, [0]]
        #     x2 = Vars[:, [1]]
        #     x3 = Vars[:, [2]]
        #     f = 4*x1 + 2*x2 + x3
        #     exIdx1 = np.where(2*x1+x2>1)[0]
        #     exIdx2 = np.where(x1+2*x3>2)[0]
        #     exIdx3 = np.where(x1+x2+x3!=1)[0]
        #     exIdx = np.unique(np.hstack([exIdx1,exIdx2,exIdx3]))
        #     alpha = 2
        #     beta = 1
        #     f[exIdx] += self.maxormins[0]*alpha *(np.max(f)-np.min(f)+beta)
        #     pop.ObjV = f

    problem = MyProblem()
    Encoding ='RI'
    NIND = 50
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges,problem.borders)
    population = ea.Population(Encoding, Field, NIND)

    myAlgorithm = ea.soea_DE_best_1_L_templet(problem, population)
    myAlgorithm.MAXGEN = 1000
    myAlgorithm.mutOper.F = 0.5
    myAlgorithm.recOper.XOVR = 0.7
    myAlgorithm.logTras = 1
    myAlgorithm.trappedValue = 1e-6
    myAlgorithm.maxTrappedCount = 100

    # [BestIndi, population] = myAlgorithm.run()
    # BestIndi.save()
    # print('评价次数：%s'% myAlgorithm.evalsNum)
    # print('时间已过%s秒'% myAlgorithm.passTime)
    # if BestIndi.sizes != 0:
    #     print('最优的目标函数值为：%s'% BestIndi.ObjV[0][0])
    #     print('最优的控制变量值为：')
    #     for i in range(BestIndi.Phen.shape[1]):
    #         print(BestIndi.Phen[0, i])
    # else:
    #     print('没找到可行解。')
    res = ea.optimize(myAlgorithm, seed=2022, verbose=True, drawing=1, outputMsg=True, drawLog=False, saveFlag=False, dirName='result')


# singleTarget()
demoOriginal()
