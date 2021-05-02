import numpy as np
import sympy as sym
import scipy.sparse as sps
from scipy.sparse.linalg import dsolve, isolve
from scipy.sparse.linalg import bicg, cg, bicgstab,gmres
import matplotlib.pyplot as plt
import pandas as pd

class FDM():
    
    def __init__(self, p, q, rhs, alfa, beta, N, mtd, solver='dsolve'):
        '''
        Úloha:
            Pravá strana rovnice:
            rhs...f(x)
            Okrajové podmínky:
            alfa...u(0)=alfa
            beta...u(1)=beta
        Metoda konečných diferencí(FDM):
            mtd...volba schématu FDM
            N...hrubost sítě S
            solver...změna řšiče soustavy AU=F
            solver = 'bicgstab' jinak dsolve
        '''
        self.p = p
        self.q = q
        self.rhs = rhs
        self.alfa = alfa 
        self.beta = beta
        self.N = N #dělení intervalu (0,1) na N podintervalů, tj N+1 bodů (x0,x1,...,xN)
        self.h = 1 / N #krok
        self.mtd = mtd
        self.solver_method = solver
        
        
    def set_network(self):
        '''
        Nastavení ekvidistantní sítě S.
        '''
        self.x = np.zeros(self.N+1)
        for i in  range(self.N+1):
            self.x[i] = i * self.h 
            
        
    def set_right_side(self):
        '''
        Inicializace funkce f(x).
        '''
        self.f = np.zeros(self.N+1)
        #x = sym.symbols('x')
        for i in range(self.N+1):
            self.f[i] = (lambda x: eval(self.rhs))(self.x[i])
            #self.f[i] = eval(self.rhs).subs(x,self.x[i])
        #print(self.f)
    
    
    def ex_solution(self):
        '''
        Výpočet přesného řešení úlohy.
        '''
        x, C1, C2 = sym.symbols('x C1 C2', real=True)
        u = sym.Function('u')
        rce = sym.Eq(- sym.sympify(self.p) * u(x).diff(x, x) + sym.sympify(self.q) * u(x).diff(x), eval(self.rhs))
        uce = sym.dsolve(rce, u(x), ics = {u(0): self.alfa, u(1): self.beta}).rhs #atribut: rhs right hand side - vrátí pravou stranu, (lhs, vrací u(x))
        #display(uce)
        #self.sym_u = uce
        self.u = np.zeros(self.N+1)
        for i in range(len(self.x)):
            self.u[i] = uce.subs(x,self.x[i])
        
        #uce = sym.lambdify(x, uce, 'numpy')
        #self.u = uce(self.x)
        
    
    def central_diff(self):
        '''
        Schéma centrální diference.
        Výstup: matice A, vektor F
        '''
        # Matice A:
        A_diags = np.zeros([3,self.N-1])        
        A_diags[0,:] = np.full(self.N-1, - self.p - self.q * self.h / 2)
        A_diags[1,:] = np.full(self.N-1,  2 * self.p)
        A_diags[2,:] = np.full(self.N-1, - self.p + self.q * self.h /2)        
        self.A = sps.spdiags(A_diags, [-1, 0, 1], self.N-1, self.N-1, format = 'csr')
        
        self.A /= self.h**2 
        
        # Pravá strana F:
        self.set_right_side()
        self.f[1] = self.f[1] + (self.q / (2 * self.h) + self.p / self.h**2 ) * self.alfa 
        self.f[-2] = self.f[-2] - (self.q / (2 * self.h) - self.p / self.h**2) * self.beta       
        self.F = np.copy(self.f[1:-1])
    
    def right_diff(self):
        '''
        Schéma pravé diference.
        Výstup: matice A, vektor F
        '''
        # Matice A:
        A_diags = np.zeros([3,self.N-1])  
        A_diags[0,:] = np.full(self.N-1, - self.p)/ self.h**2
        A_diags[1,:] = np.full(self.N-1, 2 * self.p - self.q * self.h) / self.h**2
        A_diags[2,:] = np.full(self.N-1, - self.p + self.q * self.h) / self.h**2  
        self.A = sps.spdiags(A_diags, [-1, 0, 1], self.N-1, self.N-1, format = 'csr')
        
        # Pravá strana F:
        self.set_right_side()        
        self.f[1] = self.f[1] + (self.p / self.h**2) * self.alfa 
        self.f[-2] = self.f[-2] + ( - self.q / self.h + self.p / self.h**2) * self.beta        
        self.F = np.copy(self.f[1:-1])
    
    def left_diff(self):
        '''
        Schéma levé diference.
        Výstup: matice A, vektor F
        '''
        # Matice A:
        A_diags = np.zeros([3,self.N-1])  
        A_diags[0,:] = np.full(self.N-1, - self.p - self.q * self.h)/ self.h**2
        A_diags[1,:] = np.full(self.N-1, 2 * self.p + self.q * self.h) / self.h**2
        A_diags[2,:] = np.full(self.N-1, - self.p) / self.h**2  
        self.A = sps.spdiags(A_diags, [-1, 0, 1], self.N-1, self.N-1, format = 'csr')
        
        # Pravá strana F:
        self.set_right_side()        
        self.f[1] = self.f[1] + (self.q / self.h + self.p / self.h**2 ) * self.alfa 
        self.f[-2] = self.f[-2] + self.p / self.h**2 * self.beta        
        self.F = np.copy(self.f[1:-1])
     
    def fourth_order_diff(self):
        '''
        Schéma diference 4. řádu.
        Výstup: matice A, vektor F
        '''
        # Matice A
        A_diags = np.zeros([7, self.N-1])
        A_diags[1,:] = self.p
        A_diags[2,:] = - 16 * self.p - 6 * self.h * self.q
        A_diags[3,:] = 30 * self.p
        A_diags[4,:] = - 16 * self.p + 6 * self.h * self.q
        A_diags[5,:] = self.p
        
        A_diags[3,0] = 20 * self.p
        A_diags[4,1] = - 6 * self.p + 6 * self.h * self.q
        A_diags[5,2] = - 4 * self.p
        A_diags[6,3] = 1 * self.p
        A_diags[3,-1] = 20 * self.p
        A_diags[2,-2] = - 6 * self.p - 6 * self.h * self.q
        A_diags[1,-3] = - 4 * self.p
        A_diags[0,-4] = 1 * self.p
        
        A_diags /= 12*self.h**2
        self.A = sps.spdiags(A_diags, [-3, -2, -1, 0, 1, 2, 3], self.N-1, self.N-1, format = 'csr')
       
        # Pravá strana
        self.set_right_side()       
        self.f[1] = self.f[1] + ( 11 * self.p / (12 * self.h**2) + self.q / (2 * self.h)) * self.alfa 
        self.f[2] = self.f[2] + ( - self.p / (12 * self.h**2) ) * self.alfa
        self.f[-3] = self.f[-3] + ( - self.p / (12 * self.h**2) ) * self.beta
        self.f[-2] = self.f[-2] + (  11 * self.p / (12 * self.h**2) - self.q / (2 * self.h)) * self.beta        
        self.F = np.copy(self.f[1:-1])
    
    def I_A_S(self):
        '''
        Schéma IAS.
        Výstup: matice A, vektor F
        '''
        # Matice A:
        koef = sym.coth((self.h*self.q)/(2*self.p))
        A_diags = np.zeros([3,self.N-1])        
        A_diags[0,:] = np.full(self.N-1, - self.q/(2*self.h)*(1+koef))
        A_diags[1,:] = np.full(self.N-1, + self.q/self.h*koef)
        A_diags[2,:] = np.full(self.N-1, + self.q/(2*self.h)*(1-koef))        
        self.A = sps.spdiags(A_diags, [-1, 0, 1], self.N-1, self.N-1, format = 'csr')
        
        # Pravá strana F:
        self.set_right_side()
        self.f[1] = self.f[1] + (self.q/(2*self.h)*(1+koef)) * self.alfa 
        self.f[-2] = self.f[-2] - (self.q/(2*self.h)*(1-koef)) * self.beta       
        self.F = np.copy(self.f[1:-1])
                 
    
    def solver(self):
        '''
        Řešení soustavy AU=F.
        Výstup: vektro numerického řešení U, chyba řešiče es
        '''
        if self.solver_method == 'bicgstab':
            self.U = bicgstab(self.A, self.F)[0]
        else:
            self.U = dsolve.spsolve(self.A, self.F, use_umfpack=True)
        
        self.solver_error = np.amax(np.abs(self.A @ self.U - self.F))
        
        self.U = np.insert(self.U, 0, self.alfa)
        self.U = np.insert(self.U, len(self.U), self.beta)
        
        
       
    def ex_gl_error(self, U, u):
        '''
        Globální chyba E (vektor).
        '''
        U = np.delete(U, 0)
        U = np.delete(U, -1)
        u = np.delete(u, 0)
        u = np.delete(u, -1)
        E = U - u    
        return E
        
    def ex_lok_error(self, u):
        '''
        Lokální chyba \tau (vektor).
        '''
        u = np.delete(u, 0)
        u = np.delete(u, -1)
        tau = self.A @ (u) - self.F
        return tau
    
    def max_norm(self, vec):
        '''
        Výpočet maximové normy.
        '''
        m_norm = np.max(np.absolute(vec))
        return m_norm
        
    def results(self):
        '''
        Výpočet zvoleného schématu, znormování chyb.
        '''
        self.set_network()
        self.ex_solution()
        if self.mtd == 'dc':
            self.central_diff()
        elif self.mtd == 'dr':
            self.right_diff()
        elif self.mtd == 'dl':
            self.left_diff()
        elif self.mtd == '4o':
            self.fourth_order_diff()
        elif self.mtd == 'ias':
            self.I_A_S()
        self.solver()
        
        self.E = self.ex_gl_error(self.U, self.u)
        self.n_E = self.max_norm(self.E)
        
        self.tau = self.ex_lok_error(self.u)
        self.n_tau = self.max_norm(self.tau)