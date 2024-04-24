#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 12:17:46 2021

@author: raquelalonso
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as S
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

import time

#================================================================================================================
#  Ecuación de Korteweg-de Vries                                 
#================================================================================================================

def solkdv(x,t,h):
    #empiezo en el xi y recorro todos los puntos hasta volver a llegar al xi
    xmas1 = np.concatenate([x[1:], x[:1]])
    xmas2 = np.concatenate([x[2:], x[:2]])
    xmas3 = np.concatenate([x[3:], x[:3]])

    xmenos1 = np.concatenate([x[-1:], x[:-1]])
    xmenos2 = np.concatenate([x[-2:], x[:-2]])
    xmenos3 = np.concatenate([x[-3:], x[:-3]])
    #sustituyo las derivadas parciales por las formulas de diferencias finitas 
    # centradas para derivada tercera y derivada primera
    #He sacado los coeficientes de las fomulas de la tabla del siguiente link
    #https://en.m.wikipedia.org/wiki/Finite_difference_coefficient
    
    uxxx=((xmenos3-xmas3)+8*(xmas2-xmenos2)+13*(xmenos1-xmas1))/(8*h*h*h)
    ux=((xmenos2-xmas2)+8*(xmas1-xmenos1))/(12*h)
    
    sol=-6.*x*ux-uxxx
    return(sol)
#================================================================================================================
#  Runge-Kutta   RK4                                  
#================================================================================================================
def rk4(x,dt,dx):
    
    #https://es.wikipedia.org/wiki/M%C3%A9todo_de_Runge-Kutta
    #https://wikiwaves.org/Numerical_Solution_of_the_KdV para la  FFT y ver 

    k1= dt* solkdv(x,0 ,dx)
    k2= dt* solkdv(x+k1*0.5,0,dx)
    k3= dt*solkdv(x+k2*0.5,0,dx)
    k4=dt*solkdv(x+k3,0,dx)
    return x+ (k1+2.*k2+2.*k3+k4)*1/6.
 ##con esto voy variando el tiempo (le sumo la pendiente *dt)

#================================================================================================================
#  Generación de un solitón                                  
#================================================================================================================
def solkdvexacta(x,t,v,x0):
    a = np.cosh(0.5 * np.sqrt(v) * (x - v * t - x0))
    return v / (2 * a * a)


#================================================================================================================
#   ESCRIBIR AQUÍ LOS VALORES QUE QUERAMOS SIMULAR                                      
#================================================================================================================



dx=0.1
xf=100
x=np.arange(-3,xf,dx) #Empezamos antes del 0 para que una onda centrada en x=0 no nos cree error innecesario
dt=1*10**-6
frame=1000000   #frame*dt será el tiempo recorrido
t0=0
v=14.
x0=2
Euler=False # Si True se animará por el método deseado 
RK4=False
RK45=True #Los errores unicamente están implementados para funcionar si el solitón no llega al final de la gráfica (cambiar frames o xf)
Mismagraf=True # Si quieres que se anime en la misma gráfica o cada método en gráficas distintas
frameskip=5000 #numeros de dts que nos saltaremos para hacer la animación
intervalo=20 #intervalo de tiempo en el que dibujaremos cada fotograma (animacion más rapida o lenta)
y = solkdvexacta(x, t0, v, x0)   # solkdvexacta(x, 0, 5, 5)
t=np.arange(t0,frame*dt,dt)
#valor mas alto animación más rápida
yexacta= np.zeros([t.size,y.size])
yexacta[0]=y
for i in range(1,frame):
        yexacta[i]= solkdvexacta(x, t[i], v, x0)
    
print("Condiciones del problema:")
print("dx=",dx,"m")
print("dt=",dt,"s")
print("tiempo transcurrido=",frame*dt, "s")


#================================================================================================================
#  Resolucion por nuestro método de Runge-Kutta   RK4                                 
#==============================================================================================================
if RK4==True:

    print("espere, realizando resolución por método de Runge-Kutta")
    time1=time.time()
    resrk4=np.zeros((len(t),len(x)))
    resrk4[0]=y
    for i in range(1,frame):
        resrk4[i]=rk4(resrk4[i-1],dt,dx)    
    
    time2=time.time()
    print("integración hecha, tiempo necesario: ",time2-time1)
#================================================================================================================
#  Resolucion por método de Euler (orden 1)                                
#================================================================================================================
if Euler==True:

    print("espere, realizando resolución por método de Euler")
    
    time1=time.time()
    reseul=np.zeros((len(t),len(x)))
    reseul[0]=y
    for i in range(1,frame):
       reseul[i] = reseul[i-1] + dt * solkdv(reseul[i-1],t[i-1],dx)
    
    time2=time.time()
    print("integración hecha, tiempo necesario: ",time2-time1)


#================================================================================================================
#  Resolucion por integración de scipy      RK45                                
#================================================================================================================
if RK45==True:
    print("espere, realizando resolución por integración RK45")
    time1=time.time()
    res=np.zeros((len(t),len(x)))
    res = S.odeint(solkdv,y,t,args=(dx,)) #Simularemos nuestro solitón mediante integración
    time2=time.time()
    
    print("integración hecha, tiempo necesario: ",time2-time1)

if RK45==True:
    erroresrk45=yexacta-res
    errormax=np.max(np.abs(erroresrk45))
    print("Error máximo RK45:", errormax)
if RK4==True:
    erroresrk4=yexacta-resrk4
    errormax=np.max(np.abs(erroresrk4))

    print("Error máximo RK4:", errormax)
if Euler==True:
    erroreseul=yexacta-reseul
    errormax=np.max(np.abs(erroreseul))
    print("Error máximo Euler:", errormax)        


#=========================================================================================================
#ANIMACION, NO VARIAR AQUI NADA menos axes limit o repeat usar ctrl-f
#==================================================================================================
time1=time.time()
print("espere, realizando la animación")

def inicio():
    # En la siguiente linea lo que hago es iniciar la gráfica vacía (no pintará nada)
    if RK4==True:
        line1.set_data([], [])    # Cómo véis, set_data lo que hace es colocar los datos (x e y)
    if RK45==True:
        line2.set_data( [],[])
    if Euler==True:
        line3.set_data([],[])
    line4.set_data([],[])
# Ahora pongo la función de animación
def animacion(i):
    if RK45==True:

        inte=res[i]
        line2.set_data(x, inte)
    if RK4==True:

         rk4e=resrk4[i]
         line1.set_data(x, rk4e)

    if Euler==True:

         eule=reseul[i]
         line3.set_data(x, eule)
    yexactae=yexacta[i]
    line4.set_data(x,yexactae)
    # Ahora coloco los vectores en la grafica (graf)

    plt.title("tiempo="+ "{:10.4f}".format(dt*i)+ "s")


if  Mismagraf==True:
    print("crearemos la animación")
    # Invoco una figura (generará la primera no usada y guarda la etiqueta en 'fig')
    fig1 = plt.figure(figsize=(20,10))
    # Fijo un tamañao de ejes
    ax = plt.axes(xlim=(-0, xf), ylim=(-2, 8))
    if RK4==True:

        line1, = ax.plot([], [],'ro',label='rk4')
    if RK45==True:

        line2,=ax.plot([],[],'bo',label='rk45')
    if Euler==True:
    
        line3,=ax.plot([],[],'go',label='euler')
    line4, = ax.plot([], [],'k-',label='exacta')#
    # Ya puedo montar la animación. Funciona del siguiente modo
#   fig .......... la figura en la dibujamos
#   animacion .... La función que hace las cuentas (en este ejemplo la definida antes)
#   init_func .... La función que da la posición inicial
#   frames ....... El número de 'fotogramas' que tendrá la película
#   interval ..... El número de milisegundos que tardará la película en poner cada fotograma

    anim = FuncAnimation(fig1, animacion,frames=np.arange(0, len(t), frameskip), init_func=inicio, interval=intervalo,repeat=True)
    plt.legend()
    plt.show()
    time2=time.time()    
    print("animacion hecha, tiempo necesario: ",time2-time1)

else:
    print("crearemos la animación")
    if RK4==True:
        fig1 = plt.figure(figsize=(20,10))
        # Fijo un tamañao de ejes
        ax1 = plt.axes(xlim=(-0, xf), ylim=(-2, 8))
        line1, = ax1.plot([], [],'ro-',label='rk4')
        line4, = ax.plot([], [],'k-',label='exacta')#

        
        anim1 = FuncAnimation(fig1, animacion,frames=np.arange(0, len(t), frameskip), init_func=inicio, interval=intervalo,repeat=True)
        plt.legend()
        plt.show()

    if RK45== True:
        fig2 = plt.figure(figsize=(20,10))
        # Fijo un tamañao de ejes
        ax2 = plt.axes(xlim=(-0, xf), ylim=(-2, 8))
        line2, = ax2.plot([], [],'bo-',label='rk45')
        line4, = ax.plot([], [],'k-',label='exacta')#

        
        anim2 = FuncAnimation(fig2, animacion,frames=np.arange(0, len(t), frameskip), init_func=inicio, interval=intervalo,repeat=True)
        plt.legend()
        plt.show()
    if Euler==True:
        fig3 = plt.figure(figsize=(20,10))
    # Fijo un tamañao de ejes
        ax3 = plt.axes(xlim=(-0, xf), ylim=(-2, 8))
        line3,=ax3.plot([],[],'go-',label='euler')#
        line4, = ax.plot([], [],'k-',label='exacta')#
        
        
        anim3 = FuncAnimation(fig3, animacion,frames=np.arange(0, len(t), frameskip), init_func=inicio, interval=intervalo,repeat=True)
        plt.legend()
        plt.show()
    time2=time.time()    
    print("animacion hecha, tiempo necesario: ",time2-time1)


    

#=========================================================================================================
#GUARDAR ANIMACION EN DISCO, descomentar sólo cuando queramos guardarlo, muy lento
#==================================================================================================
"""
# Esto crea un objeto de video pero, ahora, tengo que especificar el formato con el que
# se va a grabar (le digo que va a ser un mp4 (gif es demasiado pesado y lento) y que va a representar en pantalla
# 30 fotogramas por segundo)
print("espere, guardando en disco")
time1=time.time()
video = animation.FFMpegWriter(fps=30)

#CAMBIAR nombre anim por anim1, etc en mismagraf==False
anim.save(r"soliton.mp4", writer=video) 
#Hay que instalar ffmpeg antes de ejecutar el código
time2=time.time()
print("Finalizado,tiempo necesario ",time2-time1)

"""