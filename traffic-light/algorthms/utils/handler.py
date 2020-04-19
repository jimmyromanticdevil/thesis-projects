__author__ = 'Yohan'
#######BASED on Scikit Fuzzy Library by JDWarner#####

import numpy as np
import skfuzzy as fuzz
#import matplotlib.pyplot as plt

#pylab.rcParams['figure.figsize'] = (10.0, 5.0)

#Membership Function Inputs

def MFI(Q_In, mFI):
    lists = []
    for i in [0,3]:
        lists.append(fuzz.trapmf(Q_In, mFI[i]))
    for i in [1,2]:
        lists.insert(i, fuzz.trimf(Q_In, mFI[i]), )
    return lists

def ArrivalRate(ArrIn, ArrList):
    lists = []
    for i in ArrList:
        lists.append(fuzz.trapmf(ArrIn, i))
    return lists

def Exdci(Ex, ExLists): 
    lists = []
    for i in [0,2]:
        lists.append(fuzz.trapmf(Ex,ExLists[i]))
    lists.insert(1, fuzz.trimf(Ex, ExLists[1]))  
    return lists


def AR_Kategori(AR, AR_in, Arrst):
    lists = []
    for i in Arrst:
        lists.append(fuzz.interp_membership(AR, i, AR_in))
    
    AR_rendah, AR_sedang, AR_tinggi = lists
    dictx = {
       "AR_rendah":AR_rendah,
       "AR_sedang":AR_sedang,
       "AR_tinggi":AR_tinggi,
    } 
    return dictx

#Aktivasi membership kategori
#modified of http://nbviewer.ipython.org/gist/kickapoo/2cec262723390d6f386a
def Membership_Function(QOrigin, Q_In, mFI, param):
    get_mfi = MFI(QOrigin, mFI)
    lists = []
    for i in [0,1,2]:
         lists.append(fuzz.interp_membership(QOrigin, get_mfi[i], Q_In))
    lists.insert(3, fuzz.interp_membership(QOrigin, get_mfi[3], Q_In))
    QPL_sedikit, QPL_sedang, QPL_banyak, QPL_sbanyak = lists 
    dictx = {
         "%s_sedikit"%param:QPL_sedikit,
         "%s_sedang"%param:QPL_sedang,
         "%s_banyak"%param: QPL_banyak,
         "%s_sbanyak"%param: QPL_sbanyak,
    }
    return dictx
    

def doit_now(lemeina, perintis, rate):
    try:
       ########## INPUTS ########################
       #Input Domain
       QLeimena = np.arange(0), 70, 5)
       QPerintis = np.arange(0, 120, 5)
       Ar = np.arange(0, 3, .5)

       #Output Domain
       Ex = np.arange(-35, 40, 5)

       #Input data testing
       inputQL = lemeina
       inputQP = perintis
       inputAR = rate

       #Membership Function
       #Arrival Rate
       ArLists = [[0, 0, 0.5, 1.5], [0.5, 1, 2, 2.5], [1.5, 2.5, 3, 3]]
       Arrst = ArrivalRate(Ar, ArLists)
       #Output
       ExLists = [[-30, -30, -10, 0], [0, 0, 0], [0, 10, 30, 30]]
       Exd, Exc, Exi = Exdci(Ex, ExLists)


       #Queue
       mfi_leimena = [[-30, -5, 5, 15], [10, 17.5, 25], [20, 27.5, 35], [30, 40, 50, 70]]
       mfi_perintis = [[-27, -3, 10, 30], [20, 35, 50], [40, 55, 70], [60, 80, 95, 120]]

       #Passing input data to fuzzy system
       QLeimena_in = Membership_Function(QLeimena, inputQL, mfi_leimena, "QL")
       QPerintis_in = Membership_Function(QPerintis, inputQP, mfi_perintis, "QP")
       AR_in = AR_Kategori(Ar, inputAR, Arrst)

       #Implemetasi Rules (4x4x3 = 48 Rules)

       # IF (QLeimena is S) and (QPerintis is S) and (AR is R) then (Ex is D)
       QLQP = []
       QLeimena_in = sorted(QLeimena_in.items(), reverse=True)
       QLeimena_in.insert(3, QLeimena_in.pop(2))
       QPerintis_in = sorted(QPerintis_in.items(), reverse=True)
       QPerintis_in.insert(3, QPerintis_in.pop(2))

       exd = [1,2,3,4,5,13,14,15,16,17,25,26,27,28,29,37,38,39,40]
       exi = [6,7,8,9,10,11,12]
       exc = [18,19,20,21,22,23,24,30,31,32,33,34,35,36,41,42,43,44,45,46,47,48] 

       c = 1
       for i in QLeimena_in:
           for x in QPerintis_in:
               for y in AR_in.items(): 
                   result = np.fmin(i[1], np.fmin(x[1], y[1]))
                   #print "%s %s %s : %s"% (i[0], x[0], y[0], result) 
                   try:
                      if c in exd:
                         Imp = np.fmin(result, Exd)
                      if c in exc:
                         Imp = np.fmin(result, Exc)
                      if c in exi:
                         Imp = np.fmin(result, Exi)
                   except:
                      pass
                   else:
                      QLQP.append(Imp)
                   c+=1

       #', '.join(map(str,QLQP))
       Agg = np.max(np.dstack(QLQP),axis=2)
       Ex_defuzz = fuzz.defuzz(Ex, Agg, 'centroid')

       Curr_Gs = 86
       Gs_Ext = Curr_Gs + Ex_defuzz
    except Exception,err:
       return err   
    return {"Curr_Gs":Curr_Gs,"Hasil Fuzzy":Ex_defuzz,"Gs_Ext:":int(Gs_Ext)}

