#!/usr/bin/python
# -*- coding: utf-8 -*-
import csv

def find_var(key_word, filename):
    with open(filename,'r') as f:
        lis = [x.split(',') for x in f]
        for x in zip(*lis):
                print(y)

def selection_X(filename):
    A= []
    V=[]
    f = open(filename, "r")
    data = f.readlines()
    flag=False
    for line in data:
        if "--" in line:
            flag = True
            A.append(V)
            V=[]
        if "ExpX" in line:
            a = line.split(' ')
            V.append(a[0])
    return A

print(selection_X("premier_resultat.txt"))
X= selection_X("premier_resultat.txt")

def selection_ligne(filename):
    f = open(filename, "r")
    data = f.readlines()
    return data

def names_to_dico(vect_names, diconame):
    f = open(diconame, "r")
    data = f.readlines()
    A = []
    V = []
    for vec_vars in vect_names:
        V=[]
        for vec_var in vec_vars:
            var_name = vec_var[4:]
            flag = False
            for name in data:
                case = name.split(',')
                if( str(var_name) in case[3]):
                    V.append(case[4])
                    flag = True
                    break
            if(not flag):
                V.append("Variable doesn't exist")
        A.append(V)
    return A

def affiche(resultat):
    for i in range(len(resultat)):
        for j in range(len(resultat[i])):
            print(resultat[i][j])
        print('\n')

affiche(names_to_dico(selection_ligne("MARTIN.TXT"), "Dico_M.csv")
