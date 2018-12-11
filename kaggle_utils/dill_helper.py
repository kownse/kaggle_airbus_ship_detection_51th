# -*- coding: utf-8 -*-
import dill
import pickle

def save_obj(obj, name ):
    with open('obj/'+ name + '.dl', 'wb') as f:
        dill.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def load_obj(name ):
    with open('obj/' + name + '.dl', 'rb') as f:      
        return dill.load(f)

