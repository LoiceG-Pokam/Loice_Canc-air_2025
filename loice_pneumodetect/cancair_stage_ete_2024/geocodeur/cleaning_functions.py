import pandas as pd
import re
from unidecode import unidecode

import numpy as np

voirie = ["RUE","COURS","COUR","VOIE","RUELLE","ESPLANADE",
          "PLACE","SQUARE","SQ","ROND POINT","PL",
          "IMPASSE", "ALLEE", "CHEMIN" ,"ROUTE","RTE","IMP","PROMENADE","ALLE","ALL",
          "AVENUE", "BOULEVARD","BD","AVE","BLD","BVD","BLV","AVN","BV",
          "FERME","DOMAINE","LIEU DIT","QUARTIER","QUR"]

##ATT AVEC COUR : A LA FOIS DANS LES ADRESSES ET COMME VOIRIE   

numeros = ["1","2","3","4","5","6","7","8","9","0"]

bruit = [ "RESIDENCE","CHEZ","HOPITAL","SDF","RES","MME","BAT","MAISON","MR","HOTEL",
         "LOTISSEMENT","CENTRE","QUARTIER","APPT","APT","SANTE", "RETRAITE", "HOP","TRANSFERT"]


def separer_alphanumerique(elements):
    """Fonction retournant les éléments d'adresses distinct 
    exemple : si "3eme" retourne "3 eme"
    
    input : liste des éléments de l'adresse séparés par des espaces 
    output :  l'adresse nettoyée
"""
    res = []
    for element in elements:
            
        # Vérifier si l'élément contient '.', '/', ou '''
        if any(c in element for c in ['.', '/', "'"]):
            res.append(element)  # Retourne l'élément sans le modifier, dans une liste pour garder le format cohérent
        else:
            # Utiliser une expression régulière pour séparer les lettres des chiffres
            matches = re.findall(r'(\d+|\D+)', element) # ([\w]+|\d+)
            res.extend(matches)
    return res

def prep_adresse(df):
    
    """ Fonction permettant d'uniformiser l'adresse pour pouvoir réaliser du TAL dessus (Traitement du language)
    input : dataframe avec les adresses brutes 
    output : dataframe avec adresses nettoyées""" 

    for i in df.index :
        adresse = df.at[i,"adresse"].split(" ")

        adresse_part = separer_alphanumerique(adresse)
        df.at[i,"adresse"] = unidecode(' '.join(adresse_part))
        df.at[i,'adresse'] = df.at[i,'adresse'].replace('.'," ").replace(',',' ').replace(';'," ").replace('-',' ').replace("'",' ').replace("/",' ')
    return df 

def find_attribute(attributes, text, is_number=False):
    """Fonction de recherche des attributs identifiés précédemment dans nos adresses"""
    if pd.notna(text):
        if is_number:
            numbers = re.findall(r'\d+', text)
            
            return ','.join(numbers) if numbers else ""
        else:
            found_attributes = [x for x in attributes if re.search(r'\b' + re.escape(x) + r'\.?\b', text)]

            return ','.join(found_attributes) if found_attributes else ""
        
        return ""

def filtre_num_voirie(df):
    """ Fonction filtrant les éléments trouvés : si de mutliples voiries ou numéros ont été identifié
    /!\ GRANDE RUE RUE ? """

    for i in df.index:

        adresse_part = df.at[i,"adresse"].split(" ")
        # adresse_part = separer_alphanumerique(adresse)

        nums = df.loc[i,"numeros"].split(',')


        voirie_liste = df.loc[i,'voirie'].split(',')
        if len(voirie_liste) >1 :
            df.at[i,'voirie'] = voirie_liste[0]
        voirie = voirie_liste[0]

        if voirie != "":

            if len(voirie.split(' '))>1: 
                voirie = voirie.split(' ')[0]

            pos_voirie = adresse_part.index(voirie)

            for num in nums : 
                if num != "":
                    if adresse_part.index(num)== pos_voirie -1:
                        df.loc[i,'numeros'] = num
        if len(df.loc[i,'voirie'].split(',')) >1 : 
            print(df.loc[i,'voirie'].split(','))
        else : 
            df.loc[i,"numeros"] = nums[0]

    return df


def find_pos_attributes(attributes, text,is_number=False,is_bruit=False):
    """ Fonction déterminant la position de l'attribut identifié dans l'adresse """
    found_attributes = []   
    if pd.notna(text):
        words = text.split()
        for i, word in enumerate(words):
            if is_number:
                if re.match(r'\d+', word):
                    return str(i+1)
            for attr in attributes.split(','): 

                if len(attr.split(' '))>1: 
                    if word == attr.split(' ')[0]:
                        found_attributes.append(str(i+1))

                if word == attr:    
                    found_attributes.append(str(i+1))
        # if is_voirie and len(found_attributes) > 1: 
        #     res = found_attributes[0] if found_attributes else ""
        if is_bruit : 
            res = ','.join(found_attributes) if found_attributes else ""
        else : 
            res = found_attributes[0] if found_attributes else ""
        return res
    

    
    
def filtre_bruit_to_voirie(df):
    """Fonction prenant en compte l'ambiguité de la langue françasie :
    Si un bruit voirie est en réalité une voirie on remplace la voirie par ce bruit """

    for i in df[df['bruit']!= ""].index:
        if df.loc[i,'pos_numeros'] != "": 

            ## Si un bruit voirie est en réalité une voirie on remplace la voirie  
            pos_num = int(df.loc[i,'pos_numeros'])
            pos_bruit_list = df.loc[i,'pos_bruit'].split(',')

            if len(pos_bruit_list) ==1: 
                if pos_num == int(pos_bruit_list[0]) -1 :
                    df.loc[i,'pos_voirie'] = int(pos_bruit_list[0])
                    df.loc[i,'voirie'] = df.loc[i,'bruit']
                    df.loc[i,'pos_bruit'] = ""
                    df.loc[i,'bruit'] = ""
                if df.loc[i,'bruit'] == 'CHEZ':
                    adresse_part = df.loc[i,'adresse'].split(' ')
                    if adresse_part[int(pos_bruit_list[0])-2] =='DE':
                        df.loc[i,'pos_bruit'] = ""
                        df.loc[i,'bruit'] = ""    
                if df.loc[i,'bruit'] == 'HOPITAL':
                    adresse_part = df.loc[i,'adresse'].split(' ')
                    if adresse_part[int(pos_bruit_list[0])-2] =='L' or adresse_part[int(pos_bruit_list[0])-2] =='DE': 
                        df.loc[i,'pos_bruit'] = ""
                        df.loc[i,'bruit'] = ""   
                if df.loc[i,'bruit'] == 'CENTRE':
                    adresse_part = df.loc[i,'adresse'].split(' ')
                    if adresse_part[int(pos_bruit_list[0])-2] =='DU': 
                        df.loc[i,'pos_bruit'] = ""
                        df.loc[i,'bruit'] = ""     
                if df.loc[i,'bruit'] == 'HOTEL':
                    adresse_part = df.loc[i,'adresse'].split(' ')
                    if adresse_part[int(pos_bruit_list[0])-2] =='DE' or adresse_part[int(pos_bruit_list[0])-2]=="L": 
                        df.loc[i,'pos_bruit'] = ""
                        df.loc[i,'bruit'] = ""
            else : 
                bruit = df.loc[i,'bruit'].split(',')
                for j, pos_bruit in enumerate(pos_bruit_list):
                    if pos_num == int(pos_bruit) -1:
                        pos_bruit_list.pop(j)
                        df.loc[i,'pos_voirie'] = pos_bruit
                        df.loc[i,'voirie'] = bruit[j-1]
                        bruit.pop(j-1)
                df.loc[i,'bruit'] = str(bruit).replace("['",'').replace("']",'').replace("'",'')

                df.loc[i,'pos_bruit'] = str(pos_bruit_list).replace("['",'').replace("']",'').replace("'",'')    
    return df 


liste_mot = ['VILLA','QUAI','PASSAGE','RESIDENCES','PARC','HAMEAU','SENTE','SENTIER','CITE']
def clean_wrong_voirie(df):
    """ Ambiguité de la langue française : élément pas toujours considérés comme voirie mais dans ce cas oui. 
    Fonction regarde si ces mots sont précédés par un numéro """

    for i in df.index :
        # if len(df.loc[i,'pos_voirie'].astype(str).split(','))>1 : 
        #     print(f"{i},{df.loc[i,'voirie']},{df.loc[i,'pos_voirie']}")
        adresse = df.loc[i,'adresse']
        if df.loc[i,'pos_numeros']!="":
            pos_num = int(df.loc[i,'pos_numeros'])
            for mot in liste_mot: 
                mot_cont = " "+mot+" "
                if mot_cont in adresse:
                    pos_mot = df.loc[i,'adresse'].split().index(mot)
                    if pos_num == pos_mot:
                        df.at[i,'voirie'] = mot
                        df.at[i,'pos_voirie'] = pos_mot +1
                        
        elif df.loc[i,'pos_bruit'] !="" and df.loc[i,'pos_voirie']!= "":
            if len(df.loc[i,'pos_bruit'].split(','))==1: 
                if int(df.loc[i,'pos_bruit']) == int(df.loc[i,'pos_voirie']):
                    df.at[i,'pos_bruit'] = ""
                    df.at[i,'bruit'] = ""

        
    return df

def find_pos_prc_attributes(attributes, text,is_number=False):
    found_attributes = []   
    long_adresse = len(text.split())
 
    if pd.notna(text):
        word = text.split()
        
        for i, word in enumerate(word):

            if is_number:
                if re.match(r'\d+', word):
                    pos = (i+1)/long_adresse*100
                    return str(np.round(pos,2))

                #return ','.join(numbers) if numbers else ""
            elif word in attributes:
                 
                #found_attributes = [x for x in attributes if re.search(r'\b' + re.escape(x) + r'\b', text)]
                pos = (i+1)/long_adresse*100
                found_attributes.append(str(np.round(pos,2)))
                #return ','.join(found_attributes) if found_attributes else ""
                
        return ','.join(found_attributes) if found_attributes else ""
    
    
def conditional(df_res, i, pos_bruit):
    adresse_part = df_res.loc[i,'adresse'].split(' ')

    if df_res.loc[i,'pos_numeros']=="" and df_res.loc[i,'pos_voirie']=="":
        df_res.at[i,'elem_adresse'] = ""
        df_res.at[i,'elem_bruit'] = df_res.loc[i,'adresse']
    #b 
    if df_res.loc[i,'pos_numeros']=="" and df_res.loc[i,'pos_voirie']!="":
        pos_voirie = int(df_res.loc[i,'pos_voirie'])
        if pos_bruit < pos_voirie:
            df_res.at[i,'elem_bruit'] = adresse_part[:pos_voirie-1]
            df_res.at[i,'elem_adresse'] = adresse_part[pos_voirie:]
            df_res.loc[i,'bruit_AV_AP'] ="AV"
        if pos_bruit > pos_voirie:
            df_res.at[i,'elem_bruit'] = adresse_part[pos_bruit-1:]
            df_res.at[i,'elem_adresse'] = adresse_part[pos_voirie:pos_bruit-1]
            df_res.loc[i,'bruit_AV_AP'] ="AP"
    #c
    if df_res.loc[i,'pos_numeros']!="" and df_res.loc[i,'pos_voirie']=="":
        pos_numeros = int(df_res.loc[i,'pos_numeros'])
        if pos_bruit < pos_numeros:
            df_res.at[i,'elem_bruit'] = adresse_part[:pos_numeros-1]
            df_res.at[i,'elem_adresse'] = adresse_part[pos_numeros:]
            df_res.loc[i,'bruit_AV_AP'] ="AV"
        if pos_bruit > pos_numeros:
            df_res.at[i,'elem_bruit'] = adresse_part[pos_bruit-1:]
            df_res.at[i,'elem_adresse'] = adresse_part[pos_numeros:pos_bruit-1]
            df_res.loc[i,'bruit_AV_AP'] ="AP"     
    #d
    if df_res.loc[i,'pos_numeros']!="" and df_res.loc[i,'pos_voirie']!="":
        pos_numeros = int(df_res.loc[i,'pos_numeros'])
        pos_voirie = int(df_res.loc[i,'pos_voirie'])
        if pos_bruit < pos_numeros:
            df_res.at[i,'elem_bruit'] = adresse_part[:pos_numeros-1]
            df_res.at[i,'elem_adresse'] = adresse_part[pos_voirie:]
            df_res.loc[i,'bruit_AV_AP'] ="AV"
        if pos_bruit > pos_numeros:
            df_res.at[i,'elem_bruit'] = adresse_part[pos_bruit-1:]
            df_res.at[i,'elem_adresse'] = adresse_part[pos_voirie:pos_bruit-1]
            df_res.loc[i,'bruit_AV_AP'] ="AP"
            
    return df_res


def filtre_elem_adresse(df):
    df_res = df.copy()
    df_res['elem_adresse'] = ""
    df_res['elem_bruit'] = ""
    df_res['bruit_AV_AP'] =""

    for i in df_res.index : 
        adresse_part = df_res.loc[i,'adresse'].split(' ')
        ##1 
        if df_res.loc[i,'bruit'] != "":
            if len(df_res.loc[i,'pos_bruit'].split(','))==1:

                pos_bruit = int(df_res.loc[i,'pos_bruit'])
                df_res = conditional(df_res,i, pos_bruit)
            
            if len(df_res.loc[i,'pos_bruit'].split(','))>1:
                pos_bruit_list = df_res.loc[i,'pos_bruit'].split(',')
                bruit_1 = int(pos_bruit_list[0])
                bruit_2 = int(pos_bruit_list[1])
                
                ## ON A DEUX BRUITS
                if len(pos_bruit_list) == 2:

                    ## SI ILS SE SUIVENT 
                    if bruit_1 == bruit_2 -1:
                        pos_bruit = bruit_1
                        df_res = conditional(df_res,i, pos_bruit)
                    else : 

                        ## SI ILS SE SUIVENT PAS MAIS ONT NUM ET VOIRIE 
                        if df_res.loc[i,'pos_numeros']!="":
                            pos_num = int(df_res.loc[i,'pos_numeros'])
                            if bruit_1 < pos_num and bruit_2 < pos_num:
                                pos_bruit = bruit_1
                                df_res = conditional(df_res,i, pos_bruit)
                            if bruit_1 > pos_num and bruit_2 > pos_num:
                                pos_bruit = bruit_1
                                df_res = conditional(df_res,i, pos_bruit)
                            if bruit_1 < pos_num and bruit_2 > pos_num : 
                                df_res.at[i,'elem_adresse'] = adresse_part[pos_num+1:bruit_2-1]
                                df_res.at[i,'elem_bruit'] = adresse_part[:pos_num-1]+adresse_part[bruit_2-1:]
                                df_res.at[i,'bruit_AV_AP'] ="AV_AP"
                if len(pos_bruit_list) == 3:
                    if df_res.loc[i,'pos_numeros']!="":
                        pos_num = int(df_res.loc[i,'pos_numeros'])

                        if all(int(elem) > pos_num for elem in pos_bruit_list) or all(int(elem) < pos_num for elem in pos_bruit_list): 
                            pos_bruit = bruit_1
                            df_res = conditional(df_res,i, pos_bruit)
                        if bruit_1 == bruit_2 -1:
                            bruit_3 = int(pos_bruit_list[2])
                            df_res.at[i,'elem_adresse'] = adresse_part[pos_num+1:bruit_3-1]
                            df_res.at[i,'elem_bruit'] = adresse_part[:pos_num-1]+adresse_part[bruit_3-1:]
                            df_res.at[i,'bruit_AV_AP'] ="AV_AP"
                            
                        if df_res.at[i,'elem_bruit']=="": 
                            sup = [int(elem) for elem in pos_bruit_list if int(elem) > pos_num]
                            # inf = [int(elem) for elem in pos_bruit_list if int(elem) < pos_num]
                            bruit_3 = sup[0]
                            df_res.at[i,'elem_adresse'] = adresse_part[pos_num+1:bruit_3-1]
                            df_res.at[i,'elem_bruit'] = adresse_part[:pos_num-1]+adresse_part[bruit_3-1:]
                            df_res.at[i,'bruit_AV_AP'] ="AV_AP"

                if len(pos_bruit_list) == 4:
                    if df_res.loc[i,'pos_numeros']!="":
                        pos_num = int(df_res.loc[i,'pos_numeros'])

                        if all(int(elem) > pos_num for elem in pos_bruit_list) or all(int(elem) < pos_num for elem in pos_bruit_list): 
                            pos_bruit = bruit_1
                            df_res = conditional(df_res,i, pos_bruit)
                            
                        if df_res.at[i,'elem_bruit']=="":
                            sup = [int(elem) for elem in pos_bruit_list if int(elem) > pos_num]
                            # inf = [int(elem) for elem in pos_bruit_list if int(elem) < pos_num]
                            bruit_3 = sup[0]
                            df_res.at[i,'elem_adresse'] = adresse_part[pos_num+1:bruit_3-1]
                            df_res.at[i,'elem_bruit'] = adresse_part[:pos_num-1]+adresse_part[bruit_3-1:]
                            df_res.at[i,'bruit_AV_AP'] ="AV_AP"


        ##2 
        if df_res.loc[i,'bruit'] == "":
            if df_res.loc[i,'pos_numeros']=="" and df_res.loc[i,'pos_voirie']=="":
                df_res.at[i,'elem_adresse'] = df_res.loc[i,'adresse']
            #b 
            if df_res.loc[i,'pos_numeros']=="" and df_res.loc[i,'pos_voirie']!="":
                pos_voirie = int(df_res.loc[i,'pos_voirie'])
                df_res.at[i,'elem_adresse'] = adresse_part[pos_voirie:]
            #c
            if df_res.loc[i,'pos_numeros'] !="" and df_res.loc[i,'pos_voirie']=="":
                pos_numeros = int(df_res.loc[i,'pos_numeros'])
                df_res.at[i,'elem_adresse'] = adresse_part[pos_numeros:]
            #d
            elif df_res.loc[i,'pos_numeros'] !="" and df_res.loc[i,'pos_voirie']!="":
                pos_voirie = int(df_res.loc[i,'pos_voirie'])
                df_res.at[i,'elem_adresse'] = adresse_part[pos_voirie:]

        df_res.at[i,'elem_adresse'] = ' '.join(df_res.at[i,'elem_adresse'])
        df_res.at[i,'elem_bruit'] = ' '.join(df_res.at[i,'elem_bruit'])

    return df_res

