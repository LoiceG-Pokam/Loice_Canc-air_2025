def geocode(df):
    """Geocode dataframe with Etalab addok"""
    print("Proceed to geocode on address...")
    for i in df.index:
        try:
            # get json response
            r = requests.get('https://addok-data.curie.net/search?q='+df["requete"][i])
            response = r.json()
            if i%100==0 : 
                print(f"Proceed geocoding at the {i}th row")
            if response["features"]!=[]:
                # parse json to insert value in dataframe
                #df.at[i, 'nip'] = str(df["pseudo_provisoire"][i])
                df.at[i, 'x'] = str(response["features"][0]["geometry"]["coordinates"][0])
                df.at[i, 'y'] = str(response["features"][0]["geometry"]["coordinates"][1])
                df.at[i, 'score'] = str(response["features"][0]["properties"]["score"])

                if float(df["score"][i])<0.4:
                    df.at[i, 'trust_score'] = 'low'
                elif float(df["score"][i])>0.4 and float(df["score"][i])<0.65:
                    df.at[i, 'trust_score'] = 'middle'
                elif float(df["score"][i])>0.65 and float(df["score"][i])<0.9:
                    df.at[i, 'trust_score'] = 'middle'
                else:
                    df.at[i, 'trust_score'] = 'high'

                df.at[i, 'street'] = str(response["features"][0]["properties"]["name"]).replace("'", " ").upper()
                df.at[i, 'city'] = str(response["features"][0]["properties"]["city"]).replace("'", " ").upper()
                df.at[i, 'pc_city'] = str(response["features"][0]["properties"]["postcode"])
                # df.at[i, 'ic_city'] = str(response["features"][0]["properties"]["citycode"])

                context = (str(response["features"][0]["properties"]["context"]).replace("'", " ")).split(",")
                df.at[i, 'code_dept'] = context[0]
                # df.at[i, 'dept'] = context[1]

                # if len(df["code_dept"][i])==2:
                #     df.at[i, 'reg'] = context[2]
                # else:
                #     df.at[i, 'reg'] = "other"
                #df.at[i, 'code_country'] = str(df["pays"][i])
                df.at[i, 'address'] = str(response["features"][0]["properties"]["label"]).replace("'", " ").upper()
                
                    
                if df.loc[i,"codepost"].astype(str) == df["pc_city"][i].astype(str):
                    df.at[i, 'same_city'] = "true"
                else:
                    df.at[i, 'same_city'] = "false" 
                    

                    
                df.at[i, 'date_geoloc'] = str(datetime.date.today())
                #df.at[i, 'etalab_version'] = str(response["licence"])
                #df.at[i, 'ban_version'] = "2021-04-27"
            else:
                pass
        except:
            pass
    return df