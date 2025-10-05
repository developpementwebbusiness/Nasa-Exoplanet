import pandas as pd

columnnames = ['Confirmation','OrbitalPeriod','TransitDur','TransitDepth','PlanetRadius','EquilibriumTemp','InsolationFlux','StellarEffectiveTemp','StellarRadius','RA','Dec']

#'''
fileKOI = 'Python/server/utils/Data/KOIHarmony.csv'
fileK2 = 'Python/server/utils/Data/K2Harmony.csv'
fileTOI = 'Python/server/utils/Data/TOIHarmony.csv'

dfk = pd.read_csv(fileKOI, skiprows=17) #f1=0.79
dfk.columns = columnnames
dfk2 = pd.read_csv(fileK2, skiprows=17) #bad ratio
dfk2.columns = columnnames
dft = pd.read_csv(fileTOI, skiprows=17) #0.57
dft.columns = columnnames
#'''

dfHar = pd.concat([dfk, dfk2, dft], axis = 0, ignore_index=True)


binary_replace = {'CANDIDATE':'True', 
                  'FALSE POSITIVE': 'False', 
                  'NOT DISPOSITIONED': 'False', 
                  'CONFIRMED': 'True',
                  'REFUTED': 'False',
                  'APC': 'False',
                  'CP': 'True',
                  'FP': 'False',
                  'FA': 'False',
                  'KP': 'True',
                  'PC': 'True'}

dfHar = dfHar.applymap(lambda x: binary_replace.get(x, x) if isinstance(x, str) else x)

dfHar.to_csv("Python/server/utils/Data/ExoHarmonius",index=False)