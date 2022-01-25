#!/usr/bin/env python
# coding: utf-8

# # **Progetto FAD**

# ## Dataset

# In questa analisi verrà utilizzato il <b> Dataset of genomes and genetics </b>, proposto per la ML challenge Hackerearth 2021. <br>
# Un disordine genetico è una condizione medica dovuta solitamente a mutazioni nel DNA o a modifiche nel numero o nella struttura cromosomica. <br>
#     La maggior parte delle malattie genetiche derivano da mutazioni ereditarie.
#     Per questo individuare tempestivamente la malattia che un bambino ha sviluppato o potrebbe sviluppare, grazie all'utilizzo di informazioni generiche e sullo stato di salute del bambino e dei genitori, potrebbe essere estremamente utile per individuare la malattia e per poter dunque attivare tempestivamente trattamenti e controlli preventivi. <br>
#     In particolare lo studio si propone di individuare se il bambino ha sviluppato/potrebbe sviluppare disturbi dovuti all'<b>eredità genetica mitocondriale</b> (neuropatia ottica ereditaria di Leber, sindrome di Leigh, miopatia mitocondriale) ad un <b>singolo gene</b> (fibrosi cistica, malattia di Tay-Sachs e emocromatosi) o di tipo <b>multifattoriale</b> (diabete, cancro, Alzheimer).

# Per prima cosa importiamo il Dataset e ne visualizziamo le prime 5 righe per capirne la struttura. 

# In[1]:


import pandas as pd

#visualizzo tutte le colonne del dataset
pd.set_option('display.max_columns', None)

#importo il dataset
data = pd.read_csv('genetic_disorder.csv')

data.head(5)


# Notiamo che i nomi delle variabili non sono ottimali, quindi per prima cosa le rinominiamo in maniera più funzionale.

# In[2]:


#rinomino le colonne
data.rename(columns={'Patient Id': 'patient_id', 'Patient Age': 'patient_age', "Genes in mother's side":'mother_gene',
       'Inherited from father': 'father_gene', 'Maternal gene':'maternal_gene', 'Paternal gene':'paternal_gene',
       'Blood cell count (mcL)':'blood_cell', "Patient First Name":'patient_first_name','Family Name':'family_name',
       "Father's name":'father_name',
       "Mother's age":'mother_age', "Father's age":'father_age', 'Institute Name':'institute_name', 'Location of Institute':'location_institute','Status':'status', 
       'Respiratory Rate (breaths/min)': 'respiratory_rate','Heart Rate (rates/min': 'heart_rate', 'Test 1': 'test1', 
       'Test 2':'test2', 'Test 3':'test3', 'Test 4':'test4','Test 5':'test5', 'Parental consent':'parental_consent','Follow-up':'follow_up', 'Gender':'gender', 
       'Birth asphyxia':'birth_asphyxia', 'Autopsy shows birth defect (if applicable)':'autopsy_birth_defect', 
       'Place of birth':'place_of_birth', 'Folic acid details (peri-conceptional)':'folic_acid_details',
       'H/O serious maternal illness': 'maternal_illness', 'H/O radiation exposure (x-ray)' :'radiation_exposure',
       'H/O substance abuse':'substance_abuse', 'Assisted conception IVF/ART':'assisted_conception',
       'History of anomalies in previous pregnancies':'anomalies_in_pregnancies','No. of previous abortion':'num_abortion', 
       'Birth defects': 'birth_defects','White Blood cell count (thousand per microliter)':'white_blood_cell', 
       'Blood test result':'blood_test_result','Symptom 1':'symptom1', 'Symptom 2':'symptom2', 'Symptom 3':'symptom3', 
       'Symptom 4':'symptom4', 'Symptom 5':'symptom5','Genetic Disorder':'genetic_disorder', 
       'Disorder Subclass':'disorder_subclass'}, inplace=True)


# A questo punto visualizziamo tutte le informazioni generali sul dataset.

# In[3]:


data.info()


# Notiamo la presenza di <b>22083</b> record e <b>45</b> colonne. <br>
#     Inoltre osserviamo la presenza di molti <b>NaN</b>. <br>
#     Il significato delle singole colonne è spiegato nel file allegato, questo ci permette subito di individuare colonne che non sono utili ai fini del nostro studio, come l'ID del paziente, il nome del padre, il cognome del bambino, il consenso, gli indirizzi e i nomi degli istituti. <br>
#     Inoltre rimuoviamo la colonna disorder_subclass perchè non rientra nell'obiettivo del nostro studio. <br>

# In[4]:


#rimuovo le colonne non utili
data.drop("patient_id",axis=1, inplace=True)
data.drop("patient_first_name",axis=1, inplace=True)
data.drop("family_name",axis=1, inplace=True)
data.drop("father_name",axis=1, inplace=True)
data.drop("location_institute", axis=1, inplace=True)
data.drop("institute_name", axis=1, inplace=True)
data.drop("parental_consent", axis=1, inplace=True)
data.drop("disorder_subclass", axis=1, inplace=True)

#visualizzo le informazioni sul dataset aggiornato
data.info()


# Abbiamo quindi ridotto il numero di colonne a <b>37</b>. <br>
# Dato che vogliamo arrivare ad effettuare una classificazione supervisionata, rimuovo dal dataset i record che presentano NaN nella colonna genetic_disorder e che quindi non risultano etichettati, dopodiché effettuo il reset degli indici.

# In[5]:


#rimuovo i dati non etichettati
data.dropna(subset=['genetic_disorder'],inplace=True)

#reset indici
data.reset_index(drop=True, inplace= True)

data.info()


# Adesso abbiamo dunque <b>19937</b> record, ma notiamo che ci sono ancora molti NaN.
# In ogni caso visualizziamo prima i possibili valori per ogni colonna, nel caso di variabili categoriche, il range di variabilità dei dati nel caso di variabili numeriche.

# In[6]:


import numpy as np
from prettytable import PrettyTable

table = PrettyTable()
table.field_names = ['Colonna', 'Valori']

for col in data.columns:
    if data.dtypes[col]=='float64':
        #se è di tipo float stampo il range
        table.add_row([col, [np.nanmin(data[col]),np.nanmax(data[col])]]) 
    else:   
        #se è categorica stampo i valori che puo' assumere
        table.add_row([col, data[col].unique()])
        
print(table)


# Notiamo ambiguità nei valori di autopsy_birth_defect (qual è la differenza tra 'Not applicable','No','None' e 'Yes'?), inoltre notiamo che nei cinque test i valori massimo e minimo coincidono, quindi il valore assunto è costante e hanno inoltre varianza nulla. Non risultano quindi essere significativi per il nostro studio e possiamo dunque rimuovere le colonne suddette. <br> Sostituiamo inoltre i valori '-' presenti in radiation_exposure e substance_abuse, 'No record' e 'Not available' presenti in birth_asphyxia e 'inconclusive' in blood_test_result con NaN.

# Inoltre notiamo che le classi di genetic_disorder sono tre: <b>Mitochondrial genetic inheritance disorders</b>, <b>Multifactorial genetic inheritance disorders</b> e <b>Single-gene inheritance diseases</b>, che saranno le classi in cui vogliamo ripartire i nostri pazienti come obiettivo dello studio.

# In[7]:


#rimuovo ulteriori colonne ambigue o non significative
data.drop("autopsy_birth_defect",axis=1, inplace=True)
data.drop("test1",axis=1, inplace=True)
data.drop("test2",axis=1, inplace=True)
data.drop("test3",axis=1, inplace=True)
data.drop("test4",axis=1, inplace=True)
data.drop("test5",axis=1, inplace=True)


# In[8]:


#sostituisco i termini che indicano assenza dei dati con dei NaN
data['radiation_exposure'] = data["radiation_exposure"].replace({'-':np.nan, 'Not applicable':np.nan })
data["substance_abuse"] = data["substance_abuse"].replace({'-':np.nan, 'Not applicable':np.nan })
data["birth_asphyxia"] = data["birth_asphyxia"].replace({'No record':np.nan, 'Not available':np.nan})
data['blood_test_result'] = data['blood_test_result'].replace({'inconclusive':np.nan})


# Visualizziamo nuovamente i dati aggiornati.

# In[9]:


data.info()


# Adesso abbiamo lo stesso numero di osservazioni ma solamente 31 colonne.<br>
# Notiamo subito l'elevato numero di NaN presenti nella colonne birth_asphyxia, radiation_exposure e substance_abuse dunque rimuoviamo queste colonne dal nostro dataset.

# In[10]:


#rimuovo le colonne in cui mancano più del 50% dei dati
data.drop("birth_asphyxia",axis=1, inplace=True)
data.drop("radiation_exposure",axis=1, inplace=True)
data.drop("substance_abuse",axis=1, inplace=True)


# Vogliamo provare adesso a rimuovere le righe contenenti più del 10% di NaN. Le colonne (esclusa quella delle etichette), sono 30 quindi rimuoviamo quelle aventi almeno 3 NaN.

# In[11]:


#rimuovo le righe contenenti più di 3 NaN
data = data[data.isnull().sum(axis=1) < 3]
data.reset_index(drop=True, inplace= True)
data.info()


# Abbiamo adesso <b>9862</b> record, ma sono presenti ancora alcuni NaN. Cerchiamo allora di sostituirli per avere un dataset completo e poter proseguire il nostro studio.

# In[12]:


for col in data.columns:
    if data.dtypes[col]=='float64' and (col not in ['symptom1','symptom2','symptom3','symptom4','symptom5']):
        #se i dati non sono binari né categorici sostituisco la media delle colonne ai NaN
        x = data[col].mean()
        data[col] = data[col].replace(np.nan, x)
    else:
        #se i dati sono binari o categorici sostituisco il valore più frequente
        x = data[col].mode()[0]
        data[col] = data[col].replace(np.nan, x)
data.info()


# ## Statistica descrittiva

# Adesso che il nostro dataset è completo posso iniziare a effettuare delle statistiche e a visualizzare i primi grafici e le prime informazioni relative ai nostri dati.

# In[13]:


data.describe()


# Da una prima descrizione dei dati otteniamo le statistiche relative alle variabili numeriche.
# Adesso costruiamo degli istogrammi per visualizzare anche i dati categorici, osservando anche la loro distribuzione nelle tre classi.

# Iniziamo visualizzando la distribuzione dei dati nelle tre categorie.

# In[14]:


import matplotlib.pyplot as plt

#istogramma della distribuzione dei dati nelle tre categorie
plt.figure(figsize=(13,7))
hist = data['genetic_disorder'].value_counts(normalize = True).sort_index()
plt.bar(hist.index, hist.values)
plt.title(col)
plt.show()


# Notiamo subito che le classi sono molto sbilanciate e che questo deve essere ovviamente tenuto in conto nella fase di training.

# Visualizziamo le distribuzioni di tutti i dati, utilizzando istogrammi e densità. Nel caso di variabili continue o con molti record applichiamo la regola di Sturges per il numero di bin : <br>
# 
# \begin{equation}
#  \# bin = 3.3*log(\# osservazioni) 
# \end{equation}
# 
# 
# Costruiamo inoltre un array <b>symptoms</b> poiché, nonostante i suoi valori siano del tipo `float64`, essendo binarie andranno trattate come le variabili categoriche.
# 

# In[15]:


#costruisco il vettore symptom
symptoms = ['symptom1','symptom2','symptom3','symptom4','symptom5']


for col in data.columns:
    
    if data.dtypes[col]=='float64' and (col not in symptoms):
        #se la variabile non è binaria stampo l'istogramma 
        #usando il numero di bin calcolato con la regola di Sturges
        plt.figure(figsize=(6,4))
        bins_sturges=int(3.3*np.log(len(data[col])))
        plt.hist(data[col], bins=bins_sturges,density=True)
        plt.title(col)
        plt.show()
        
    #escludo l'ultima colonna che abbiamo già visto al passo precedente
    elif col != 'genetic_disorder':    
        #se la variabile è binaria stampo gli istogrammi normalizzati
        plt.figure(figsize=(6,4))
        hist = data[col].value_counts(normalize = True).sort_index()
        plt.bar(hist.index, hist.values)
        plt.title(col)
        plt.show()    


# Adesso effettuiamo gli stessi grafici precedenti ma evidenziando l'appartenenza alle tre classi. 

# In[16]:


#disegno i grafici, evidenziando le classi di appartenenza
for col in data.columns:
    if data.dtypes[col]=='float64' and (col not in symptoms):
        data.groupby('genetic_disorder')[col].plot.hist(width=3, alpha=0.5, density=True, figsize=(12,6))
        data.groupby('genetic_disorder')[col].plot.density()
        plt.title(col)
        plt.legend()
        plt.show()
    elif col != 'genetic_disorder':
        plt.figure(figsize=(6,4))
        pd.crosstab(data[col],data['genetic_disorder']).plot.bar(stacked=True)
        plt.show()


# Già questa prima analisi ci suggerisce varie possibili relazioni tra le fature e le classi. Per esempio i sintomi 4 e 5 sono presenti una percentuale molto alta dei bambini che soffrono di disordini multifattoriali.

# Proviamo adesso a capire se potrebbero intercorrere relazioni tra le feature. Iniziamo visualizzando gli scatterplot tra le coppie di feature.

# In[17]:


import seaborn as sns

#scatterplot tra feature
sns.pairplot(data)
plt.show()


# Non sembrano esserci particolari rapporti tra le variabili numeriche del dataset (quelle categoriali non sono inserite dal pairplot).

# Proviamo allora a costruire delle cross-table per capire le relazioni tra le variabili categoriali e binarie.

# In[18]:


for col1 in data.columns:
    for col2 in data.columns:
        if col1 != col2 and col1 != 'genetic_disorder' and col2 != 'genetic_disorder':
            if data.dtypes[col1] == 'object' or (col1 in symptoms):
                if data.dtypes[col2] == 'object' or (col2 in symptoms):
                    pd.crosstab(data[col1], data[col2]).plot.bar(stacked=True)
                    tab = pd.crosstab(data[col1], data[col2])
                    tab


# Notiamo che sembra esserci correlazione tra la presenza di un difetto genetico nella madre e nel ramo materno e nell'assenza di un difetto genetico nel padre e nel ramo paterno. Proviamo a vedere se effettivamente c'è una possibile relazione attraverso delle regressioni.

# Prima di continuare  dobbiamo però sostituire le nostre variabili categoriali con variabili numeriche. Introduco inoltre variabili dummy nel caso del sesso in quanto le tre classi (male, Female, Ambiguous) non sono ordinabili.

# In[19]:


#sostituisco le variabili categoriche con variabili dummy o numeriche
data["mother_gene"] = data["mother_gene"].replace({'Yes':1, 'No': 0})
data["father_gene"] = data["father_gene"].replace({'Yes':1, 'No': 0})
data["maternal_gene"] = data["maternal_gene"].replace({'Yes':1, 'No': 0})
data["paternal_gene"] = data["paternal_gene"].replace({'Yes':1, 'No': 0})
data["status"] = data["status"].replace({'Alive':0, 'Deceased':1})
data["respiratory_rate"] = data["respiratory_rate"].replace({'Normal (30-60)':0, 'Tachypnea':1})
data["heart_rate"] = data["heart_rate"].replace({'Normal':0, 'Tachycardia':1})
data["follow_up"] = data["follow_up"].replace({'Low':0, 'High':1})
data["place_of_birth"] = data["place_of_birth"].replace({'Institute':0, 'Home':1})
data["folic_acid_details"] = data["folic_acid_details"].replace({'Yes':1, 'No': 0})
data["maternal_illness"] = data["maternal_illness"].replace({'Yes':1, 'No': 0})
data["assisted_conception"] = data["assisted_conception"].replace({'Yes':1, 'No': 0})
data["anomalies_in_pregnancies"] = data["anomalies_in_pregnancies"].replace({'Yes':1, 'No': 0})
data["birth_defects"] = data["birth_defects"].replace({'Multiple':1, 'Singular': 0})
data["blood_test_result"] = data["blood_test_result"].replace({'abnormal':2,'slightly abnormal':1, 'normal': 0})

is_male = []
is_female = []

for row in range(len(data)):
    if data.iloc[row]['gender'] == 'Male':
        is_male.append(1)
    else:
        is_male.append(0)
        
    if data.iloc[row]['gender'] == 'Female':
        is_female.append(1)
    else:
        is_female.append(0)
data['is_female'] = is_female
data['is_male'] = is_male
data.drop("gender",axis=1, inplace=True)


# Applico adesso la regressione, logistica nel caso di variabili binarie e lineare nel caso di variabili non binarie, per capire se effettivamente sono presenti possibili relazioni tra le variabili.

# In[20]:


from statsmodels.formula.api import logit
from statsmodels.formula.api import ols


for regr in data.columns:
    #costruisco la stringa da passare alla funzione di regressione
    if regr != 'genetic_disorder':
        stringa = regr + ' ~ '
        for col in data.columns:
            if (col!= regr) and (col != 'genetic_disorder'):
                stringa = stringa + col + ' + '
        stringa = stringa[:-2] #rimuovo l'ultimo segno '+'
        
        if len(data[regr].unique()) > 2:
            #se la feature non è binaria applico la regressione lineare
            model = ols(stringa, data).fit()
            descr = model.summary()
            print(descr)
            
        else:
            #se la feature è binaria applico la regressione logistica
            model = logit(stringa, data).fit()
            descr = model.summary()
            print(descr)


# Noto tuttavia che i valori di $R^2$ sono molto piccoli e nessuno di questo risulta essere significativo. Quindi non posso ridurre il numero delle mie feature con questa tecnica.

# ## Classificazione

# Il dataset è pronto per effettuare la classificazione che ci eravamo proposti.

# In[21]:


data.info()


# Cercheremo adesso di effettuare la classificazione dei dati, abbiamo a disposizione **28** feature e **9862** record. <br>
# Dobbiamo però ancora rimediare al dataset non bilanciato. Procederò dunque con entrambe le tecniche di **undersampling** e **oversampling**, applicando più algoritmi di classificazione e confrontando i risultati.
# Prima creo una copia del dataset per non rischiare di modificarlo e verifico il numero di campioni per classe.

# In[22]:


#creo una copia del dataset
data_cl = data.copy()
data_cl.head()

#controllo il numero di record per ogni classe
print(data['genetic_disorder'].value_counts())


# ### Undersampling

# Inizio con la tecnica dell'undersampling, in particolare ridurrò il numero di elementi del dataset utilizzandone il numero minimo possibile perchè siano equinumerosi. <br>
# Costruirò quindi un nuovo dataset avente **1041** campioni per classe.

# In[23]:


#creo il dataset undersampling

#costruisco il dataset
data_cl_u =  data_cl[data_cl.genetic_disorder=='Multifactorial genetic inheritance disorders'].copy()
#aggiungo lo stesso numero di campioni per ogni classe
data_cl_u = data_cl_u.append(data_cl[data_cl.genetic_disorder=='Single-gene inheritance diseases'].sample(n=len(data_cl[data_cl.genetic_disorder=='Multifactorial genetic inheritance disorders'])).copy())
data_cl_u = data_cl_u.append(data_cl[data_cl.genetic_disorder=='Mitochondrial genetic inheritance disorders'].sample(n=len(data_cl[data_cl.genetic_disorder=='Multifactorial genetic inheritance disorders'])).copy())

#verifico che le classi abbiano effettivamente lo stesso numero di elementi
print(data_cl_u.genetic_disorder.value_counts())


# Adesso che il dataset è pronto posso effettuare la suddivisione in training set e control set.

# In[24]:


from sklearn.model_selection import train_test_split

#suddivido il dataset in dati di training e di test
train_data_u, test_data_u = train_test_split(data_cl_u, test_size=0.25)


# ### Oversampling

# Un'altra tecnica che è possibile utilizzare è quella dell'oversampling. Stavolta invece di togliere dati ne verranno aggiunti, replicando quelli già presenti.
# Costruirò quindi un dataset che avrà **5015** elementi per classe.

# In[25]:


#creo il dataset oversampling

#aumento i dati della classe meno numerosa
data_prov1 = data_cl[data_cl.genetic_disorder=='Multifactorial genetic inheritance disorders'].copy()
for i in range(4):
    data_prov1 = data_prov1.append(data_cl[data_cl.genetic_disorder=='Multifactorial genetic inheritance disorders'].copy())
    
#aumento i dati della classe intermedia
data_prov2 = data_cl[data_cl.genetic_disorder=='Single-gene inheritance diseases'].copy()
for i in range(2):
    data_prov2 = data_prov2.append(data_cl[data_cl.genetic_disorder=='Single-gene inheritance diseases'].copy())

#creo il nuovo dataset 
data_cl_o = data_cl[data_cl.genetic_disorder=='Mitochondrial genetic inheritance disorders'].copy()
data_cl_o = data_cl_o.append(data_prov1.sample(n=len(data_cl[data_cl.genetic_disorder == 'Mitochondrial genetic inheritance disorders'])))
data_cl_o = data_cl_o.append(data_prov2.sample(n=len(data_cl[data_cl.genetic_disorder == 'Mitochondrial genetic inheritance disorders'])))

#verifico che le classi abbiano effettivamente lo stesso numero di elementi
print(data_cl_o.genetic_disorder.value_counts())

#suddivido il dataset in dati di training e di test
train_data_o, test_data_o = train_test_split(data_cl_o, test_size=0.25)


# ### Alberi di classificazione

# Applico il primo algoritmo di classificazione, fisso la profondità degli alberi a 15.

# In[26]:


from sklearn.tree import DecisionTreeClassifier 

#effettuo la classificazione con gli alberi di classificazione
depth = 15
dt = DecisionTreeClassifier(max_depth=depth)

#dataset undersampling
dt.fit(train_data_u.drop('genetic_disorder',axis=1),train_data_u['genetic_disorder'])
#calcolo le accuracy di training e di test
acc_train_tree_u = dt.score(train_data_u.drop('genetic_disorder',axis=1), train_data_u['genetic_disorder'])
acc_test_tree_u = dt.score(test_data_u.drop('genetic_disorder',axis=1), test_data_u['genetic_disorder'])

#dataset oversampling
dt.fit(train_data_o.drop('genetic_disorder',axis=1),train_data_o['genetic_disorder'])
#calcolo le accuracy di training e test
acc_train_tree_o = dt.score(train_data_o.drop('genetic_disorder',axis=1), train_data_o['genetic_disorder'])
acc_test_tree_o = dt.score(test_data_o.drop('genetic_disorder',axis=1), test_data_o['genetic_disorder'])


# In[27]:


#stampo i risultati ottenuti
print('Dataset undersampling:')
print('Accuracy di training albero di regressione con profondità {}: {}'.format(depth, acc_train_tree_u))
print('Accuracy di test albero di regressione con profondità {}: {}'.format(depth, acc_test_tree_u))
print('\n')
print('Dataset oversampling')
print('Accuracy di training albero di regressione con profondità {}: {}'.format(depth,acc_train_tree_o))
print('Accuracy di test albero di regressione con profondità {}: {}'.format(depth,acc_test_tree_o))


# Si ottengono buoni risultati con il dataset ottenuto mediante oversampling, meno buoni sono invece i risultati che si ottengono con il dataset ottenuto mediante undersampling, specialmente in fase di test.

# ### Random forest

# Un altro posssibile algoritmo è quello del **Random forest**, in cui utilizzo molti alberi con una piccola profondità e scelgo a quale classe assegnare il campione mediante tecnica di voting.

# In[28]:


from sklearn.ensemble import RandomForestClassifier

#introduco un seed perchè il mio algoritmo dia sempre gli stessi esiti
np.random.seed(3333)

#introduco l'oggetto rand_forest scegliendo profondità e numero di alberi
rf_depth = 3
trees = 300
rand_forest = RandomForestClassifier(max_depth = rf_depth, n_estimators = trees)

#dataset undersampling
rand_forest.fit(data_cl_u.drop('genetic_disorder',axis=1),data_cl_u['genetic_disorder'])
#calcolo le accuracy
acc_train_rf_u= rand_forest.score(train_data_u.drop('genetic_disorder',axis=1), train_data_u['genetic_disorder'])
acc_test_rf_u = rand_forest.score(test_data_u.drop('genetic_disorder',axis=1), test_data_u['genetic_disorder'])

#dataset oversampling
rand_forest.fit(data_cl_o.drop('genetic_disorder',axis=1),data_cl_o['genetic_disorder'])
#calcolo le accuracy
acc_train_rf_o = rand_forest.score(train_data_o.drop('genetic_disorder',axis=1), train_data_o['genetic_disorder'])
acc_test_rf_o = rand_forest.score(test_data_o.drop('genetic_disorder',axis=1), test_data_o['genetic_disorder'])


# In[29]:


#stampo i risultati ottenuti
print('Dataset undersampling:')
print('Accuracy di training random forest con {} alberi di profondità {}: {}'.format(trees, rf_depth, acc_train_rf_u))
print('Accuracy di test random forest con {} alberi di profondità {}: {}'.format(trees, rf_depth, acc_test_rf_u))
print('\n')
print('Dataset oversampling')
print('Accuracy di training random forest con {} alberi di profondità {}: {}'.format(trees,rf_depth,acc_train_rf_o))
print('Accuracy di test random forest con {} alberi di profondità {}: {}'.format(trees, rf_depth,acc_test_rf_o))


# I risultati non sono ottimali, otteniamo due classificatori deboli.

# ### K-nn

# L'ultimo algoritmo è il **K-nn**, questo consiste nel classificare un campione individuando i k campioni che distano meno da esso e assegnare a questo la classe più frequente tra i K.

# Calcolo il miglior valore di k per ogni dataset.

# In[30]:


from sklearn.neighbors import KNeighborsClassifier as KNN


# In[31]:


#miglior k per il dataset undersampling
test_accuracies = []

#range k
k_values = range(1,25) 

#calcolo le accuracy al variare di k
for k in k_values:
    knn_k = KNN(n_neighbors = k)
    knn_k.fit(train_data_u.drop('genetic_disorder',axis=1), train_data_u.genetic_disorder)
    test_accuracies.append(knn_k.score(test_data_u.drop('genetic_disorder',axis=1), test_data_u.genetic_disorder))

#stampo l'accuracy e il valore di k
best_index = np.argmax(test_accuracies)
best_k_u = k_values[best_index]
best_accuracy_u = test_accuracies[best_index]
print("Migliore accuracy di test: {}".format(best_accuracy_u))
print("Migliore k: {}".format(best_k_u))

#stampo il grafico (k, accuracy k-nn)
plt.figure(figsize=(10,10))
plt.plot(k_values,test_accuracies)
plt.grid()
plt.show()


# In[32]:


#miglior k per il dataset oversampling
test_accuracies = []

#range k
k_values = range(1,25) 

#calcolo le accuracy al variare di k
for k in k_values:
    knn_k = KNN(n_neighbors=k)
    knn_k.fit(train_data_o.drop('genetic_disorder',axis=1), train_data_o.genetic_disorder)
    test_accuracies.append(knn_k.score(test_data_o.drop('genetic_disorder',axis=1), test_data_o.genetic_disorder))

#stampo l'accuracy e il valore di k
best_index = np.argmax(test_accuracies)
best_k_o = k_values[best_index]
best_accuracy_o = test_accuracies[best_index]
print("Migliore accuracy di test: {}".format(best_accuracy_o))
print("Migliore k: {}".format(best_k_o))

#stampo il grafico (k, accuracy k-nn)
plt.figure(figsize=(10,10))
plt.plot(k_values,test_accuracies)
plt.grid()
plt.show()


# In[33]:


#introduco l'oggetto knn e specifico il numero k per ogni dataset, ricavato nel procedimento precedente 
k_o = best_k_o
k_u = best_k_u
knn_o = KNN(n_neighbors = k_o)
knn_u = KNN(n_neighbors = k_u)

#dataset undersampling
knn_u.fit(train_data_u.drop('genetic_disorder',axis=1), train_data_u.genetic_disorder)
#calcolo le accuracy
acc_train_knn_u= knn_u.score(train_data_u.drop('genetic_disorder',axis=1), train_data_u['genetic_disorder'])
acc_test_knn_u = knn_u.score(test_data_u.drop('genetic_disorder',axis=1), test_data_u['genetic_disorder'])

#dataset oversampling
knn_o.fit(train_data_o.drop('genetic_disorder',axis=1), train_data_o.genetic_disorder)
#calcolo le accuracy
acc_train_knn_o= knn_o.score(train_data_o.drop('genetic_disorder',axis=1), train_data_o['genetic_disorder'])
acc_test_knn_o = knn_o.score(test_data_o.drop('genetic_disorder',axis=1), test_data_o['genetic_disorder'])


# In[34]:


#stampo i risultati ottenuti
print('Dataset undersampling:')
print('Accuracy di training {}-nn: {}'.format(k_u, acc_train_knn_u))
print('Accuracy di test {}-nn: {}'.format(k_u, acc_test_knn_u))
print('\n')
print('Dataset oversampling:')
print('Accuracy di training {}-nn: {}'.format(k_o, acc_train_knn_o))
print('Accuracy di test {}-nn: {}'.format(k_o, acc_test_knn_o))


# Anche questo algoritmo porta ad un buon classificatore nel caso del dataset più grande, mentre non è un buon classificatore nel caso del datset più piccolo.<br>

# Avendo trovato tre buoni classificatori, concludiamo il nostro studio con un confronto tra essi.

# In[35]:


table = PrettyTable()
table.field_names = ['Dataset', 'Algoritmo', 'Accuracy training', 'Accuracy test']


table.add_row(['Undersampling', 'Classification tree', acc_train_tree_u, acc_test_tree_u])
table.add_row(['Oversampling', 'Classification tree', acc_train_tree_o, acc_test_tree_o])
table.add_row(['Undersampling', 'Random forest', acc_train_rf_u, acc_test_rf_u])
table.add_row(['Oversampling', 'Random forest', acc_train_rf_o, acc_test_rf_o])
table.add_row(['Undersampling', '{}-nn'.format(k_u), acc_train_knn_u, acc_test_knn_u])
table.add_row(['Oversampling', '{}-nn'.format(k_o), acc_train_knn_o, acc_test_knn_o]) 
    
        
print(table)


# Possiamo concludere dunque affermando che il modello migliore ottenuto è quello mediante **Classification Tree** con l'utilizzo del dataset ottenuto mediante **oversampling** ma è un buon modello anche il **1-nn** ottenuto con l'utilizzo del dataset ottenuto mediante **oversampling**.
