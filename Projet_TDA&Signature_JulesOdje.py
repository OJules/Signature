#!/usr/bin/env python
# coding: utf-8

# # Projet TDA + Signature : Analyse des données EMG

# ## Exploration des données (Description des données) 

# ### Analysons la structure du dossier et les fichiers de chacun de ces dossiers

# In[1]:


import os
def explore_directory_structure(root_path):
    print("Structure du dossier :")
    for root, dirs, files in os.walk(root_path):
        level = root.replace(root_path, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = ' ' * 4 * (level + 1)
        for file in files:
            print(f"{sub_indent}{file}")

# Utilisation
root_path = "C:/Users/odjen/Downloads/emg+data+for+gestures/EMG_data_for_gestures-master"
explore_directory_structure(root_path)


# In[2]:


import os
import pandas as pd
import numpy as np

def process_all_data(root_path):
    """
    Traite tous les fichiers dans tous les dossiers
    """
    all_data = {}
    
    # Parcourons chaque dossier numéroté
    for folder in sorted(os.listdir(root_path)):
        if folder.isdigit() or folder.zfill(2):  
            folder_path = os.path.join(root_path, folder)
            
            # Vérifions si c'est un dossier
            if os.path.isdir(folder_path):
                folder_data = []
                print(f"Contenu du dossier {folder}:")  # Debug
                
                # Listons tous les fichiers du dossier
                files = os.listdir(folder_path)
                print(f"Fichiers trouvés: {files}")  # Debug
                
                # Lisons chaque fichier dans le dossier
                for file in sorted(files):
                    if 'raw_data' in file:  # Changé le critère de filtrage
                        file_path = os.path.join(folder_path, file)
                        try:
                            data = pd.read_csv(file_path, delimiter='\s+')  # ou un autre délimiteur
                            folder_data.append(data)
                        except Exception as e:
                            print(f"Erreur lors du chargement de {file_path}: {e}")
                
                all_data[folder] = folder_data

    return all_data

def analyze_loaded_data(data_dict):
    """
    Analyse la structure des données chargées
    """
    for folder in sorted(data_dict.keys()):
        print(f"\nDossier {folder}:")
        data_list = data_dict[folder]
        print(f"Nombre de fichiers : {len(data_list)}")
        if data_list:  # Si la liste n'est pas vide
            print("Structure du premier fichier :")
            print(f"Colonnes : {data_list[0].columns.tolist()}")
            print(f"Dimensions : {data_list[0].shape}")
            print("\nPremières lignes :")
            print(data_list[0].head())

# Utilisation
root_path = "C:/Users/odjen/Downloads/emg+data+for+gestures/EMG_data_for_gestures-master"

# Chargement et analyse des données
data_dict = process_all_data(root_path)
analyze_loaded_data(data_dict)

Nous avons 8 canaux EMG (channel1 à channel8)
Une colonne temporelle (time)
Une colonne de classe
Dimensions : 63196 lignes × 10 colonnes par fichier
2 fichiers par dossier 
# ## Découpage de la série temporelle de manière à récupérer plusieurs séries à label unique

# In[3]:


def split_by_label(data_dict):
    """
    Découpe les séries temporelles pour obtenir des séries à label unique
    """
    series_by_label = []
    
    for folder, data_list in data_dict.items():
        for data in data_list:
            # Identifions les changements de label
            label_changes = data['class'].diff().ne(0)
            # Ajoutons True au début pour marquer le début de la première série
            label_changes.iloc[0] = True
            
            # Créons un identifiant de groupe
            group_id = label_changes.cumsum()
            
            # Groupons les données par label consécutif
            for group, group_data in data.groupby(group_id):
                if len(group_data) > 0:  # Vérifier que le groupe n'est pas vide
                    series_by_label.append({
                        'data': group_data[['channel1', 'channel2', 'channel3', 'channel4', 
                                          'channel5', 'channel6', 'channel7', 'channel8']],
                        'label': group_data['class'].iloc[0],
                        'start_time': group_data['time'].iloc[0],
                        'end_time': group_data['time'].iloc[-1],
                        'folder': folder,
                        'length': len(group_data)
                    })
    
    return series_by_label

# Utilisation
split_series = split_by_label(data_dict)

# Analyse des séries obtenues
def analyze_split_series(split_series):
    """
    Analyse les séries découpées
    """
    print(f"Nombre total de séries : {len(split_series)}")
    
    # Statistiques par label
    labels = {}
    for series in split_series:
        label = series['label']
        if label not in labels:
            labels[label] = {'count': 0, 'total_length': 0}
        labels[label]['count'] += 1
        labels[label]['total_length'] += series['length']
    
    print("\nStatistiques par label :")
    for label, stats in labels.items():
        print(f"\nLabel {label}:")
        print(f"Nombre de séries : {stats['count']}")
        print(f"Longueur moyenne : {stats['total_length']/stats['count']:.2f}")

# Visualisons quelques séries
def plot_example_series(split_series, num_examples=3):
    """
    Visualise quelques exemples de séries découpées
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(num_examples, 1, figsize=(15, 5*num_examples))
    
    for i in range(min(num_examples, len(split_series))):
        series = split_series[i]
        axes[i].plot(series['data'])
        axes[i].set_title(f"Label {series['label']} (Durée: {series['end_time'] - series['start_time']})")
        axes[i].legend(series['data'].columns)
    
    plt.tight_layout()
    plt.show()

# Exécutons l'analyse
split_series = split_by_label(data_dict)
analyze_split_series(split_series)
plot_example_series(split_series)

Structure des données par label :

Total de 1813 séries identifiées
Label 0 : 940 séries (longueur moyenne ≈ 2899)
Labels 1-6 : 144 séries chacun (longueur moyenne ≈ 1700-1760)
Label 7 : 8 séries (longueur moyenne ≈ 1712)
Label nan : 1 série (longueur = 1)
# ## TDA pour créer un modèle de classification

# ### Selection et traitement de sous ensemble pour la TDA

# In[10]:


import numpy as np
from ripser import ripser
from persim import plot_diagrams
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

def extract_tda_features(diagram):
    """
    Extrait des features à partir d'un diagramme de persistance
    """
    if diagram is None:
        return [0] * 12
    
    features = []
    try:
        for dim in range(min(3, len(diagram))):
            persistence = diagram[dim][:,1] - diagram[dim][:,0]
            persistence = persistence[~np.isnan(persistence)]
            persistence = persistence[~np.isinf(persistence)]
            
            if len(persistence) > 0:
                features.extend([
                    float(np.mean(persistence)),
                    float(np.std(persistence)) if len(persistence) > 1 else 0.0,
                    float(np.max(persistence)),
                    float(len(persistence))
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
    except Exception as e:
        print(f"Erreur dans extract_tda_features: {e}")
        features = [0.0] * 12
        
    return features

def select_balanced_subset(split_series, samples_per_label=10):
    """
    Sélectionne un nombre égal d'échantillons pour chaque label
    """
    subset = []
    label_count = {}
    
    for series in split_series:
        label = series['label']
        if label not in label_count:
            label_count[label] = 0
        
        if label_count[label] < samples_per_label:
            subset.append(series)
            label_count[label] += 1
    
    print("Distribution des labels dans le sous-ensemble:")
    for label, count in label_count.items():
        print(f"Label {label}: {count} séries")
    
    return subset

def process_subset(subset):
    """
    Traite un sous-ensemble de données
    """
    X = []
    y = []
    
    for series in tqdm(subset):
        try:
            # Préparons les données
            data_array = series['data'].values
            scaled_data = StandardScaler().fit_transform(data_array)
            
            # Calculons le diagramme de persistance
            diagrams = ripser(scaled_data)['dgms']
            
            # Extraire les features
            features = extract_tda_features(diagrams)
            
            X.append(features)
            y.append(series['label'])
            
        except Exception as e:
            print(f"Erreur de traitement: {e}")
            continue
    
    return np.array(X), np.array(y)

def visualize_subset_results(X, y):
    """
    Visualise les résultats du sous-ensemble
    """
    unique_labels = np.unique(y)
    print("\nRésumé des résultats:")
    print(f"Nombre total d'échantillons traités: {len(y)}")
    print("Distribution par label:")
    for label in unique_labels:
        count = np.sum(y == label)
        print(f"Label {label}: {count} échantillons")
    
    print(f"\nDimension des features: {X.shape[1]}")

# Exécution
print("Étape 1: Sélection du sous-ensemble...")
subset = select_balanced_subset(split_series, samples_per_label=5)

print("\nÉtape 2: Traitement du sous-ensemble...")
X_subset, y_subset = process_subset(subset)

print("\nÉtape 3: Visualisation des résultats...")
visualize_subset_results(X_subset, y_subset)

# Sauvegardons les résultats
np.save('X_subset.npy', X_subset)
np.save('y_subset.npy', y_subset)


# ### Nettoyage et construction du modèle de classification avec la classe 0

# In[13]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def clean_data(X, y):
    """
    Nettoie les données plus rigoureusement
    """
    # Convertissons y en array numpy si ce n'est pas déjà fait
    y = np.array(y)
    
    # Vérifions les valeurs invalides dans X et y
    mask_x = ~np.any(np.isnan(X) | np.isinf(X), axis=1)
    mask_y = ~np.isnan(y)
    mask = mask_x & mask_y
    
    print(f"Nombre d'échantillons avant nettoyage : {len(X)}")
    print(f"Nombre d'échantillons après nettoyage : {sum(mask)}")
    
    # Convertissons y en entiers
    y_clean = y[mask].astype(int)
    
    return X[mask], y_clean

# Vérification des données
print("Données avant nettoyage:")
print("X shape:", X_subset.shape)
print("y shape:", y_subset.shape)
print("\nValeurs uniques dans y:", np.unique(y_subset))

# Nettoyage des données
X_clean, y_clean = clean_data(X_subset, y_subset)

print("\nDonnées après nettoyage:")
print("X shape:", X_clean.shape)
print("y shape:", y_clean.shape)
print("Valeurs uniques dans y:", np.unique(y_clean))

# Distribution des classes
print("\nDistribution des classes après nettoyage:")
value_counts = pd.Series(y_clean).value_counts().sort_index()
print(value_counts)

def classify_data(X, y):
    """
    Fonction de classification avec vérification supplémentaire
    """
    # Vérification finale des données
    if len(np.unique(y)) < 2:
        print("Pas assez de classes différentes pour la classification")
        return None, None, None
    
    if len(X) < 10:  # Nombre minimal d'échantillons arbitraire
        print("Pas assez d'échantillons pour la classification")
        return None, None, None
        
    # Split des données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    
    # Création et entraînement du modèle
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Prédictions
    y_pred = clf.predict(X_test)
    
    return clf, y_test, y_pred

# Lançons la classification si nous avons assez de données
if len(X_clean) > 0:
    clf, y_test, y_pred = classify_data(X_clean, y_clean)
    if clf is not None:
        print("\nRapport de classification:")
        print(classification_report(y_test, y_pred))
        
        print("\nMatrice de confusion:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
else:
    print("Pas assez de données valides pour la classification")

Nous avons 41 échantillons total avec une distribution équilibrée (5 échantillons par classe + 1 nan)
8 classes (0 à 7)

une Accuracy : 42%
avec une Meilleure performance pour la classe 0 (precision=1.00, recall=1.00)
F1-score moyen : 0.34
# ### construction du modèle de classification sans la classe 0

# In[15]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_data_without_class_0(X, y):
    """
    Prépare les données en excluant la classe 0
    """
    mask = y != 0
    X_filtered = X[mask]
    y_filtered = y[mask]
    
    print("Données après filtrage :")
    print(f"X shape: {X_filtered.shape}")
    print(f"y shape: {y_filtered.shape}")
    print("\nDistribution des classes :")
    print(pd.Series(y_filtered).value_counts().sort_index())
    
    return X_filtered, y_filtered

def tda_classification_without_0(X_clean, y_clean):
    """
    Classification TDA sans la classe 0
    """
    # Filtrons les données
    X_filtered, y_filtered = prepare_data_without_class_0(X_clean, y_clean)
    
    # Normalisons les données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filtered)
    
    # Split des données
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, 
        y_filtered, 
        test_size=0.3, 
        stratify=y_filtered,
        random_state=42
    )
    
    # Entraînons le modèle
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Prédictions
    y_pred = clf.predict(X_test)
    
    # Affichons les résultats
    print("\nRapport de classification :")
    print(classification_report(y_test, y_pred))
    
    # Matrice de confusion
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matrice de Confusion')
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    plt.show()
    
    return clf, X_test, y_test, y_pred

def visualize_class_separation(X, y, clf):
    """
    Visualise la séparation des classes avec PCA
    """
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='Set3')
    plt.title('Séparation des classes (PCA)')
    plt.xlabel('Première composante principale')
    plt.ylabel('Deuxième composante principale')
    plt.colorbar(scatter)
    plt.show()

# Exécutons la classification
clf, X_test, y_test, y_pred = tda_classification_without_0(X_clean, y_clean)

# Visualisons la séparation
visualize_class_separation(X_clean[y_clean != 0], y_clean[y_clean != 0], clf)

Nous avons 35 échantillons avec une distribution toujours équilibrée (5 échantillons par classe)
7 classes (1 à 7)


les Performances sont: 
Accuracy : 27%
F1-score moyen : 0.24

Les performances sont globalement meilleures avec la classe 0
La classe 0 semble être la plus facile à classifier
Le retrait de la classe 0 n'a pas amélioré la séparation des autres classes
# ## Création du complexe simplicial sur les séries temporelles

# In[17]:


# Importation des bibliothèques nécessaires pour les complexes simpliciaux
import numpy as np
from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt

def create_and_analyze_simplicial_complex(data_series):
    """
    Crée et analyse le complexe simplicial des séries temporelles
    """
    # Calculons les diagrammes de persistance
    diagrams = ripser(data_series, maxdim=2)['dgms']
    
    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Diagramme de persistance
    plot_diagrams(diagrams, show=False, ax=axes[0])
    axes[0].set_title('Diagramme de Persistance')
    
    # Barcode plot
    for i, diagram in enumerate(diagrams):
        for birth, death in diagram:
            if death != np.inf:
                axes[1].plot([birth, death], [i, i], 'b-')
        axes[1].set_title('Barcode Plot')
    
    plt.tight_layout()
    plt.show()
    
    return diagrams

# Analyse pour les données avec et sans classe 0
def compare_topology():
    # Avec classe 0
    print("Analyse topologique avec classe 0:")
    diagrams_with_0 = create_and_analyze_simplicial_complex(X_clean)
    
    # Sans classe 0
    mask = y_clean != 0
    print("\nAnalyse topologique sans classe 0:")
    diagrams_without_0 = create_and_analyze_simplicial_complex(X_clean[mask])
    
    return diagrams_with_0, diagrams_without_0

# Exécutons l'analyse
diagrams_with_0, diagrams_without_0 = compare_topology()

Avec la classe 0 :

Dans le Diagramme de persistance, nous avons de nombreux points H₀ (bleu) loin de la diagonale, indiquant des composantes connexes persistantes, des points H₁ (orange) indiquant quelques cycles persistants et une Plage de valeurs plus large (0-200) 


Au niveau du Barcode, nous avons des Barres H₀ plus longues, une Structure plus complexe et une Plus grande persistance des features topologiques.

Sans la classe 0 :

Dans le Diagramme de persistance, nous avons moins de points H₀, une Échelle réduite (0-60) et Moins de features persistantes


Dans le Barcode, nous avons une Structure similaire mais simplifiée avec une Persistance globalement plus faible.

Tout Cela explique pourquoi la classe 0 était plus facilement identifiable (structure topologique distincte).
Les performances de classification étaient meilleures avec la classe 0
La structure topologique est plus simple sans la classe 0, mais peut-être moins discriminante