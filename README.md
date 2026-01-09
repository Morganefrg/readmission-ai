# 🏥 Prédiction du risque de réadmission hospitalière (30 jours)

## 🎯 Objectif
Ce projet illustre comment un outil de **machine learning** peut aider les établissements
de santé à **prioriser le suivi post-hospitalisation**, en estimant le risque de réadmission
d’un patient à moins de **30 jours** après sa sortie.

⚠️ Projet de démonstration (POC) à visée pédagogique.  
Il ne s’agit **pas** d’un outil de diagnostic médical.

---

## 🏥 Intérêt métier (secteur santé)
La réadmission à 30 jours est un indicateur clé :
- de qualité des soins,
- de charge pour les équipes hospitalières,
- et de coûts pour les établissements de santé.

Un outil de priorisation permet :
- d’identifier les patients nécessitant un suivi renforcé,
- d’optimiser l’allocation des ressources soignantes,
- d’appuyer la prise de décision clinique (sans la remplacer).

---

## 🧠 Approche IA
- Données anonymisées de parcours patient
- Prétraitement via pipeline scikit-learn
- Modèle **interprétable** de régression logistique
- Calcul d’un score de risque individuel
- Application web interactive avec **Streamlit**

---

## ⚙️ Choix techniques
- **Régression logistique** : modèle simple, robuste et explicable
- **Pipeline ML** : nettoyage, encodage, normalisation
- **Seuil de décision ajustable** selon la stratégie métier
- **Streamlit** : visualisation rapide et accessible

Ces choix privilégient la **lisibilité**, la **robustesse** et la **compréhension métier**.

---

## 🏗️ Architecture du projet
