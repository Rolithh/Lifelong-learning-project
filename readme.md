# PROJET 11 - Lifelong learning for never ended robot learning

Laetitia Haye, Victoire Miossec, Alexandre Presa



## Rappels
Dans le cadre de la recherche en apprentissage automatique et en robotique, le problème de préhension est fondamental pour permettre aux robots une certaine autonomie dans leur environnement face à des objets, à l'instar des êtres humains. Dans ce contexte, les méthodes de *lifelong learning* (apprentissage continu) sont toutes indiquées : adaptables, continues (apprentissage permanent, même en action).
Cependant, le problème rencontré aujourd'hui repose sur l'idée de l'apprentissage d'une nouvelle prise sans oublier les précédentes : mon robot doit apprendre à saisir une pomme alors qu'il sait déjà saisir un livre mais sans oublier la façon dont il saisit le livre. Ce phénomène est décrit sous le nom de *catastrophic forgetting*. Il s'agit donc d'imiter la plasticité cérébrale pour créer, supprimer et mutualiser les synapses dont le robot aurait besoin.
Trois grandes approches de l'apprentissage continu ont été développées au fil du temps :
- d'abord, l'approche *memory replay* : un buffer où l'on stocke les différentes tâches apprises. Efficace mais la quantité de mémoire nécessaire est très vite colossale, et les algorithmes explosent en complexité spatiale.
- Ensuite, l'approche *architecture growing* : où l'on essaye d'imiter le fonctionnement du cerveu en faisant évoluer l'architecture du réseau de neurones, le plus souvent en ajoutant des paramètres. De la même façon, ceux-ci multiplient la mémoire nécessaire au processus et les algorithmes explosent en complexité spatiale.
- Enfin, l'approche par régularisation : le fait d'appliquer une régularisation à la méthode d’optimisation de sorte à
empêcher l’oubli des concepts précédemment appris. Même si l'utilisation de la mémoire est largement moindre, alheureusement le *catastrophic forgetting* demeure.
Ainsi donc, aucune approche parfaite n'existant et les différents axes se complétant, il pourrait être intéressant de développer et coder des méthodes hybrides. Nous nous sommes donc efforcés de trouver, parmi les papiers de recherche existants, trois méthodes qui nous paraissaient intéressantes et de les implémenter.
En voici la présentation et les codes.



## Algorithmes codés

### FICHIERS
De manière générale, les fichiers sont découpés comme suit :

##### model.py
définit les fonctions pour charger le CNN AlexNet pré-entrainé sur ImageNet, augmenter l’architecture du réseau (ajoute 5 neurones sur la dernière couche), définir la loss, et pour créer l’architecture finale (après les ajouts progressifs de neurones durant l’entrainement) pour pouvoir recharger le modèle final après l’avoir enregistré.

##### train.py 
entraînement du modèle (adapté pour un problème de régression avec 5 variables).

##### plot.py
plot la loss pendant l’entraînement en fonction des époques.

##### preprocess.py 
pré-traitement sur les images.

##### data_loader.py et data_loader2.py
charge les données à partir de la bdd qu’on a créé avec les 5 domaines définis (extraite de Jacquard) pour les éléments de train ou de test.

### APPRENTISSAGE AUTOMATIQUE
Nous avons codé l'apprentissage classique afin d'avoir un élément de comparaison pour nos trois méthodes en terme de temps d'évaluation, d'exactitude des résultats et de l'évolution de ceux-ci avec l'entraînement du modèle. Vous retrouverez donc dans l'archive 6 fichiers comme décrits ci-dessus.


### LWF

#### Description :
Learning without Forgetting est une méthode qui repose sur l’architecture growing. Lorsqu’on souhaite apprendre un nouveau domaine, on ajoute à la couche de sortie des neurones avec des poids qui seront spécifiques à ce domaine. Le modèle d’origine a des paramètres partagés θs (ceux du réseau de neurones convolutionnels) et des paramètres propres aux domaines précédents θo. On en ajoute des nouveaux θn pour le nouveau domaine.
Le principe est d’entraîner conjointement tous les poids θs, θo, et θn pour qu’ils fonctionnent bien sur les anciens comme le nouveau domaine.

La spécificité de cette méthode est d’ajouter à la loss un terme (Lold) qui correspond à l’écart entre les prédictions du nouveau modèle, et les réponses que donne le réseau d’origine pour les sorties des anciens domaines à chaque image du nouveau domaine. Autrement dit, on fait passer les nouvelles images dans l’ancien modèle et on enregistre les valeurs en sortie sur les anciens domaines (Yo), et ces valeurs deviennent des valeurs cibles sur les anciens domaines pour le nouveau modèle.
Cette méthode permet ainsi de ne pas faire varier de manière trop importante les anciens poids pour préserver les performances sur les anciens domaines tout en ayant la liberté de les ajuster un peu.  Son intérêt (et c’est ce qui la distingue du « joint training ») réside dans le fait qu’elle n’a pas recours à des images de ces anciens domaines mais se sert uniquement de celles du nouveau domaine.

**A corriger : la loss ne converge pas pendant l’entraînement, il y a un bug quelque part.**

### Eléments à modifier pour de meilleurs résultats :
Hyperparamètres qui peuvent être optimisés : momentum, weight_decay, lr (learning rate), n_epochs (nombre d’époques pour l’entraînement), batch_size (deuxième argument lorsqu’on instancie la classe « ContinualJacquardLoader »). On peut également ajouter une constante lambda pour donner différents poids aux termes de la loss (modifie l’importance accordée aux anciens domaines par rapport au nouveau).
Le choix de la méthode de descente du gradient, actuellement SGD, peut aussi être revue.
Enfin, une étape de « warm-up » (cf papier) peut être implémentée pour accélérer l’entraînement.


### DGR
#### Description :
utiliser un GAN (type WGAN) pour générer les données des tâches précédentes.

#### Apprentissage :
- apprentissage du GAN : on génère les données des tâches précédentes grâce au GAN déjà entraîné, on reentraine ce GAN avec ces données et les nouvelles données.
- apprentissage du predictor : on génère les données des tâches précédentes grâce au GAN et on genre les ground truth des tâches précédentes avec le predictor actuel. On reentraine le predictor avec ces données et les nouvelles

#### Paramètres :
ils sont tous décrits dans le fichier main


### CLEAR
#### Description
un autoencodeur permet de trier les données selon si elles sont éloignés de celles déjà vues (tri basé sur un seuillage,  l'erreur de reconstruction, le seuil est régulièrement mis à jour durant l'apprentissage). Les modèles sont alors entraînés sur les données "nouvelles" en utilisant une pénalisation EWC calculée sur toutes les données

##### Apprentissage
Pour l'autoencodeur et le predictor on trie les données selon si elles sont nouvelles ou non. On reentraine l'autoencodeur sur les nouvelles données de autoencodeur puis on met à jour la pénalisation EWC. Même principe pour le predictor. Enfin on recalcule Les valeurs des seuils.

##### Paramètres
- taille max des buffers pour l'autoencodeur et le predictor
- paramètre alpha de CLear (utilisé pour les seuils)
- paramètre gamma de EWC
- nombre d'épochs pour chaque apprentissage
- learning rate, batch size, weight decay


## Résultats
Comme nous l'avons précisé à l'oral, la méthode CLeaR ressort largement gagnante de ces essais : un temps d'apprentissage court et une précision moyenne de plus de 50%, très rapidement atteint, malgré un échec (près de 6% de succès) au cours de la première tâche. La méthode DGR pour le moment présente de mauvais résultats, mais ils sont biaisés : nous avons dû réduire les performances de l'algorithme (à travers son gradient) pour obtenir des résultats exploitables en un temps raisonnables (en moins d'une semaine). Enfin, la méthode LwF ne donne pas de résultats exploitables pour le moment, la loss ne convergeant pas. Nous avons donné des pistes pour débugger le code dans la section LwF.
A noter que nous n'avons pu évaluer les méthodes sur la base de donnée Jacquard, le serveur ne renvoyant plus de réponse depuis une dizaine de jours.
