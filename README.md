# Rapport du hackathon

### Sujet : Classification de conversations en utilisant nltk, scikit-learn

#### I - Le contexte du projet

 Meetdeal est une start-up spécialisée dans la digitalisation du parcours client dans le domaine automobile. Des solutions “live tchat” omni canaux ont été développées afin de générer des leads pour ses clients. Des agents conversationnels sont chargés de répondre aux visiteurs par l’intermédiaire des bulles de tchat présents sur les sites web de nos partenaires.
Après différentes analyses, on constate que les objets des conversations engagées par les visiteurs peuvent être différents (tous ne sont pas des futurs clients) : parfois ils ont besoin d’une information technique sur leur véhicule, d’un dépannage, d’informations pour une location...
Dans le but d’optimiser la productivité de nos services, il nous a été demandé de construire un modèle de classification permettant d’identifier la demande du visiteur.
Ce hackathon a été réalisé par Nadia BEBESHINA, cheffe de projet data chez Meetdeal, et moi. Nous avons discuté quelques jours avant le début du hackathon des lignes directrices que nous allons suivre afin de mener à bien ce projet. De ce fait, nous avons réalisé le hackathon en parallèle, en échangeant régulièrement sur nos points d’avancement et en s’entraidant si besoin était. Pour le partage de code ou de fichier, nous avons utilisé nos outils quotidiens comme GitLab et S3 (AWS).

#### II - Les réalisations et les résultats
Dès le début de hackathon, nous avons bien déterminé les étapes du projet :
1) Définir quels labels nous souhaitons classer
2) Récupérer et pré-traiter les données
3) Appliquer un algorithme de clustering pour vérifier le nombre de classes
4) Entraîner le modèle de classification
5) Évaluer le modèle
6) Développer une interface streamlit pour utiliser le modèle afin de prédire

Dans un premier temps, nous avons défini, en fonction de la demande de l’équipe “Qualité”, quels allaient être les labels que nous souhaitions classer. Pour cette première expérience, nous avons décidé de conserver quatre classes : achat véhicule neuf, achat véhicule d’occasion, service après-vente, location.
Avant de procéder au paramétrage et à l’entraînement du modèle, il fallait déjà récupérer les données. Pour ce faire, on a utilisé la librairie zenpy1 pour requêter et charger les données depuis Zendesk, l’outil de gestion de la relation client utilisé par l’entreprise.
Pour la récupération, on a choisi de se concentrer sur des leads associés à toutes marques confondues, depuis le 1er janvier 2022. Il est possible de récupérer pour chaque lead différents champs : la catégorie (véhicule neuf/d’occasion), la motorisation souhaitée, le code postal, la ville, la civilité, la catégorie socio-professionnelle, l’usage prévu pour le véhicule,
Cependant, les champs qui nous intéressaient étaient liés à la conversation : le statut de la conversation (open, hold, pending), le jour, l’heure, le nombre d’échanges, l’URL (où la conversation a été enclenchée), le type d’appareil, le système d’exploitation et le navigateur web utilisé...
 
 La librairie zenpy nous permet d’itérer sur les tickets et de récupérer toutes ces informations en plus du contenu textuel (la conversation) à l’aide d’outils déjà développés par nos soins.
À chaque itération, on a écrit dans un fichier CSV chargé sur S3 les résultats obtenus. Ce processus nous a permis de décider à quel moment on souhaite arrêter pour tenter notre expérience. Nous avons conservé 6018 lignes.
Toutefois, avant de procéder à l’entraînement du modèle, nous souhaitions valider de façon non supervisée notre intuition concernant le nombre de classes identifiables, compte tenu des données à disposition. Afin de répondre à cette interrogation, à partir des champs énumérés précédemment, nous avons appliqué un algorithme de clustering à nos données. Après différentes recherches sur Internet, nous avons sélectionné l’algorithme K-Prototypes car il permet de clusteriser à partir de données à la fois catégorielles et numériques (ce qui est notre cas ici).
Avant cela, nous avons effectué un pré-traitement de nos données : nous avons regroupé le champ “type d’appareil” en deux classes : PC et Smartphone , les systèmes d’exploitation en plusieurs classes : iOS, Android, Windows, Mac OS et de même pour les navigateurs Web. En effet, les champs récupérés au préalable contenaient les numéros de version de chaque logiciel.

Une fois ce pré-traitement réalisé, il ne fallait pas oublier de retirer toutes les lignes du DataFrame possédant des valeurs nulles, auquel cas la suite n’aurait pas fonctionné.
Ensuite, l’algorithme K-Prototypes nécessite qu’on lui précise quelles sont les colonnes du DataFrame de type catégorielle, sous forme d’une liste avec les index des colonnes correspondantes. De plus, il faut ensuite convertir le DataFrame en tableau numpy, format requis par l’algorithme.
Enfin, on pouvait utiliser l’algorithme de clustering K-Prototypes, implémenté dans la librairie python kmodes2, en lui fournissant en entrée : le nombre de clusters souhaités (int), les données (numpy array), les index des colonnes catégorielles (liste).
Mais l’intérêt de notre démarche n’était pas de clusteriser nos données, mais plutôt de connaître le nombre de classes adéquat à nos données.
Pour cela, on a appliqué l’algorithme avec différents nombres de clusters (entre 1 et 9), et tracer la courbe Elbow (une méthode heuristique), qui calcule la valeur de la fonction coût pour chaque nombre de clusters.
 
 On observe sur le graphique la courbe en forme de coude. Le nombre de clusters optimal se situe au niveau de ce coude, soit trois ou quatre clusters. En effet, augmenter le nombre de clusters rendra l’ajustement meilleur car il y aura plus de paramètres à utiliser. Mais au-delà de quatre clusters dans notre cas, cet ajustement devient excessif, ce qui est reflété par la forme du coude.
Ce résultat nous a confortés dans l’idée que le nombre de quatre classes semble correct pour la classification avec les données dont on disposait.
Notre but était d’entraîner un modèle de classification de texte. Pour cela, on avait seulement besoin de deux colonnes dans nos données : la conversation et le label.

Cela nécessitait une nouvelle fois un petit pré-traitement de notre part. Les données récupérées au format csv comportent la conversation sous forme de liste de tuple, où chaque tuple possède comme éléments : l’heure, l’auteur et le message. Il fallait donc joindre tous les messages présents dans la liste en enlevant l’auteur de chaque message afin d’obtenir une seule et unique chaîne de caractères. Ensuite, pour l’entraînement du modèle, à l’aide de la librairie nltk3, on a appliqué les tâches classiques du Traitement Automatique du Langage Naturel, à savoir la tokenisation, la lemmatisation et la suppression des “stop-words”.

Enfin, nos données devaient être étiquetées pour l’entraînement, car il s’agit d’un apprentissage supervisé. Cela était possible, car chaque conversation possède des tags, soigneusement renseignés par l’agent. Une simple recherche dans la liste des tags permet d’associer chaque conversation à une des classes que l’on souhaite déterminer. Avant de lier chaque étiquette à un identifiant unique, on observait la distribution de nos données en fonction de ces mêmes étiquettes.

Dans les données que nous avons récupérées, la classe “service après-vente” était sous-représentée par rapport aux autres.
Il est bien entendu impossible de passer en entrée des données sous forme de chaîne de caractères au modèle d’apprentissage. Il faut convertir les chaînes de caractères en vecteurs de caractéristiques numériques de taille fixe. Pour notre première expérimentation, pour construire ces vecteurs, nous avons donc utilisé les fréquences des termes (tf-idf), à l’aide de la librairie scikit-learn4 et de la méthode TfidfVectorizer.

Il est possible d’utiliser plusieurs modèles d’apprentissage automatique pour classer des données textuelles avec plusieurs classes.
Nous avons donc appliqué plusieurs modèles à nos données et évalué lequel obtenait les meilleurs résultats. On a choisi des modèles implémentés dans la librairie scikit-learn : le Linear SVC (Support Vector Machines à noyau linéaire), le Multinomial NB (classifieur naïf bayésien multinomiale) et la régression logistique.

Pour chaque modèle, nous avons procédé à une validation croisée, c’est-à-dire découpé le modèle aléatoirement en cinq segments dans notre cas, où chaque segment a servi une fois de base de test. Le modèle le plus robuste obtient cinq scores très proches. Les résultats obtenus sont les suivants :
 Intuitivement, on a décidé de sélectionner le modèle Linear SVC : on a obtenu en moyenne 84% d’accuracy. En testant différentes paramétrisations du modèle (pénalisation, fonction coût, stratégie multi-classes) ou d’autres types de noyau, nous n’avons pas réussi à obtenir un meilleur résultat par rapport à la paramétrisation par défaut.
 
En évaluant plus en détail notre modèle, on constate qu’il est assez performant. En effet, en se basant sur le F1-score (bon compromis entre précision et rappel), métrique utilisée pour les problèmes de classification à plusieurs classes avec des données déséquilibrées (ce qui est notre cas, la classe “apv” est sous-représentée), on observe de bons résultats (83% et 85%) avec les métriques “macro-average” (qui accorde autant d’importance à chaque classe) et “weighted-average” (qui prend en compte la taille de la représentation de chaque classe).

 On observe également que la classe numéro deux (achat véhicule d’occasion) possède un f1-score en dessous des autres : on a 62% de rappel sur cette classe, c’est-à-dire que l’on est moins performant sur la prédiction de conversations appartenant à cette classe.
 
Le dernier objectif du hackathon était de développer une petite interface de visualisation (un peu comme une mise en production) permettant d’utiliser notre modèle. Nous l’avons enregistré et à l’aide de nos connaissances de la librairie streamlit, nous avons pu développer rapidement cette petite application.

L’utilisateur dispose d’un champ pour entrer la conversation dont il souhaite connaître la classe. Lorsqu’il appuie sur entrée, les résultats apparaissent en dessous avec la classe prédite et sa probabilité associée.
 
#### III - Le retour d’expérience
 
La problématique étudiée lors de ce hackathon étant une demande interne de l’entreprise, nous n’avons pas perdu de temps en matière d’adaptation aux données et à l’utilisation d’outils. Cependant, étant donné que nous travaillions sur des problématiques différentes en ce moment, il a fallu revoir certains aspects du code de récupération des données car les infrastructures de l’entreprise évoluent.

Le travail en présentiel aurait peut-être pu nous permettre d’avancer plus rapidement, car la communication y est toujours plus facile. Cependant, dans le cadre du hackathon, nous avons tout de même pu échanger de manière fluide sur Mattermost (outil de communication de l’entreprise) et nous appeler plusieurs fois dans la journée sur Amazon Chime.

Bien que l’on travaille beaucoup plus efficacement sur une problématique lors d’un hackathon, il nous a manqué du temps pour obtenir un résultat beaucoup plus pointu. Si c’était à refaire, nous récupérerions sûrement les données à l’avance afin de nous concentrer davantage sur la partie “Apprentissage Automatique”. En effet, avec plus de temps, on aurait pu expérimenter notre apprentissage avec seulement les répliques de l’agent conversationnel ou celles du visiteur. On aurait également pu optimiser et régulariser le modèle, mais aussi en tester d’autres (régression logistique avec plusieurs classes, réseaux de neurones). Enfin, on pourrait également améliorer l’interface streamlit, avec une meilleure ergonomie de l’application, mais aussi en affichant plus de détails sur la prédiction (les probabilités de prédiction de chaque classe ou les termes ayant eu du poids dans cette prédiction, par exemple).

L’avantage est que nous allons pouvoir, dans les prochaines semaines, retravailler sur cette problématique, en reprenant la base effectuée lors de ce hackathon. Il était ouvert à tous, notamment aux autres développeurs de l’entreprise, qui n’ont pu y participer. Je pense qu’avoir des regards différents et des compétences variées seraient bénéfiques dans l’avancement d’un projet comme celui-ci.
Nous avons pris beaucoup de plaisir à travailler durant ces deux jours et cette expérience sera très sûrement renouvelée prochainement.
