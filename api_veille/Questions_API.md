# Questions autour de la notion d'API



- ## Qu'est-ce que HTTP ?

HTTP (Hypertext Transfer Protocol) est un protocole qui permet de récupérer des ressources telles que des documents HTML. Il est à la base de tout échange de données sur le Web. C'est un protocole de type client-serveur, ce qui signifie que les requêtes sont initiées par le destinataire (qui est généralement un navigateur web). HTTP détermine comment la page est transmise du serveur au client.

Lorsqu'on saisit une adresse Internet dans votre navigateur Web et qu’un site est affiché quelques secondes plus tard, cela signifie qu’une communication a été établie entre le navigateur et le serveur Web via HTTP. On peut donc dire que le HTTP est la langue dans laquelle le navigateur Web parle au serveur Web afin de lui communiquer ce qui est demandé.

Un document complet est construit à partir de différents sous-documents qui sont récupérés, par exemple du texte, des descriptions de mise en page, des images, des vidéos, des scripts et bien plus.

HTTP n'est pas seulement utilisé pour récupérer des documents, mais aussi  pour des images, des vidéos ou bien pour renvoyer des contenus vers des  serveurs, comme des résultats de formulaires HTML.

![](/home/apprenant/Pictures/diagram-of-http-communication-process-fr.png)

​													Représentation schématique du processus de communication conformément au protocole HTTP



- ## Qu'est ce qu'une API ?

Une API (Application Programming Interface) est un ensemble de définitions et de protocoles qui facilite la création et l'intégration de logiciels d'applications. Elle permet de rendre disponibles les données ou les fonctionnalités d’une application existante afin que d’autres applications les utilisent. Utiliser une API permet donc d’utiliser un programme existant plutôt que de le re-développer.

De ce fait, une API expose, rend disponible des fonctionnalités ou des données. Pour les utiliser, la plupart des API requièrent une clé. Celle-ci permet à l’API de nous identifier comme étant un utilisateur ayant les droits nécessaires pour se servir de l’API.

Une des principales fonctionnalités qu’on retrouve quand on utilise une API est l’exposition de données. Cela signifie que les contenus d’un site (pages, articles) sont accessibles grâce à des endpoints, dans un format de données structurées. Concrètement, en se rendant à une certaine URL, les données d’un site seront affichés le plus souvent au format JSON.

On peut donc imaginer une API comme un site exposant des informations, lesquelles seront récupérées par un ou plusieurs sites. En plus d’exposer des données, une API peut également exposer des services, des fonctionnalités. 

Par exemple, si on veut faire un envoi massif d'e-mails, pas besoin de louer ou de configurer un serveur. On peut utiliser une API dédiée pour ça. Aussi, lorsqu'on retrouve un champ adresse dans un formulaire et qu'on nous suggère une adresse préformatée après avoir tapé les premiers caractères, ceci est aussi dû à la présence d'un API (Google Places notamment).



- ## Quelle est la spécificité des API REST ?

REST (pour *REpresentational State Transfer*) est une type d’architecture d’API (et non un protocole) qui fournit un certain nombre de normes et de conventions à respecter pour faciliter la communication entre applications.

Dans le cas d'une API REST (REpresentational State Transfer), on retrouve une architecture qui doit respecter quelques règles. On en définit 6 en général. Elles sont les suivantes :

1. Une architecture client-serveur composée de clients, de serveurs et de ressources. Dans un API, client et serveur sont séparés, ce qui signifie que les codes peuvent chacun être modifiés sans affecter l’autre, tant que tous deux continuent de communiquer dans le même format.
2. Des communications client-serveur sans état, ce qui signifie que le contenu du client n'est jamais stocké sur le serveur entre les requêtes ; les informations sur l'état de la session sont stockées sur le client.
3. Des données qui peuvent être mises en mémoire cache pour éviter certaines interactions entre le client et le serveur.
4. Une interface uniforme entre les composants qui permet un transfert standardisé des informations au lieu d'un échange personnalisé en fonction des besoins d'une application. Roy Fielding, le créateur de REST, décrit ceci comme « la fonction centrale qui distingue le style architectural REST des autres styles basés sur le réseau ».
5. Un système à couches où des couches hiérarchiques peuvent assurer la  médiation dans les interactions entre le client et le serveur.
6. Du code à la demande qui permet au serveur d'étendre la fonctionnalité  d'un client en transférant le code exécutable (recommandation  facultative, car elle réduit la visibilité).



- ## Qu'est ce qu'un URI, un endpoint, une opération ?

**URI** : Un URI (Uniform Resource Identifier) est une courte chaîne de caractères identifiant une ressource, qu'elle soit abstraite ou physique, sur un réseau et dont la syntaxe respecte une norme. Les URI peuvent par exemple identifier des sites Web, tout comme ils peuvent identifier des expéditeurs ou des destinataires de courriels. Les URI peuvent contenir jusqu'à 5 parties : le schéma (indique le protocole utilisé), l'autorité (identifie le domaine), le chemin (indique le chemin d’accès à la ressource), la requête (représente une action de requête) et le fragment (désigne un aspect partiel d’une ressource). Seuls le schéma et le chemin apparaissent nécessairement dans l'URI. Voici le format d'une URI :

```http
scheme :// authority path ? query # fragment
```

**Endpoint** : Un endpoint est ce qu’on appelle une extrémité d’un canal de communication. Autrement dit, lorsqu’une API interagit avec un autre système, les points de contact de cette communication sont considérés comme des endpoints. Un endpoint peut inclure une URL d’un serveur ou d’un service. Chaque endpoint est l’emplacement à partir duquel les API peuvent accéder aux ressources dont elles ont besoin pour exécuter leur fonction. Un endpoint représente l’endroit où les API envoient les demandes et où réside la ressource.

**Opération** : Une opération est une unité de l'API REST que l'on peut appeler. Une opération comprend un verbe HTTP et un chemin URL qui dépend du contexte de l'API. On peut simplement définir l'opération comme une action réalisée à travers la requête que l'on souhaite faire passer à l'aide de notre API. Dans le contexte de l'API REST, on utilise les verbes HTTP existants plutôt que d’inclure l’opération dans l’URI de la ressource. Ainsi, généralement pour une ressource, il y a 4 opérations possibles (CRUD pour Create, Read, Update, Delete). HTTP propose les verbes correspondants : Create => POST, Read => GET, Update => PUT, Delete => DELETE.



- ## Que trouve-t-on dans la documentation d'une API REST ?


Dans la documentation d'une API REST, on retrouve l'ensemble des opérations que l'on peut réaliser (que l'on peut appeler méthodes ici), les descriptions des ressources présentes dans l'API, les endpoints, les paramètres, les exemples de requêtes et de réponses. On peut aussi y retrouver des sections plus conceptuelles pour l'API telles que des tutoriels pour bien débuter, les codes d'erreurs, l'authentification ou l'autorisation à faire des requêtes.



- ## Utilisez Postman pour faire 3 requêtes sur l'API publique de votre choix. Partagez les requêtes ainsi que les réponses

Dans le cadre de cet exercice, j'ai décidé de travailler sur Numbers API, qui répertorie des faits sur les nombres en général. L'adresse de l'API est la suivante : http://numbersapi.com/#42

Voici mes 3 requêtes :

- une requête qui permet d'afficher un fait sur un nombre aléatoire compris entre 10 et 20 : http://numbersapi.com/random?min=10&max=20

- une requête qui permet d'afficher un fait aléatoire sur une date donnée : http://numbersapi.com/2/29/date

- une requête qui permet d'afficher un fait aléatoire sur une année donnée : http://numbersapi.com/random/year



- ## Réalisez une API REST à l'aide de FastAPI qui, pour une certaine phrase, renvoie la prédiction du sentiment





- ## Sources

https://developer.mozilla.org/fr/docs/Web/HTTP/Overview

https://www.ionos.fr/digitalguide/hebergement/aspects-techniques/definition-protocole-http/

https://www.agencedebord.com/api-definition-utilisation/

https://www.redhat.com/fr/topics/integration/whats-the-difference-between-soap-rest

https://practicalprogramming.fr/api-rest

https://www.redhat.com/fr/topics/api/what-are-application-programming-interfaces

https://www.youtube.com/watch?v=lsMQRaeKNDk

https://www.ionos.fr/digitalguide/sites-internet/developpement-web/le-uniform-resource-identifier/
