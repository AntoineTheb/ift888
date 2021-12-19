# -*- coding: utf-8 -*-

import numpy as np
from sklearn import linear_model


class Regression:
    def __init__(self, lamb, m=1):
        self.lamb = lamb
        self.w = None
        self.M = m

    def fonction_base_polynomiale(self, x):
        """
        Fonction de base qui projette la donnee x vers un espace polynomial tel que mentionne au chapitre 3.
        Si x est un scalaire, alors phi_x sera un vecteur à self.M dimensions : (x^1,x^2,...,x^self.M)
        Si x est un vecteur de N scalaires, alors phi_x sera un tableau 2D de taille NxM

        NOTE : En mettant phi_x = x, on a une fonction de base lineaire qui fonctionne pour une regression lineaire
        """

        if np.isscalar(x):
            return x ** np.arange(self.M+1)

        return x[:, None] ** np.arange(self.M+1)

    def recherche_hyperparametre(self, X, t):
        """
        Validation croisee de type "k-fold" pour k=10 utilisee pour trouver la meilleure valeur pour
        l'hyper-parametre self.M.

        Le resultat est mis dans la variable self.M

        X: vecteur de donnees
        t: vecteur de cibles
        """
        # AJOUTER CODE ICI
        M_min = 1
        M_max = 201
        lamb_min = 0.0001
        lamb_max = 1
        lambs = list(np.geomspace(lamb_min, lamb_max, num=20))
        k = 10


        # Liste des items
        liste_indices = np.arange(len(X), dtype=np.int)
        # Pas nécéssaire de shuffle ?
        np.random.shuffle(liste_indices)

        # Split les indices en k "chunks"
        folds = np.array_split(liste_indices, 10)

        best_mean_error = np.inf

        for M in range(M_min, M_max):
            self.M = M

            for lamb in lambs:
                self.lamb = lamb

                erreur = np.zeros(k)
                for j in range(k):
                    # Le chunk d'indices est celui de validation
                    valid_indices = folds[j]
                    # Les autres chunks serviront à l'entrainement
                    train_indices = np.concatenate([f for i, f in enumerate(folds) if i != j])

                    # Sélection des données
                    x_valid = X[liste_indices[valid_indices]]
                    t_valid = t[liste_indices[valid_indices]]
                    x_train = X[liste_indices[train_indices]]
                    t_train = t[liste_indices[train_indices]]

                    # Entrainement et calcul d'erreur
                    self.entrainement(x_train, t_train)
                    pred_valid = np.array([self.prediction(x) for x in x_valid])
                    erreur[j] = np.sum(self.erreur(t_valid, pred_valid))

                mean_error = np.mean(erreur)
                if mean_error <= best_mean_error:
                    best_mean_error = mean_error
                    best_M = M
                    best_lamb = lamb

        self.M = best_M
        self.lamb = best_lamb
        print('M trouvé: {}'.format(self.M))
        print('lamb trouvé: {}'.format(self.lamb))

    def entrainement(self, X, t, using_sklearn=False):
        """
        Entraîne la regression lineaire sur l'ensemble d'entraînement forme des
        entrees ``X`` (un tableau 2D Numpy, ou la n-ieme rangee correspond à l'entree
        x_n) et des cibles ``t`` (un tableau 1D Numpy ou le
        n-ieme element correspond à la cible t_n). L'entraînement doit
        utiliser le poids de regularisation specifie par ``self.lamb``.

        Cette methode doit assigner le champs ``self.w`` au vecteur
        (tableau Numpy 1D) de taille D+1, tel que specifie à la section 3.1.4
        du livre de Bishop.

        Lorsque using_sklearn=True, vous devez utiliser la classe "Ridge" de
        la librairie sklearn (voir http://scikit-learn.org/stable/modules/linear_model.html)

        Lorsque using_sklearn=Fasle, vous devez implementer l'equation 3.28 du
        livre de Bishop. Il est suggere que le calcul de ``self.w`` n'utilise
        pas d'inversion de matrice, mais utilise plutôt une procedure
        de resolution de systeme d'equations lineaires (voir np.linalg.solve).

        Aussi, la variable membre self.M sert à projeter les variables X vers un espace polynomiale de degre M
        (voir fonction self.fonction_base_polynomiale())

        NOTE IMPORTANTE : lorsque self.M <= 0, il faut trouver la bonne valeur de self.M

        """

        # AJOUTER CODE ICI
        if self.M <= 0:
            self.recherche_hyperparametre(X, t)

        phi_X = self.fonction_base_polynomiale(X)

        if using_sklearn:
            reg = linear_model.Ridge(alpha=self.lamb, fit_intercept=False)
            reg.fit(phi_X, t)
            self.w = reg.coef_
        else:
            mat = self.lamb*np.identity(self.M+1) + np.dot(phi_X.T, phi_X)
            vec = np.dot(phi_X.T, t)
            self.w = np.linalg.solve(mat, vec)

    def prediction(self, x):
        """
        Retourne la prediction de la regression lineaire
        pour une entree, representee par un tableau 1D Numpy ``x``.

        Cette methode suppose que la methode ``entrainement()``
        a prealablement ete appelee. Elle doit utiliser le champs ``self.w``
        afin de calculer la prediction y(x,w) (equation 3.1 et 3.3).
        """
        # AJOUTER CODE ICI
        return np.dot(self.fonction_base_polynomiale(x), self.w)

    @staticmethod
    def erreur(t, prediction):
        """
        Retourne l'erreur de la difference au carre entre
        la cible ``t`` et la prediction ``prediction``.
        """
        # AJOUTER CODE ICI
        return (t-prediction)**2
