import matplotlib.pyplot as plt

def poly_newton(x, y, h = 0.1):
    """
    Calcule le polynôme de Newton engendré des points passés en paramètres.

    Paramètres :
        x (list): Liste des abscisses des points.
        y (list): Liste des ordonnées des points.
        h (float) : "Pas" en x utilisé pour construire le tableau de points en sortie

    Retour :
        x_newton (list): Liste des abscisses des points du polynôme de Newton. 
        y_newton (list): Liste des ordonnées des points du polynôme de Newton. 
    """
    n = len(x)

    # Tableau des différences divisées selon : {f[x0], f[x0, x1], ... , f[x0, ..., xn]}
    coeffecients = [difference_divisee(x, y, 0, i) for i in range (n)]


    # Retourne le facteur en forme de (x-x0)(x-x1)... d'un terme d'un polynôme de Newton
    def facteur_en_x_moins_xi(n, x_cible):
        retour = 1
        for i in range (n): retour *= (x_cible - x[i])
        return retour

    nb_points = int((x[-1] - x[0])/h)
    x_newton = [x[0] + i*h for i in range (nb_points)]    
    y_newton = []

    for i in range (nb_points):
        y_courant = 0
        for j in range (len(coeffecients)):
            y_courant += coeffecients[j] * facteur_en_x_moins_xi(j, x_newton[i])
            
        y_newton.append(y_courant)

    return x_newton, y_newton
    

def difference_divisee(x, y, i, j):
    """
    Calcule la différence divisée f[x_i, ..., x_j] pour les points (x, y).
    
    Paramètres :
        x (list): Liste des abscisses des points.
        y (list): Liste des ordonnées des points.
        i (int): Indice de début pour la différence divisée.
        j (int): Indice de fin pour la différence divisée.
    
    Retour :
        float: La différence divisée f[x_i, ..., x_j].
    """
    # Cas de base : différence divisée d'ordre 0
    if i == j:
        return y[i]
    
    # Appel récursif pour calculer les différences d'ordre > 0
    return ((difference_divisee(x, y, i + 1, j) - difference_divisee(x, y, i, j - 1))/(x[j] - x[i]))

def plot_trajectory(x, y):
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, color='blue')

    # Définition des étiquettes et du titre
    plt.title('Trajectoire du volant de badminton')
    plt.xlabel('Position X (mètres)')
    plt.ylabel('Position Y (mètres)')
    plt.grid(True)
    plt.show()

    


if __name__ == "__main__":
    x, y = poly_newton([0, 3, 6], [1, 3, 0])
    plot_trajectory(x, y)