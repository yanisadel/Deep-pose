import data
import predictions


def entrainement_knn():
    data.fill_csv_signes()
    data.fill_csv_niveau()
    print("a")
    predictions.predictions2()
    print("b")
    print(list(y_test2))
    print("c")
    print(pourcentage_réussite_2,erreur2)
    predictions.predictions()
    print(list(y_test))
    print(pourcentage_réussite,erreur)
 
if __name__ == '__main__':
    entrainement_knn()

    

