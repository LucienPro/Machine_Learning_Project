import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    evidence = []
    labels = []

    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        
        # Ignorer la première ligne en-tête
        header = next(csvreader)  

        # Convertir les valeurs des colonnes selon les spécifications
        for row in csvreader:
            evidence_row = [
                            # Convertir en int
                            int(row[i]) if header[i] in ['Administrative','Informational', 'ProductRelated', 'OperatingSystems', 'Browser', 'Region',
                                                         'TrafficType'] else  
                            # Convertir en float
                            float(row[i]) if header[i] in ['Administrative_Duration', 'Informational_Duration',
                                                          'ProductRelated_Duration', 'BounceRates', 'ExitRates',
                                                          'PageValues', 'SpecialDay'] else
                            # Convertir en 0 ou 1
                            0 if header[i] == 'VisitorType' and row[i] == 'New_Visitor' else  
                            # Convertir en 0 ou 1
                            0 if header[i] == 'VisitorType' and row[i] == 'Other' else  
                            # Convertir en 0 ou 1
                            1 if header[i] == 'VisitorType' and row[i] == 'Returning_Visitor' else
                            0 if header[i] == 'Weekend' and row[i] == 'FALSE' else  
                            1 if header[i] == 'Weekend' and row[i] == 'TRUE' else
                            # Convertir le mois en chiffre
                            0 if header[i] == 'Month' and row[i] == 'Jan' else  
                            1 if header[i] == 'Month' and row[i] == 'Feb' else
                            2 if header[i] == 'Month' and row[i] == 'Mar' else
                            3 if header[i] == 'Month' and row[i] == 'Apr' else
                            4 if header[i] == 'Month' and row[i] == 'May' else
                            5 if header[i] == 'Month' and row[i] == 'June' else
                            6 if header[i] == 'Month' and row[i] == 'Jul' else
                            7 if header[i] == 'Month' and row[i] == 'Aug' else
                            8 if header[i] == 'Month' and row[i] == 'Sep' else
                            9 if header[i] == 'Month' and row[i] == 'Oct' else
                            10 if header[i] == 'Month' and row[i] == 'Nov' else
                            11 if header[i] == 'Month' and row[i] == 'Dec' else
                            row[i]
                            for i in range(len(header) - 1)]

            label = 1 if row[-1] == 'TRUE' else 0  # Convertir en 0 ou 1

            evidence.append(evidence_row)
            labels.append(label)

    return evidence, labels

def train_model(evidence, labels):
    #Création du classifcateur avec le premier voisin le plus proche
    model = KNeighborsClassifier(n_neighbors=1)
    
    #Entrainement du modèle
    model.fit(evidence, labels)
    
    return model

def evaluate(labels, predictions):

    # Nombre de vrai positif
    vrai_positifs = sum((true == 1) and (predicted == 1) for true, predicted in zip(labels, predictions))
    
    #Nombre devrai negatif
    vrai_negatifs = sum((true == 0) and (predicted == 0) for true, predicted in zip(labels, predictions))
    
    #Nombre de faux positifs
    faux_positifs = sum((true == 0) and (predicted == 1) for true, predicted in zip(labels, predictions))
    
    #Nombre de fauxx negatfifs
    faux_negatifs = sum((true == 1) and (predicted == 0) for true, predicted in zip(labels, predictions))
    
    # Calculer la sensibilité et la spécificité
    sensitivity = vrai_positifs / (vrai_positifs + faux_negatifs) if vrai_positifs + faux_negatifs != 0 else 0.0
    
    # Calculer la sensibilité et la spécificité
    specificity = vrai_negatifs / (vrai_negatifs + faux_positifs) if vrai_negatifs + faux_positifs != 0 else 0.0

    return sensitivity, specificity

if __name__ == "__main__":
    main()
