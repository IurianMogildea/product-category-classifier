# Clasificarea automată a produselor

## Descriere

Acest proiect implementează un sistem de Machine Learning pentru
clasificarea automată a produselor în categorii.

Modelul utilizează titlul produsului ca informație principală pentru
predicția categoriei.

## Dataset

Datasetul utilizat este `products.csv`.

Datasetul conține informații despre produse, comercianți, numărul de
vizualizări, ratingul comerciantului și data listării.

Variabila țintă este:

`Category Label`

Caracteristica textuală principală este:

`Product Title`

## Etapele proiectului

1. Încărcarea datelor
2. Explorarea datasetului
3. Curățarea datelor
4. Analiza distribuției categoriilor
5. Prelucrarea textului
6. Vectorizarea TF-IDF
7. Împărțirea datelor în train și test
8. Antrenarea modelelor
9. Compararea modelelor
10. Evaluarea performanței
11. Alegerea modelului final
12. Salvarea modelului
13. Predicția unor produse noi

## Modele utilizate

Au fost comparate două modele:

- Logistic Regression
- Multinomial Naive Bayes

Modelele sunt evaluate folosind:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Modelul final este ales pe baza celui mai bun F1 Score.

## TF-IDF

Pentru transformarea titlurilor produselor în caracteristici numerice
este utilizată metoda TF-IDF.

Sunt utilizate unigrame și bigrame.
