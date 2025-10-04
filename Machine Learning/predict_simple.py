import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def main():
    print("="*70)
    print("TP1 - RÉGRESSION SIMPLE : PRIX EN FONCTION DE LA SURFACE")
    print("="*70)
    
    # Étape 1: Charger les données
    print("\n" + "="*50)
    print("ÉTAPE 1: CHARGEMENT DES DONNÉES")
    print("="*50)
    
    data = {
        "surface": [20,30,40,50,60,70,80,90,100],
        "prix": [150000,220000,280000,330000,400000,480000,560000,620000,700000]
    }
    df = pd.DataFrame(data)
    
    print("Dataset chargé avec succès:")
    print(df)
    print(f"\nDimensions du dataset: {df.shape}")
    print(f"Colonnes: {list(df.columns)}")
    
    # Statistiques descriptives
    print("\n--- STATISTIQUES DESCRIPTIVES ---")
    print(f"Surface - Moyenne: {df['surface'].mean():.1f}m², Min: {df['surface'].min()}m², Max: {df['surface'].max()}m²")
    print(f"Prix - Moyenne: {df['prix'].mean():,.0f}€, Min: {df['prix'].min():,}€, Max: {df['prix'].max():,}€")
    print(f"Coefficient de corrélation: {df['surface'].corr(df['prix']):.4f}")
    
    # Étape 2: Visualisation
    print("\n" + "="*50)
    print("ÉTAPE 2: VISUALISATION DES DONNÉES")
    print("="*50)
    print("Affichage du scatter plot...")
    
    plt.figure(figsize=(12, 6))
    plt.scatter(df["surface"], df["prix"], color='blue', s=80, alpha=0.7)
    
    # Ajouter les annotations des valeurs
    for i, row in df.iterrows():
        plt.annotate(f"{row['prix']:,}€", 
                    (row['surface'], row['prix']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, alpha=0.8)
    
    plt.title("Prix des logements en fonction de la surface", fontsize=14, fontweight='bold')
    plt.xlabel("Surface (m²)", fontsize=12)
    plt.ylabel("Prix (€)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Préparation des données
    X = df[["surface"]].values
    y = df["prix"].values

    # Étape 3: Division entraînement/test
    print("\n" + "="*50)
    print("ÉTAPE 3: DIVISION DES DONNÉES")
    print("="*50)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Données d'entraînement: {len(X_train)} échantillons")
    print(f"Données de test: {len(X_test)} échantillons")
    print(f"\nDétail des splits:")
    print(f"X_train (surfaces): {X_train.flatten()}")
    print(f"y_train (prix): {y_train}")
    print(f"X_test (surfaces): {X_test.flatten()}")
    print(f"y_test (prix): {y_test}")

    # Étape 4: Modèle linéaire simple
    print("\n" + "="*50)
    print("ÉTAPE 4: ENTRAÎNEMENT DU MODÈLE LINÉAIRE")
    print("="*50)
    
    model = LinearRegression().fit(X_train, y_train)
    
    print("Modèle linéaire entraîné avec succès!")
    print(f"Équation du modèle: Prix = {model.coef_[0]:.2f} × Surface + {model.intercept_:.2f}")

    # Étape 5: Évaluation sur le jeu de TEST
    print("\n" + "="*50)
    print("ÉTAPE 5: ÉVALUATION DU MODÈLE")
    print("="*50)
    
    y_pred = model.predict(X_test)

    # Métriques détaillées
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Coefficients
    pente = model.coef_[0]
    intercept = model.intercept_

    print("\n--- RÉSULTATS DÉTAILLÉS ---")
    print(f"Pente (coefficient): {pente:.2f} €/m²")
    print(f"Intercept: {intercept:.2f} €")
    print(f"Équation: Prix = {pente:.2f} × Surface + {intercept:.2f}")
    
    print("\n --- MÉTRIQUES DE PERFORMANCE ---")
    print(f"MAE (Erreur Absolue Moyenne): {mae:,.2f} €")
    print(f"MSE (Erreur Quadratique Moyenne): {mse:,.2f} €")
    print(f"RMSE (Racine de l'Erreur Quadratique Moyenne): {rmse:,.2f} €")
    print(f"R² (Coefficient de Détermination): {r2:.6f}")
    
    # Interprétation des résultats
    print("\n--- INTERPRÉTATION ---")
    print(f"• Chaque m² supplémentaire augmente le prix de {pente:.2f} €")
    print(f"• L'erreur moyenne de prédiction est de {mae:,.0f} €")
    print(f"• Le modèle explique {r2*100:.2f}% de la variance des prix")
    
    # Comparaison prédictions vs réalité
    print("\n--- COMPARAISON PRÉDICTIONS VS RÉALITÉ ---")
    for i, (surface_reelle, prix_reel, prix_pred) in enumerate(zip(X_test.flatten(), y_test, y_pred)):
        erreur = prix_reel - prix_pred
        pourcentage_erreur = (erreur / prix_reel) * 100
        print(f"Surface {surface_reelle}m²: Réel = {prix_reel:,}€, Prédit = {prix_pred:,.0f}€, "
              f"Erreur = {erreur:,.0f}€ ({pourcentage_erreur:+.1f}%)")

    # Étape 6: Sauvegarde du modèle
    print("\n" + "="*50)
    print("ÉTAPE 6: SAUVEGARDE DU MODÈLE")
    print("="*50)
    
    joblib.dump(model, "model_linear_simple.joblib")
    print("✓ Modèle linéaire sauvegardé sous 'model_linear_simple.joblib'")

    # Bonus: Régression polynomiale degré 2
    print("\n" + "="*50)
    print("BONUS: RÉGRESSION POLYNOMIALE (degré 2)")
    print("="*50)
    
    poly_model = make_pipeline(
        PolynomialFeatures(degree=2), 
        LinearRegression()
    ).fit(X_train, y_train)
    
    y_poly_pred = poly_model.predict(X_test)
    
    # Métriques polynomiales
    mae_poly = mean_absolute_error(y_test, y_poly_pred)
    mse_poly = mean_squared_error(y_test, y_poly_pred)
    rmse_poly = np.sqrt(mse_poly)
    r2_poly = r2_score(y_test, y_poly_pred)
    
    print("Modèle polynomial entraîné avec succès!")
    print(f"MAE: {mae_poly:,.2f} €")
    print(f"MSE: {mse_poly:,.2f} €")
    print(f"RMSE: {rmse_poly:,.2f} €")
    print(f"R²: {r2_poly:.6f}")
    
    joblib.dump(poly_model, "model_poly_deg2.joblib")
    print("Modèle polynomial sauvegardé sous 'model_poly_deg2.joblib'")
    
    # Comparaison des modèles
    print("\n--- COMPARAISON DES MODÈLES ---")
    print("Modèle Linéaire vs Modèle Polynomial (degré 2):")
    print(f"MAE:  {mae:,.2f} € vs {mae_poly:,.2f} € → {'Polynomial meilleur' if mae_poly < mae else 'Linéaire meilleur'}")
    print(f"R²:   {r2:.6f} vs {r2_poly:.6f} → {'Polynomial meilleur' if r2_poly > r2 else 'Linéaire meilleur'}")
    
    # Discussion sur-apprentissage
    print("\n" + "="*50)
    print("ANALYSE DU SUR-APPRENTISSAGE")
    print("="*50)
    print("• Avec seulement 9 observations, le risque de surapprentissage est élevé")
    print("• Le modèle polynomial (3 paramètres) est plus complexe que le linéaire (2 paramètres)")
    print("• Sur petit dataset, le modèle linéaire est généralement plus robuste")
    print("• La légère amélioration du polynomial peut être due au surajustement")
    print("RECOMMANDATION: Le modèle linéaire est préféré pour sa simplicité et robustesse")

def predict(surface_value):
    """Fonction pour prédire le prix à partir de la surface"""
    try:
        model = joblib.load("model_linear_simple.joblib")
        X = np.array([[surface_value]])
        prediction = float(model.predict(X)[0])
        
        # Calcul du prix au m²
        prix_au_metre_carre = prediction / surface_value if surface_value > 0 else 0
        
        print(f"\n PRÉDICTION pour {surface_value}m²:")
        print(f"   Prix estimé: {prediction:,.2f} €")
        print(f"   Prix au m²: {prix_au_metre_carre:,.2f} €/m²")
        
        return prediction
    except FileNotFoundError:
        print("Erreur: Modèle non trouvé. Veuillez d'abord exécuter main()")
        return None

if __name__ == "__main__":
    main()
    
    # Tests de prédiction détaillés
    print("\n" + "="*70)
    print("TESTS DE PRÉDICTION")
    print("="*70)
    
    surfaces_test = [25, 45, 65, 85, 105, 120]
    print("Prédictions pour différentes surfaces:")
    
    for surface in surfaces_test:
        prediction = predict(surface)
        if prediction:
            # Message d'avertissement pour l'extrapolation
            if surface < 20 or surface > 100:
                print(f"Attention: extrapolation hors plage des données d'entraînement!")
    
    print("\n" + "="*70)
    print("FIN DU PROGRAMME - TP1 TERMINÉ")
    print("="*70)