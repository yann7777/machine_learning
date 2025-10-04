import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

def main():
    # 1. Charger les données
    data = {
        "surface": [20, 30, 40, 50, 60, 70, 80, 90, 100, 45, 55, 75, 85, 95, 25, 35],
        "emplacement": [3, 5, 7, 6, 8, 9, 7, 8, 9, 4, 6, 8, 9, 10, 2, 4],
        "accessibilite": [2, 4, 6, 7, 5, 8, 9, 7, 8, 5, 7, 6, 8, 9, 3, 5],
        "prix": [120000, 180000, 280000, 320000, 380000, 520000, 580000, 600000,
                750000, 220000, 340000, 480000, 640000, 720000, 140000, 200000]
    }
    
    df = pd.DataFrame(data)
    print("Dataset chargé")
    print(df)
    
    # Préparer les données
    X = df[["surface", "emplacement", "accessibilite"]]
    y = df["prix"]
    
    # Diviser les données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardiser
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Modèle 1: Régression Linéaire
    print("\n=== RÉGRESSION LINÉAIRE ===")
    model_linear = LinearRegression()
    model_linear.fit(X_train_scaled, y_train)
    y_pred_linear = model_linear.predict(X_test_scaled)
    
    print(f"MAE: {mean_absolute_error(y_test, y_pred_linear):.0f} €")
    print(f"R²: {r2_score(y_test, y_pred_linear):.3f}")
    
    # Modèle 2: Random Forest
    print("\n=== RANDOM FOREST ===")
    model_rf = RandomForestRegressor(random_state=42)
    model_rf.fit(X_train_scaled, y_train)
    y_pred_rf = model_rf.predict(X_test_scaled)
    
    print(f"MAE: {mean_absolute_error(y_test, y_pred_rf):.0f} €")
    print(f"R²: {r2_score(y_test, y_pred_rf):.3f}")
    
    # Comparaison
    print("\n=== COMPARAISON ===")
    print(f"MAE: Linéaire = {mean_absolute_error(y_test, y_pred_linear):.0f} € vs RF = {mean_absolute_error(y_test, y_pred_rf):.0f} €")
    print(f"R²:  Linéaire = {r2_score(y_test, y_pred_linear):.3f} vs RF = {r2_score(y_test, y_pred_rf):.3f}")
    
    # Sauvegarder
    joblib.dump(model_rf, "model_rf_multivarie.joblib")
    joblib.dump(scaler, "scaler_multivarie.joblib")
    print("\nModèles sauvegardés")
    
    # Synthèse
    print("\n=== SYNTHÈSE ===")
    if r2_score(y_test, y_pred_rf) > r2_score(y_test, y_pred_linear):
        print("Modèle choisi: RANDOM FOREST (meilleur R²)")
    else:
        print("Modèle choisi: RÉGRESSION LINÉAIRE")

def predict(surface, emplacement, accessibilite):
    """Prédire le prix"""
    try:
        model = joblib.load("model_rf_multivarie.joblib")
        scaler = joblib.load("scaler_multivarie.joblib")
        
        # Utiliser DataFrame pour garder les noms des features
        features = pd.DataFrame({
            'surface': [surface],
            'emplacement': [emplacement],
            'accessibilite': [accessibilite]
        })
        
        features_scaled = scaler.transform(features)
        
        prix_pred = model.predict(features_scaled)[0]
        
        print(f"\nPrédiction pour {surface}m², emp {emplacement}/10, acc {accessibilite}/10:")
        print(f"Prix: {prix_pred:.0f} €")
        
        return prix_pred
        
    except:
        print("Erreur: exécutez main() d'abord")
        return None

if __name__ == "__main__":
    main()
    
    # Tests
    print("\n=== PRÉDICTIONS ===")
    predict(60, 7, 6)
    predict(85, 9, 8)