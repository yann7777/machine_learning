import cv2
import face_recognition
import joblib
import numpy as np
import time

EMBED_PATH = "embeddings.pkl"
TOLERANCE = 0.55  # seuil de distance, ajuster (0.4 strict -> 0.6 permissif)

def load_known_embeddings(path=EMBED_PATH):
    data = joblib.load(path)
    names = []
    encodings = []
    for name, enc_list in data.items():
        for enc in enc_list:
            names.append(name)
            encodings.append(enc)
    return names, encodings

def main():
    try:
        names_db, enc_db = load_known_embeddings()
    except Exception as e:
        print("Erreur lors du chargement des embeddings:", e)
        return

    cap = cv2.VideoCapture(0)  # 0 -> webcam locale
    if not cap.isOpened():
        print("Impossible d'ouvrir la caméra.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Option: redimensionner pour accélérer
        small_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Détection + encodage
        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        for (top,right,bottom,left), face_encoding in zip(face_locations, face_encodings):
            distances = face_recognition.face_distance(enc_db, face_encoding)
            if len(distances) > 0:
                min_idx = np.argmin(distances)
                min_dist = distances[min_idx]
                if min_dist <= TOLERANCE:
                    name = names_db[min_idx]
                else:
                    name = "Personne inconnue"
            else:
                name = "Base vide"
                min_dist = 1.0

            # Restaure coordonnées à la taille réelle (on avait réduit par 0.5)
            top *= 2; right *= 2; bottom *= 2; left *= 2
            cv2.rectangle(frame, (left,top), (right,bottom), (0,255,0), 2)
            cv2.putText(frame, f"{name} ({min_dist:.2f})", (left, top-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)

        cv2.imshow("Reconnaissance", frame)
        key = cv2.waitKey(1)
        if key == 27:  # ESC pour quitter
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()