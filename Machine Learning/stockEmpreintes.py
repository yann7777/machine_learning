import os
import face_recognition
import joblib

def extract_and_store(known_folder="known_people", out_file="embeddings.pkl"):
    embeddings = {}
    for person_dir in os.listdir(known_folder):
        person_path = os.path.join(known_folder, person_dir)
        if not os.path.isdir(person_path):
            continue
        encs = []
        for fname in os.listdir(person_path):
            if not fname.lower().endswith((".jpg",".jpeg",".png")):
                continue
            img_path = os.path.join(person_path, fname)
            image = face_recognition.load_image_file(img_path)
            face_locations = face_recognition.face_locations(image)
            if len(face_locations) == 0:
                print(f"Aucun visage trouvé dans {img_path}, ignorer.")
                continue
            # On convertit en encodage (1er visage)
            face_enc = face_recognition.face_encodings(image, known_face_locations=face_locations)
            if len(face_enc) > 0:
                encs.append(face_enc[0])
        if encs:
            embeddings[person_dir] = encs
            print(f"{person_dir}: {len(encs)} empreintes extraites.")
    joblib.dump(embeddings, out_file)
    print("Embeddings sauvegardés dans", out_file)

if __name__ == "__main__":
    extract_and_store()