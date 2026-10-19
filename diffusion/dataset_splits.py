import os
import shutil
import random

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Chemin vers le dossier actuel qui contient tous les dossiers de patients
SOURCE_DIR = "/home/fellahr/data/Original_dataset_cropped_registered_128x80x64_1x1x2"

# Chemin vers le nouveau dossier où seront créés train, val et test
OUTPUT_DIR = "/home/mardenlim/codebases/Michelangelo/diffusion/lung_data"

# Ratios de séparation (70% train, 15% validation, 15% test)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Seed pour garantir que le mélange est toujours exactement le même
SEED = 42

# Mode de transfert :
# True  = COPIE les dossiers (conserve l'original intact, nécessite du stockage)
# False = DÉPLACE les dossiers (plus rapide, ne prend pas de stockage supplémentaire)
COPY_FILES = True


# ==============================================================================
# SCRIPT
# ==============================================================================
def create_dataset_split():
    random.seed(SEED)

    # Validation des ratios
    assert abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) < 1e-5, \
        "La somme des ratios (TRAIN_RATIO + VAL_RATIO + TEST_RATIO) doit être égale à 1.0"

    if not os.path.exists(SOURCE_DIR):
        raise FileNotFoundError(f"Le dossier source '{SOURCE_DIR}' n'existe pas.")

    # Récupérer la liste des patients (exclut les fichiers cachés du système comme .DS_Store)
    patients = sorted([p for p in os.listdir(SOURCE_DIR) if not p.startswith('.')])
    total = len(patients)

    if total == 0:
        print("Aucun patient trouvé dans le dossier source.")
        return

    print(f"Total de patients détectés : {total}")

    # Mélanger la liste de manière reproductible
    random.shuffle(patients)

    # Calculer le nombre de patients par split
    n_train = int(total * TRAIN_RATIO)
    n_val = int(total * VAL_RATIO)
    # Le test prend le reste pour s'assurer de ne perdre aucun patient à cause des arrondis
    n_test = total - n_train - n_val

    splits = {
        'train': patients[:n_train],
        'val': patients[n_train:n_train + n_val],
        'test': patients[n_train + n_val:]
    }

    print(f"\n📊 Répartition :")
    print(f"  • Train : {len(splits['train'])} patients ({TRAIN_RATIO*100:.0f}%)")
    print(f"  • Val   : {len(splits['val'])} patients ({VAL_RATIO*100:.0f}%)")
    print(f"  • Test  : {len(splits['test'])} patients ({TEST_RATIO*100:.0f}%)")
    print("-" * 50)

    # Organiser les fichiers/dossiers dans train/val/test
    action_str = "Copie" if COPY_FILES else "Déplacement"
    print(f"\n{action_str} des données en cours...\n")

    for split_name, patient_list in splits.items():
        split_dir = os.path.join(OUTPUT_DIR, split_name)
        os.makedirs(split_dir, exist_ok=True)

        for p in patient_list:
            src_path = os.path.join(SOURCE_DIR, p)
            dst_path = os.path.join(split_dir, p)

            if COPY_FILES:
                if os.path.isdir(src_path):
                    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                else:
                    shutil.copy2(src_path, dst_path)
            else:
                shutil.move(src_path, dst_path)

    print(f"✅ Opération terminée ! Les dossiers sont dans : '{OUTPUT_DIR}'")


if __name__ == "__main__":
    create_dataset_split()