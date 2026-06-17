import json
import os

def load_config():
    # Construit le chemin absolu vers config.json situé à la racine de 'app'
    # __file__ correspond à Utilities/ConfigLoader.py
    # os.path.dirname(__file__) correspond au dossier Utilities
    # le 2ème dirname remonte au dossier app
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'config.json')
    
    try:
        with open(config_path, 'r') as file:
            config = json.load(file)
            return config
    except FileNotFoundError:
        print(f"Erreur : Le fichier {config_path} est introuvable.")
        return {}
    except json.JSONDecodeError:
        print(f"Erreur : Le fichier {config_path} est mal formaté.")
        return {}