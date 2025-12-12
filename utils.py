# small helpers used by the project; currently minimal
import os

def list_pdfs(folder: str):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.pdf')]
