# Importation des bibliothèques nécessaires pour la construction de l'API, le traitement des images,
# l'interaction avec Azure Blob Storage, et l'utilisation de modèles de deep learning.
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import ViTForImageClassification, ViTImageProcessor, DetrForSegmentation, DetrImageProcessor, DetrForObjectDetection
from PIL import Image
from io import BytesIO
from azure.storage.blob import BlockBlobService
import torch

# Informations d'identification pour accéder au service Azure Blob Storage.
accountName = "dlflstorage"
accountKey = "kSGlxWZmB1SOlyC8jpwJ5E2GdEqt7hlB09gDSwwPXMUznJVu+EyhK5g9+Hjq3Ixp4RZruble1cPO+AStovskFQ=="
containerName = "entree"

# Définition d'un modèle Pydantic pour structurer et valider les données d'entrée de l'API.
# Ici, l'API attend un nom de blob (fichier dans Azure Blob Storage) comme entrée.
class BlobInfo(BaseModel):
    blob_name: str

class ObjectDetectionInfo(BaseModel):
    blob_name: str

# Initialisation de l'application FastAPI.
app = FastAPI()

# Configuration de CORS (Cross-Origin Resource Sharing) pour permettre à l'API de recevoir des requêtes
# de différentes origines. Ceci est particulièrement utile lors de l'intégration avec des applications frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Autorise toutes les origines
    allow_credentials=True,
    allow_methods=["*"],  # Autorise toutes les méthodes HTTP
    allow_headers=["*"],  # Autorise tous les headers HTTP
)

# Nom du conteneur Azure prédéfini
container_name = containerName

# Initialisation du service Azure Blob Storage avec les informations d'identification fournies.
blob_service = BlockBlobService(account_name=accountName, account_key=accountKey)

# Chargement du modèle de Vision Transformer (ViT) et du processeur associé pour la classification d'images.
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# Initialisation du modèle DETR.
# detr_processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')
# detr_model = DetrForSegmentation.from_pretrained('facebook/detr-resnet-50')
model_fb = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
processor_fb = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

# Définition d'un endpoint POST dans l'API qui attend un nom de blob en tant que donnée d'entrée,
# télécharge l'image correspondante depuis Azure Blob Storage, la traite et la classe.
@app.post("/classify-image/")
async def classify_image(blob_info: BlobInfo):
    try:
        # Téléchargement du blob spécifié depuis Azure Blob Storage en mémoire.
        blob = blob_service.get_blob_to_bytes(container_name, blob_info.blob_name)

        # Ouverture de l'image téléchargée pour traitement.
        image = Image.open(BytesIO(blob.content))

        # Traitement de l'image avec le processeur ViT pour la préparation de la classification.
        inputs = processor(images=image, return_tensors="pt")

        # Classification de l'image en utilisant le modèle ViT.
        outputs = model(**inputs)

        # Extraction des scores de prédiction (logits) et identification de la classe avec le score le plus élevé.
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()

        # Traduction de l'indice de la classe prédite en son libellé correspondant.
        predicted_class = model.config.id2label[predicted_class_idx]

        # Retour de la classe prédite comme réponse de l'API.
        return {"predicted_class": predicted_class}
    except Exception as e:
        # En cas d'erreur (par exemple, fichier non trouvé ou problème de traitement), renvoie une erreur HTTP 500.
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect-objects/")
async def detect_objects(info: ObjectDetectionInfo):
    try:
        blob = blob_service.get_blob_to_bytes(container_name, info.blob_name)
        image = Image.open(BytesIO(blob.content))

        # Utilisez le bon processeur pour l'image DETR.
        inputs = processor_fb(images=image, return_tensors="pt")
        outputs = model_fb(**inputs)

        # Utilisez le bon processeur pour post-traiter les sorties du modèle DETR.
        # La méthode 'post_process_object_detection' fait partie de 'DetrImageProcessor', pas de 'ViTImageProcessor'.
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor_fb.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        formatted_results = [
            {
                "label": model_fb.config.id2label[label.item()],
                "score": round(score.item(), 3),
                "box": [round(i, 2) for i in box.tolist()],
            }
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"])
            if score > 0.9  # Filtre les détections par un seuil de confiance
        ]

        return {"detections": formatted_results}
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))