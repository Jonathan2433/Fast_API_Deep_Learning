from azure.storage.blob import BlockBlobService
from FastApi.config import accountName, accountKey, blobName, containerName

accountName = accountName
accountKey = accountKey
blobName = blobName
containerName = containerName


# Chemin local où vous souhaitez enregistrer le fichier téléchargé
download_file_path = 'blob.jpg'

# Créer un service blob à partir des informations d'identification
blob_service = BlockBlobService(account_name=accountName, account_key=accountKey)

# Téléchargement du blob dans un fichier local
blob_service.get_blob_to_path(containerName, blobName, download_file_path)

print(f"Le fichier a été téléchargé à {download_file_path}")