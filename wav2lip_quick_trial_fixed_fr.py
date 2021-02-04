# -*- coding: utf-8 -*-

""" On commence par monter le dossier Drive pour pouvoir y accéder depuis l'environnement virtuel.
 Colab va vous demander ici de cliquer sur un lien pour autoriser l'accès. Cela vous fournira un code qu'il faudra coller dans l'invite qui s'ouvrira.
"""

from google.colab import drive
drive.mount('/content/gdrive')

""" On récupère le dépôt github dans notre Drive """

!git clone https://github.com/Rudrabha/Wav2Lip.git "/content/gdrive/My Drive/Wav2Lip"

""" Maintenant, il faut télécharger le fichier wav2lip.gan.pth et le placer dans le dossier Wav2Lip/checkpoints/ dans Drive.

Lien: https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA?e=n9ljGW

"""

""" On installe les dépendances """
!pip uninstall tensorflow tensorflow-gpu

!cd "/content/gdrive/My Drive/Wav2Lip/" && pip install -r requirements.txt

""" On récupère le modèle pour la détection des visages """
!wget "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth" -O "/content/gdrive/My Drive/Wav2Lip/face_detection/detection/sfd/s3fd.pth"

""" On peut désormais lancer le script !

Ajoutez des fichiers audio et vidéo dans votre dossier Wav2Lip sur Drive et adaptez la commande ci-dessous.
"""

!cd "/content/gdrive/My Drive/Wav2Lip" && python inference.py --checkpoint_path checkpoints/wav2lip_gan.pth --face "input_vid.mp4" --audio "input_audio.wav"

""" Pour vérifier qu'un résultat a été généré """
!ls "/content/gdrive/My Drive/Wav2Lip/results/"

""" Les résultats sont maintenant dans le dossier Wav2Lip/results/ """

"""## ** Variations à essayer **

1.   Utiliser plus de padding pour la région du menton
"""

!cd Wav2Lip && python inference.py --checkpoint_path checkpoints/wav2lip_gan.pth --face "../sample_data/input_vid.mp4" --audio "../sample_data/input_audio.wav" --pads 0 20 0 0

"""2.   Utiliser resize_factor pour réduire la résolution de la vidéo, car il y a une chance d'obtenir de meilleurs résolutats sur des vidéos basses résolutions. Pourquoi ? Parce que le modèle a été entraîné sur des visages basse résolution. (Traduction du notebook original. Honnêtement, j'ai eu peu de résultats avec cette option, il vaut mieux réduire ses vidéos à la main avec ffmpeg.) """

!cd Wav2Lip && python inference.py --checkpoint_path checkpoints/wav2lip_gan.pth --face "../sample_data/input_vid.mp4" --audio "../sample_data/input_audio.wav" --resize_factor 2
