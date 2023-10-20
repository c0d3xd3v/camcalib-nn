import gdown

url = 'https://drive.google.com/drive/folders/1ooaYwvNuFd-iEEcmOQHpLunJEmo7b4NM'
output = 'SUN360.zip'
gdown.download(url, output, quiet=False) 
