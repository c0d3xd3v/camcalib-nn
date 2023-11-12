import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile

from CNN.DeepCalibOutputLayer import FocAndDisOut
from CNN.LoadCNN import loadMobileNetRegression

# does not work, due to the lack of implemented functionality in pytorch
#model = torchvision.models.inception_v3(weights=None)
model = torchvision.models.mobilenet_v3_large(weights=None)
model.classifier = FocAndDisOut(960)
#model = loadMobileNetRegression()
model.eval()

script_model = torch.jit.script(model)
script_model_vulkan = optimize_for_mobile(script_model, backend='vulkan')

dummy_input = torch.ones(8, 3, 299, 299, dtype=torch.float32)
dummy_input = dummy_input.to('vulkan')

with torch.no_grad():
    for k in range(100):
        output = script_model_vulkan.forward(dummy_input)
        print(output)

print(script_model_vulkan)

#import urllib
#from PIL import Image
#from torchvision import transforms

#url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
#try: urllib.URLopener().retrieve(url, filename)
#except: urllib.request.urlretrieve(url, filename)


#input_image = Image.open(filename)
#preprocess = transforms.Compose([
#    transforms.Resize(256),
#    transforms.CenterCrop(224),
#    transforms.ToTensor(),
#    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#])
#input_tensor = preprocess(input_image)
#input_batch = input_tensor.unsqueeze(0)

#if torch.is_vulkan_available():
#    input_batch = input_batch.to('vulkan')
#    model.to(torch.float32)
#    model.to('vulkan')

#with torch.no_grad():
#    output = model(input_batch)
# Tensor of shape 1000, with confidence scores over ImageNet's 1000 classes
#print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
#probabilities = torch.nn.functional.softmax(output[0], dim=0)
#print(probabilities)
