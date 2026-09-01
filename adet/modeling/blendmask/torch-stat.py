from torchstat import stat
from torchvision.models import resnet50
model = resnet50()
stat(model, (3, 224, 224))