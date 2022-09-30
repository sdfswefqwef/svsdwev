import torch
from PIL import Image
from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args

# from torchvision import transforms as T
# from typing import Tuple
# from timm.data import auto_augment

# def rand_augment_transform(magnitude=5, num_layers=3):
#     # These are tuned for magnitude=5, which means that effective magnitudes are half of these values.
#     hparams = {
#         'rotate_deg': 30,
#         'shear_x_pct': 0.9,
#         'shear_y_pct': 0.2,
#         'translate_x_pct': 0.10,
#         'translate_y_pct': 0.30
#     }
#     ra_ops = auto_augment.rand_augment_ops(magnitude, hparams, transforms=_RAND_TRANSFORMS)
#     # Supply weights to disable replacement in random selection (i.e. avoid applying the same op twice)
#     choice_weights = [1. / len(ra_ops) for _ in range(len(ra_ops))]
#     return auto_augment.RandAugment(ra_ops, num_layers, choice_weights)

# def get_transform(img_size: Tuple[int], augment: bool = False, rotation: int = 0):
#     transforms = []
#     if augment:
#         transforms.append(rand_augment_transform())
#     if rotation:
#         transforms.append(lambda img: img.rotate(rotation, expand=True))
#     transforms.extend([
#         T.Resize(img_size, T.InterpolationMode.BICUBIC),
#         T.ToTensor(),
#         T.Normalize(0.5, 0.5)
#     ])
#     return T.Compose(transforms)


# # Load model and image transforms
# parseq = torch.hub.load('baudm/parseq', 'trba', pretrained=True).eval()
# from strhub.models.crnn.system import CRNN as ModelClass
# from strhub.models.parseq.system import PARSeq as ModelClass
# parseq = ModelClass.load_from_checkpoint("outputs/crnn/2022-09-28_21-25-02/checkpoints/last.ckpt").eval()

# torch.save(parseq, 'tensor.pt')
parseq = torch.load('tensor.pt', map_location=torch.device('cpu')).eval()

img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)

img = Image.open('show (2).png').convert('RGB')
img = img_transform(img).unsqueeze(0)
logits = parseq(img)
logits.shape

# # Greedy decoding
pred = logits.softmax(-1)
label, confidence = parseq.tokenizer.decode(pred)
print('Decoded label = {}'.format(label[0]))



# ./train.py trainer.max_epochs=200 trainer.gpus=1 +trainer.accelerator=gpu dataset=real +experiment=parseq
# outputs/parseq/sneedium/checkpoints/last.ckpt

# ./train.py +experiment=crnn ckpt_path=outputs/crnn/2022-09-27_05-22-47/checkpoints/last.ckpt

# ./read.py outputs/crnn/2022-09-27_05-22-47/checkpoints/last.ckpt --images demo_images/* 



# pip install transformers[torch]
