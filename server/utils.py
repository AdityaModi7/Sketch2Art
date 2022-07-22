import kornia
import torch
import pickle
import base64

from PIL import Image
from io import BytesIO
from models.resnet import ResnetModel
from models.edge import create_model


class Generator:
    """
    Wrapper class for the StyleGAN3 Generator
    """

    def __init__(self, model_pkl, edge_pkl, landmark_pkl, anchor=True, multi_trunc=False):
        # Load in files
        with open(model_pkl, "rb") as f:
            self.G = pickle.load(f)['G_ema'].cuda()
        self.sketcher = create_model(edge_pkl)
        self.sketcher.eval()
        self.point_detector = ResnetModel(3, 56).cuda()
        # self.point_detector.load_state_dict(torch.load(landmark_pkl))
        # If anchor is true, then anchor the position to the avg
        if anchor and hasattr(self.G.synthesis, 'input'):
            avg_w = self.G.mapping.w_avg.unsqueeze(0)
            shift = self.G.synthesis.input.affine(avg_w)
            self.G.synthesis.input.affine.bias.data.add_(shift.squeeze(0))
            self.G.synthesis.input.affine.weight.data.zero_()
        # If mutlitrunc is enabled, then

    def sample_dlatent(self, psi=1, cutoff=10):
        z = torch.randn([1, self.G.z_dim]).cuda()
        w = self.G.mapping(z, None,
                           truncation_psi=psi,
                           truncation_cutoff=cutoff)
        return w

    def render_latent(self, w):
        img = self.G.synthesis(w, noise_mode="const")
        img = img.clamp(-1, 1)
        return img

    def extract_sketch(self, im_tensor):
        sketch = self.sketcher(im_tensor)
        return sketch

    def generate(self):
        w = self.sample_dlatent()
        img = self.render_latent(w) * 0.5 + 0.5
        sketch = self.extract_sketch(img)
        landmarks = None
        return ImageData(w, img, sketch, landmarks)


class ImageData:

    def __init__(self, w, image, sketch, landmarks):
        self.w = w
        self.image = image
        self.sketch = sketch
        self.landmarks = landmarks

    def get_image_pillow(self):
        img = (self.image.permute(0, 2, 3, 1) * 255).to(torch.uint8)
        img = img[0].detach().cpu().numpy()
        return Image.fromarray(img, 'RGB')

    def get_image_base64(self):
        pillow_image = self.get_image_pillow()
        buffered = BytesIO()
        pillow_image.save(buffered, format="JPEG")
        imgstr = base64.b64encode(buffered.getvalue())
        return imgstr

    def get_sketch_pillow(self):
        sketch = (self.sketch.permute(0, 2, 3, 1) * 255).to(torch.uint8)
        sketch = sketch[0, :, :, 0].detach().cpu().numpy()
        return Image.fromarray(sketch, 'L')

    def get_sketch_base64(self):
        pass

    def convert_img_base64(pillow_image):
        buffered = BytesIO()
        pillow_image.save(buffered, format="JPEG")
        imgstr = base64.b64encode(buffered.getvalue())
        return imgstr

    def json(self,):
        return {
            "real_image": self.get_image_base64(),
            "sketch": self.get_sketch_base64(),
            "landmarks": self.landmarks.detach().cpu().numpy().tolist()
        }


if __name__ == "__main__":
    g = Generator("ckpts\\close_ups\\sg3.pkl",
                  "ckpts\\close_ups\\netG.pth", "ckpts\\landmarks.pt")
    im = g.generate()
    im.get_image_pillow().save("1.png")
    im.get_sketch_pillow().save("2.png")
