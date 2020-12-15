import matplotlib.pyplot as plt
from skimage.transform import resize

images = [
    ("imagenet_ctest", "ILSVRC2012_val_00047808"),
    ("imagenet_ctest", "ILSVRC2012_val_00036582"),
    ("imagenet_ctest", "ILSVRC2012_val_00045961"),
    ("places205", "37fab399d6e95a20c11c246549b3ddbc"),
    ("places205", "85ba693cf30381cf09f311d5ed7315c3"),
    ("coco_test_2017", "000000067603"),
    ("coco_test_2017", "000000126124"),
    ("coco_test_2017", "000000140715"),
    ("coco_test_2017", "000000181372"),
    ("coco_test_2017", "000000229116"),
]

WIDTH = 200

shapes = {}

for i, image in enumerate(images):
    surfix = 'JPEG' if image[0] == "imagenet_ctest" else 'jpg'
    filename = f"{image[0]}/{image[1]}.{surfix}"
    img = plt.imread(f'../dataset/{filename}')
    plt.imsave(f'../../figs/output/gt_{i}.jpg', img)
    shapes[image[1]] = img.shape

for network in "chroma_gan", "fusion_2obj", "fusion_2obj_huber_instance", "instcolorization":
    for i, image in enumerate(images):
        surfix = 'JPEG' if (image[0] == "imagenet_ctest" and (network is not 'instcolorization')) else 'jpg'
        surfix2 = 'psnr_reconstructed.jpg' if network == 'chroma_gan' else (
            '' if network == 'instcolorization' else '_reconstructed.jpg')
        filename = f"{network}/{image[0]}/{image[1]}.{surfix}{surfix2}"
        img = plt.imread(f'../result/{filename}')
        img = resize(img, shapes[image[1]])
        plt.imsave(f'../../figs/output/{network}_{i}.jpg', img)
