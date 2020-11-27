from keras.engine.saving import load_model

from chroma_instance.model.fusion import InstanceColorModel

load_weight_path = '../../../weights/chroma_gan/imagenet.h5'
img_shape = (224, 224)
batch_size = 2

instance_generator = InstanceColorModel(img_shape)

chroma_gan = load_model(load_weight_path)

chroma_gan_layers = [layer.name for layer in chroma_gan.layers]

fusion_layer_names = [layer.name for layer in instance_generator.model.layers]

for i, layer in enumerate(fusion_layer_names):
    if layer == 'fg_model_3':
        print('model 3 skip')
        continue
    if len(layer) < 2:
        continue
    if layer[:3] == 'fg_':
        try:
            j = chroma_gan_layers.index(layer[3:])
            print('Before:')
            print('    instance')
            print(instance_generator.model.layers[i].get_weights()[0][0])
            print('    chroma gan')
            print(chroma_gan.layers[j].get_weights()[0][0])

            instance_generator.model.layers[i].set_weights(chroma_gan.layers[j].get_weights())

            print('After:')
            print('    instance')
            print(instance_generator.model.layers[i].get_weights()[0][0])
            print()
        except Exception as e:
            print(e)
            print(f'Layer {layer} not found in chroma gan.')
            print()