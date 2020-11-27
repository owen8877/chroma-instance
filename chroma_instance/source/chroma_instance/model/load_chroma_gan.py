from keras.engine.saving import load_model

from chroma_instance.model.fusion import fusion_network

load_weight_path = '../../../weights/chroma_gan/imagenet.h5'
img_shape = (224, 224)
batch_size = 2

fusion_generator = fusion_network(img_shape, 2)

chroma_gan = load_model(load_weight_path)

chroma_gan_layers = [layer.name for layer in chroma_gan.layers]

fusion_layer_names = [layer.name for layer in fusion_generator.layers]

for i, layer in enumerate(fusion_layer_names):
    if layer == 'model_3':
        print('model 3 skip')
        continue
    try:
        j = chroma_gan_layers.index(layer)
        print('Before:')
        print('    fusion')
        print(fusion_generator.layers[i].get_weights()[0])
        print('    chroma gan')
        print(chroma_gan.layers[j].get_weights()[0])

        fusion_generator.layers[i].set_weights(chroma_gan.layers[j].get_weights())

        print('After:')
        print('    fusion')
        print(fusion_generator.layers[i].get_weights()[0])
        print()
    except Exception as e:
        print(e)
        print(f'Layer {layer} not found in chroma gan.')
        print()