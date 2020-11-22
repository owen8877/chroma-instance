# ChromaGAN
if [ ! -f ../weights/chroma_gan/imagenet.h5 ]; then
  pushd ../weights/chroma_gan
  wget http://dev.ipol.im/~lraad/chromaGAN/model/my_model_colorization.h5
  mv my_model_colorization.h5 imagenet.h5
  popd
fi

# Instcolorization
if [ ! -f ../weights/instcolorization/checkpoints.zip ]; then
  echo "Downloading..."
  pushd instcolorization
  python download.py
  popd
  echo "Finish download."
  pushd ../weights/instcolorization/
  unzip checkpoints.zip
  popd
fi

# Mask RCNN
if [ ! -f ../weights/mask_rcnn/coco.h5 ]; then
  pushd ../weights/mask_rcnn
  wget https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
  mv mask_rcnn_coco.h5 coco.h5
  popd
fi