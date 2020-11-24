# ChromaGAN
#if [ ! -f ../weights/chroma_gan/imagenet.h5 ]; then
#  [ -d ../weights/chroma_gan ] || mkdir -p ../weights/chroma_gan
#  pushd ../weights/chroma_gan
#  wget http://dev.ipol.im/~lraad/chromaGAN/model/my_model_colorization.h5
#  mv my_model_colorization.h5 imagenet.h5
#  popd
#fi

# Instcolorization
#if [ ! -f ../weights/instcolorization/checkpoints.zip ]; then
#  echo "Downloading..."
#  pushd instcolorization
#  python download.py
#  popd
#  echo "Finish download."
#  [ -d ../weights/instcolorization/ ] || mkdir -p ../weights/instcolorization/
#  pushd ../weights/instcolorization/
#  unzip checkpoints.zip
#  popd
#fi

# Mask RCNN
#if [ ! -f ../weights/mask_rcnn/coco.h5 ]; then
#  [ -d ../weights/mask_rcnn ] || mkdir -p ../weights/mask_rcnn
#  pushd ../weights/mask_rcnn
#  wget https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
#  mv mask_rcnn_coco.h5 coco.h5
#  popd
#fi

# COCO Training
if [ ! -d ../dataset/train ]; then
  pushd ../dataset
  wget http://images.cocodataset.org/zips/val2014.zip
  unzip val2014.zip
  mv val2014/val2014 train
  rm -r val2014
  popd
fi