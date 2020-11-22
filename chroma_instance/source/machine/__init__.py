import os

is_colab = 'PDT' in os.uname().version

if is_colab:
    print('Running on colab!')
else:
    print("Running on Desktop!")
