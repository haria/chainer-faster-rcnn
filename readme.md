Usage:

0. dependencies
    sudo apt-get install python-opencv
    sudo pip install numpy
    sudo pip install cython
    sudo pip install chainer

1. rebuild nms moudle if you got problems, (if you use ubuntu 14.04 x64, should work by default.)
    cd lib
    python setup.py build_ext -i
    
2. prepare an image list and 
    python demo.py --gpu 0 --list image.lst 
    # image.lst looks like
    ./test/1.jpg
    ./test/2.jpg
    ./test/3.jpg
    ./test/4.jpg
    ./test/5.jpg
    ./test/6.jpg
    ./test/7.jpg
    ./test/8.jpg
    ./test/9.jpg
3. test a single image
    python demo.py --gpu 0 --image_in ./test/1.jpg --image_out ./out.jpg
