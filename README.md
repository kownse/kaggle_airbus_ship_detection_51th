My first baby borned during this competition and I have to focus on him. 
So I end up with only 1 place to silver even that I started this 2 month ago.
But I am happy with it since this is life.

### unets and mrcnn
This is a image segmentation challange, again. The main trend for this kind of problem is unet and mrcnn.
I've tried both of them, especially many backend and decoder methods in unets.
It turns out unets had better results in my case although Others who are experienced in mrcnn also got high score.
I tried xception, resnet backend in keras, se_resnext and resnet backend in pytorch and fastai.
The best single model result come from the resnet in fastai procedure with 768x768 resolution which got 0.82 in private LB.

### lightgbm second layer model
This is my first time to carry out the lightgbm ensemble which was first seen in the 2018 dsb.
The main idea was to collect every statistic and neighbor information for each segment identified by the unets
and then use lightgbm to classify if they are a true prediction or not.
With this method, the score in private LB improved to 0.8467.
