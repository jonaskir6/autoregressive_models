MADE: 
27.05: Jonas Kirch implemented datasets.py  
30.05: Niclas & Jonas Sternemann implemented networks.py together  
07.06: Niclas & Jonas Kirch added cuda support + bugfixes  
03.06: Ilias implemented sampling.py  
07.06: Fixed MADE (together)  
10.06: tested various parameter settings (together) 

PixelCNN: 
13.06: Jonas K. implemented initial form of datasets.py & networks.py (classes PixelCNN, MaskedConv2d & ResidualBlock without Color Channel dependencies) 
17.06: Group Session: started completion.py, train loop and tried adding color channel dependencies to networks.py 
17.06: Niclas fixed Residual Blocks, added nets forward function, cleanup of variables, added mnist support for datasets.py 
19.06: Ilias finished completion.py(get_image, mask_image, complete, plot), added finished sampling.py (still needs testing when the network is fixed) 
20:06: Jonas K. finished fixing networks.py (color channel dependencies, mask generation, nn Sequential for all classes, Residual Blocks and in_channels & out_channels) -> PixelCNN() call works 
20.06: Jonas S. removed padding in final layers of network to get the right output shape + added implementation for evaluation.py 
30.06: Jonas K. implemented alternative Version of PixelCNN (PixelCNNnew) 
01.07: Jonas K. fixed PixelCNN network, added Batch Norm and 3 Channel MNIST
02.07: Group session: Discussed final model and fixed some things (Jonas S and Niclas fixed and added completion for notebook; Ilias added mnist notebook; Jonas fixed general network structure) 
12.07: Group session: created final presentation
