MADE: 
27.05: Jonas Kirch implemented datasets.py  
30.05: Niclas & Jonas Sternemann implemented networks.py together  
07.06: Niclas & Jonas Kirch added cuda support + bugfixes  
03.06: Ilias implemented sampling.py  
07.06: Fixed MADE (together)  
10.06: tested various parameter settings (together) 

PixelCNN: 
13.06: Jonas implemented initial form of datasets.py & networks.py (classes PixelCNN, MaskedConv2d & ResidualBlock without Color Channel dependencies) 
17.06: Group Session: started completion.py, train loop and tried adding color channel dependencies to networks.py 
17.06: Niclas fixed Residual Blocks, added nets forward function, cleanup of variables, added mnist support for datasets.py 
19.06: Ilias finished completion.py(get_image, mask_image, complete, plot), added finished sampling.py (still needs testing when the network is fixed) 
20:06: Jonas finished fixing networks.py (color channel dependencies, mask generation, nn Sequential for all classes, Residual Blocks and in_channels & out_channels) -> PixelCNN() call works 
20.06: Jonas S. removed padding in final layers of network to get the right output shape + added implementation for evaluation.py 
