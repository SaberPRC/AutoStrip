## Automatic Brain Skull-stripping (AutoStrip)
Repo for Architecture and Implementation Details of age- and contrast-agnostic, prior knowledge-guided automatic skull-stripping (**_AutoStrip_**) 
### [<font color=#F8B48F size=3>License</font> ](./LICENSE)
```
Copyright IDEA Lab, School of Biomedical Engineering, ShanghaiTech. Shanghai, China

Licensed under the the GPL (General Public License);
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Repo for Automatic Brain Skull-stripping (AutoStripping)
Contact: JiamengLiu.PRC@gmail.com
```

This work is a refined version of previously proposed brain extraction tool <https://github.com/SaberPRC/Auto-BET>

### Main Framework
In this work, we utilize the `prior-defined age-specific` brain atlas to eliminate the intensity contrast and field of view variations, to enabling `spatially-precise` and `longitudinally-consistent` brain skull-stripping, including two steps:

<div style="text-align: center">
  <img src="Figure/framework.png" width="80%" alt="BrainParc Framework">
</div>

1. `Redundant tissue removal`: Utilizing a series of age-specific brain atlas to propagate the predefined brain mask to generate pseudo brain mask to remove redundant non-brain tissues, such as those in the facial, neck, and shoulder regions, thus enhancing the robustness of brain extraction.

2. `Learning-based brain skull-stripping`: Utilizing the attention-driven, neighboring-aware segmentation network to effectively extract brain tissues from the masked pseudo-brain by utilizing comprehensive contextual information from larger-scale patches.

