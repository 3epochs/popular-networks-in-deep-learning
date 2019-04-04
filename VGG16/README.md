# background
AlexNet did a great job in ImageNet classification, and OverFeat got first place in localization.
VGG is some sort of combinations of these former architectures. And it got 1th place in localization and 2nd place in classification.

# changes
It was not simply copy those two former architectures, it do make some important changes.
- Smaller convolution kernel size. In author's experiments, all kernel_size were changed to `3 * 3` instead of `7 * 7` or `5 * 5` for two reasons:
    
    - improving performance
    - computing efficiently: `7 * 7`  has more conv filter parameters: 3 stacking `3*3` conv layers' parameters are 3 * (C * 3 * 3* C) = 27C^2, while `7 * 7` single conv layer's parameters: C * 7 * 7 * C = 49C^2, they got same view(insight?), but big kernel's parameters num is 81% more than small kernel.
- Deeper Network(11, 13, 16, 19), it's not convincing to say how depth matters with these 6 experiments since channels are change too.
- max-pool kernel size changed: AlexNet's pool kernel is `3 * 3`, I think the main reason author choose this size is that information loss of `2 * 2` is smaller compare to `3 * 3`
- Replace three fc layers with three convolution layers
    
