# Demo

<https://www.bilibili.com/video/av59793400/?p=13>

[blog](https://www.ybliu.com/2020/07/3D-semantic-mapping-RGBD.html)

# Build


1.  Build ORB SLAM2
2.  Use `catkin_make` build ros project

# Run bag file

-   Download *demo.bag* and pretrained model from the author.
-   Modify parameters: `semantic_slam/params/semantic_cloud.yaml`

```sh
roslaunch floatlazer_semantic_slam semantic_mapping.launch

rosbag play -l  --clock demo.bag
```


# Change Notes

1.  Rename `ORB_SLAM2` to `FLOATLAZER_ORB_SLAM2`
2.  Rename ros package name from `semantic_slam` to
    `floatlazer_semantic_slam`

