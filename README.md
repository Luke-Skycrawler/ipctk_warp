
Warp implementation of distance gradient and hessian in [IPC toolkit](https://ipctk.xyz/). Use analytical eigen analysis to replace ugly autogen functions. All functions are wrapped in `@wp.func` to enable parallel hessian compute on cuda threads. 

The following distance are supported: 

- [x] edge_edge_distance 
- [x] point_triangle_distance 
- [x] point_line_distance
- [x] edge_edge_cross_squarednorm


##### Acknowlegment
Large part of the code come from the paper [A Unified Analysis of Penalty-Based Collision Energies](https://github.com/Alvf/Collision-Penalty-Analysis).