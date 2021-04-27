glslc.exe shader.vert -O0 -o vert.spv -g
glslc.exe shader.frag -O0 -o frag.spv -g
glslc.exe full_screen.vert -O0 -o full_screen.spv -g
glslc.exe resolve.frag -O0 -o resolve.spv -g
glslc.exe postprocess.frag -O0 -o postprocess.spv -g
glslc.exe debug_points_vert.vert -O0 -o debug_points_vert.spv -g
glslc.exe debug_points_frag.frag -O0 -o debug_points_frag.spv -g
glslc.exe initialLighting.comp -O0 -o initialLighting.spv -g
