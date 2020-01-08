#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 1) uniform sampler2D diffuse;

layout(location = 0) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

void main() {
  outColor = texture(diffuse, fragTexCoord);// *vec4(fragColor, 1);
  //outColor = vec4(fragTexCoord, 0, 1);
}
