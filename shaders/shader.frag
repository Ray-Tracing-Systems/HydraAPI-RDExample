#version 450
#extension GL_ARB_separate_shader_objects : enable


layout(binding = 0) uniform UniformBufferObject {
  vec4 color;
  vec4 emission;
  vec3 bmin;
  vec3 bmax;
  uvec3 gridSize;
  mat4 modelViewProj;
} ubo;

layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec4 fragNormal_emissionMult;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outNormal;

void main() {
  outColor = texture(texSampler, fragTexCoord) * vec4(fragColor, 1);
  if (outColor.w == 0)
    discard;
  outNormal = fragNormal_emissionMult;
}