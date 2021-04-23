#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec4 fragNormal_emissionMult;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outNormal;

void main() {
  outColor = pow(texture(texSampler, fragTexCoord), vec4(2.2)) * vec4(fragColor, 1);
  if (outColor.w == 0)
    discard;
  outNormal = vec4(normalize(fragNormal_emissionMult.xyz), fragNormal_emissionMult.w);
}