#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform sampler2D frame;

layout(location = 0) in vec2 fragTexCoord;

layout(location = 0) out vec3 outColor;

void main() {
  vec3 meanColor = textureLod(frame, vec2(0.5, 0.5), 100).rgb;
  float meanLuminance = (0.2126 * meanColor.r + 0.7152 * meanColor.g + 0.0722 * meanColor.b);
  float T = 1.0 / meanLuminance;
  vec3 frameColor = texture(frame, fragTexCoord).rgb;
  outColor = 1.0 - exp(-T * frameColor);
}
