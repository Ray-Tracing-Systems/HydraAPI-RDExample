#version 450
#extension GL_ARB_separate_shader_objects : enable

struct DirectLight {
  vec3 position;
  float innerRadius;
  vec3 direction;
  float outerRadius;
  vec3 color;
  float padding;
};

layout(binding = 0) uniform Lights {
  DirectLight directLights[1];
} lights;

layout (binding = 1) uniform DrawConsts {
  mat4 invViewProj;
};

layout(binding = 2) uniform sampler2D diffuse;
layout(binding = 3) uniform sampler2D normals;
layout(binding = 4) uniform sampler2D depthTex;

layout(location = 0) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

float saturate(float a) {
  return clamp(a, 0, 1);
}

vec3 ComputeLighting(vec3 worldPos, vec3 normal, DirectLight light) {
  vec3 pointToLight = light.position - worldPos;
  vec3 pointToLightDir = normalize(pointToLight);
  float lightDot = dot(-pointToLightDir, normalize(light.direction));
  if (lightDot <= 0) {
    return vec3(0);
  }
  vec3 tangentVec = -pointToLight - normalize(light.direction) * lightDot * length(pointToLight);
  float radius = length(tangentVec);
  float distMult = saturate((light.outerRadius - radius) / (light.outerRadius - light.innerRadius));
  return max(dot(pointToLightDir, normal), 0.0) * min(light.color, 1.0) * distMult;
}

void main() {
  vec3 diffuse = texture(diffuse, fragTexCoord).rgb;
  vec3 normal = texture(normals, fragTexCoord).xyz * 2 - 1;
  float depth = texture(depthTex, fragTexCoord).x;
  vec4 unproj = (invViewProj * vec4(fragTexCoord * vec2(2, 2) - 1, depth, 1));
  unproj /= unproj.w;
  outColor.rgb = diffuse * ComputeLighting(unproj.xyz, normal, lights.directLights[0]);
}
