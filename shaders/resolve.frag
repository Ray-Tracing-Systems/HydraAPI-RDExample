#version 450
#extension GL_ARB_separate_shader_objects : enable

#define float3 vec3
#define uint32_t uint

#include "../lights.h"

layout(push_constant) uniform lighttmPC
{
  uint directLightsCount;
  uint spotLightsCount;
};

layout(binding = 0) uniform Lights {
  DirectLight directLights[MAX_DIRECT_LIGHTS];
  SpotLight spotLights[MAX_SPOT_LIGHTS];
  mat4 TM[MAX_LIGHTS];
} lights;

layout (binding = 1) uniform DrawConsts {
  mat4 invViewProj;
};

layout(binding = 2) uniform sampler2D diffuse;
layout(binding = 3) uniform sampler2D normals;
layout(binding = 4) uniform sampler2D depthTex;
layout(binding = 5) uniform sampler2DArray shadowMap;

layout(location = 0) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

float saturate(float a) {
  return clamp(a, 0, 1);
}

vec3 ComputeLighting(vec3 worldPos, vec3 normal, DirectLight light) {
  if (length(light.direction) < 1e-5) {
    return vec3(0);
  }
  vec3 pointToLight = light.position - worldPos;
  vec3 pointToLightDir = normalize(pointToLight);
  float lightDot = dot(-pointToLightDir, normalize(light.direction));
  if (lightDot <= 0) {
    return vec3(0);
  }
  vec4 lightPos = lights.TM[light.id] * vec4(worldPos, 1.0);
  lightPos /= lightPos.w;
  vec2 lightUv = lightPos.xy * 0.5 + 0.5;
  float shadow = 0.0;
  ivec2 shadowSize = textureSize(shadowMap, 0).xy;
  for (float x = -1.5; x <= 1.5; x += 1.0) {
    for (float y = -1.5; y <= 1.5; y += 1.0) {
      float shadowDepth = texture(shadowMap, vec3(lightUv + vec2(x, y) / shadowSize, light.id)).x;
      shadow += shadowDepth < lightPos.z ? 0.0 : 1.0 / 16.0;
    }
  }

  vec3 tangentVec = -pointToLight - normalize(light.direction) * lightDot * length(pointToLight);
  float radius = length(tangentVec);
  float distMult = saturate((light.outerRadius - radius) / (light.outerRadius - light.innerRadius));
  return max(dot(pointToLightDir, normal), 0.0) * light.color * distMult * shadow;
}

vec3 ComputeLighting(vec3 worldPos, vec3 normal, SpotLight light) {
  if (length(light.direction) < 1e-5) {
    return vec3(0);
  }
  vec3 pointToLight = light.position - worldPos;
  vec3 pointToLightDir = normalize(pointToLight);
  float lightDot = dot(-pointToLightDir, normalize(light.direction));
  if (lightDot <= 0) {
    return vec3(0);
  }
  vec4 lightPos = lights.TM[light.id] * vec4(worldPos, 1.0);
  lightPos /= lightPos.w;
  vec2 lightUv = lightPos.xy * 0.5 + 0.5;
  float shadow = 0.0;
  ivec2 shadowSize = textureSize(shadowMap, 0).xy;
  for (float x = -1.5; x <= 1.5; x += 1.0) {
    for (float y = -1.5; y <= 1.5; y += 1.0) {
      float shadowDepth = texture(shadowMap, vec3(lightUv + vec2(x, y) / shadowSize, light.id)).x;
      shadow += shadowDepth < lightPos.z - 1e-5f ? 0.0 : 1.0 / 16.0;
    }
  }

  float distMult = saturate((light.innerCos - dot(normalize(light.direction), -normalize(pointToLight))) / (light.innerCos - light.outerCos));
  return max(dot(pointToLightDir, normal), 0.0) * light.color * distMult * shadow;
}

void main() {
  vec3 diffuse = texture(diffuse, fragTexCoord).rgb;
  vec3 normal = texture(normals, fragTexCoord).xyz;
  float emissionMult = texture(normals, fragTexCoord).a;
  float depth = texture(depthTex, fragTexCoord).x;
  vec4 unproj = (invViewProj * vec4(fragTexCoord * vec2(2, 2) - 1, depth, 1));
  unproj /= unproj.w;
  if (emissionMult > 0) {
    outColor.rgb = diffuse * emissionMult;
  } else {
    vec3 lighting = vec3(0);
    for (uint i = 0; i < directLightsCount; ++i) {
      lighting += ComputeLighting(unproj.xyz, normal, lights.directLights[i]);
    }
    for (uint i = 0; i < spotLightsCount; ++i) {
      lighting += ComputeLighting(unproj.xyz, normal, lights.spotLights[i]);
    }
    outColor.rgb = diffuse * lighting;
  }
}
