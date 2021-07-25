#version 450
#extension GL_ARB_separate_shader_objects : enable

#define float3 vec3
#define uint32_t uint

#include "../lights.h"

layout(push_constant) uniform lighttmPC
{
  float directMultiplier;
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
  vec3 bmin;
  vec3 bmax;
  uvec3 gridSize;
  vec3 averageLighting;
};

const int MAX_VIRTUAL_PATCHES = 4;

layout(binding = 2) buffer readonly layout1 { vec4 lightingBuffer[]; };
layout(binding = 3) buffer readonly layout2 { vec4 lightingWeightsBuffer[]; };

layout(binding = 4) uniform sampler2D diffuse;
layout(binding = 5) uniform sampler2D normals;
layout(binding = 6) uniform sampler2D depthTex;
layout(binding = 7) uniform sampler2DArray shadowMap;

layout(location = 0) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

float saturate(float a) {
  return clamp(a, 0, 1);
}

const float PI = 3.14159265359f;

vec2 getAnglesForNormal(vec3 normal) {
  return vec2(acos(normal.z), atan(normal.y, normal.x));
}

vec3 sampleLighting(vec3 worldPos, vec3 normal) {
  vec3 locIdx = (worldPos - bmin) / (bmax - bmin) * gridSize - 0.5;
  vec3 threeDimIndices[8];
  uvec3 cellThreeDimIndices[8];
  for (int i = 0; i < 8; ++i) {
    threeDimIndices[i] = locIdx + vec3(i & 1, (i & 2) >> 1, (i & 4) >> 2);
    threeDimIndices[i] = clamp(threeDimIndices[i], vec3(0, 0, 0), gridSize - 1);
    cellThreeDimIndices[i] = uvec3(threeDimIndices[i]);
  }
  locIdx = clamp(locIdx, vec3(0, 0, 0), gridSize - 1);
  uvec3 idx = uvec3(locIdx);
  uint flatIndices[8];
  flatIndices[0] = idx.x + idx.y * gridSize.x + idx.z * gridSize.x * gridSize.y;
  for (int i = 0; i < 8; ++i) {
    flatIndices[i] = cellThreeDimIndices[i].x + cellThreeDimIndices[i].y * gridSize.x + cellThreeDimIndices[i].z * gridSize.x * gridSize.y;
  }

  vec3 lerps = vec3(0, 0, 0);
  lerps = locIdx - idx;

  vec3 lighting = vec3(0, 0, 0);
  float weightSum = 0.0;
  normal = normalize(normal);
  bool isEmpty[8];
  for (uint i = 0; i < 8; ++i) {
    isEmpty[i] = true;
    for (int j = 0; j < MAX_VIRTUAL_PATCHES; ++j) {
      vec4 row0 = lightingWeightsBuffer[MAX_VIRTUAL_PATCHES * flatIndices[i] + j];
      isEmpty[i] = isEmpty[i] && max(dot(row0.xyz, normal), 0.0) * row0.w < 1e-5;
    }
  }
  float lerpWeights[8];
  for (uint i = 0; i < 8; ++i) {
    vec3 cellLerps = vec3((i & 1) == 1 ? 1.0 - lerps.x : lerps.x, (i & 2) != 0 ? 1.0 - lerps.y : lerps.y, (i & 4) != 0 ? 1.0 - lerps.z : lerps.z);
    cellLerps = 1.0 - cellLerps;
    lerpWeights[i] = (isEmpty[i ^ 0x1] ? 1.0 : cellLerps.x) * (isEmpty[i ^ 0x2] ? 1.0 : cellLerps.y) * (isEmpty[i ^ 0x4] ? 1.0 : cellLerps.z);
  }

  for (uint i = 0; i < 8; ++i) {
    vec3 voxelLighting = vec3(0);
    float voxelWeight = 0;
    for (int j = 0; j < MAX_VIRTUAL_PATCHES; ++j) {
      vec4 row = lightingWeightsBuffer[MAX_VIRTUAL_PATCHES * flatIndices[i] + j];
      float weight = max(dot(row.xyz, normal), 0.0) * row.w * lerpWeights[i];
      voxelWeight += weight;
      voxelLighting += lightingBuffer[MAX_VIRTUAL_PATCHES * flatIndices[i] + j].rgb * weight;
    }
    lighting += voxelLighting;
    weightSum += voxelWeight;
  }
  lighting /= max(weightSum, 1.0);
  return mix(averageLighting, lighting, min(1.0, weightSum));
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
    outColor.rgb = diffuse * (lighting * directMultiplier + sampleLighting(unproj.xyz, normal));
  }
  outColor.rgb = pow(outColor.rgb, vec3(1.0 / 2.2));
}
