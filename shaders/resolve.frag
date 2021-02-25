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
  vec3 bmin;
  vec3 bmax;
  uvec3 gridSize;
};

layout(binding = 2) buffer readonly layout1 { vec4 lightingBuffer[]; };
layout(binding = 3) buffer readonly layout2 { vec4 lightingWeightsBuffer[]; };

layout(binding = 4) uniform sampler2D diffuse;
layout(binding = 5) uniform sampler2D normals;
layout(binding = 6) uniform sampler2D depthTex;

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
  for (uint i = 0; i < 8; ++i) {
    vec4 row0 = lightingWeightsBuffer[3 * flatIndices[i] + 0];
    vec4 row1 = lightingWeightsBuffer[3 * flatIndices[i] + 1];
    vec4 row2 = lightingWeightsBuffer[3 * flatIndices[i] + 2];

    vec3 weights = vec3(0, 0, 0);
    weights.x = max(dot(row0.xyz, normal), 0.0) * row0.w;
    weights.y = max(dot(row1.xyz, normal), 0.0) * row1.w;
    weights.z = max(dot(row2.xyz, normal), 0.0) * row2.w;

    vec3 cellLerps = vec3((i & 1) == 1 ? 1.0 - lerps.x : lerps.x, (i & 2) != 0 ? 1.0 - lerps.y : lerps.y, (i & 4) != 0 ? 1.0 - lerps.z : lerps.z);
    cellLerps = 1.0 - cellLerps;

    weights *= cellLerps.x * cellLerps.y * cellLerps.z;
    lighting.x += dot(lightingBuffer[3 * flatIndices[i] + 0].rgb, weights);
    lighting.y += dot(lightingBuffer[3 * flatIndices[i] + 1].rgb, weights);
    lighting.z += dot(lightingBuffer[3 * flatIndices[i] + 2].rgb, weights);
    weightSum += dot(weights, vec3(1, 1, 1));
  }
  return lighting / weightSum;
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
  vec3 normal = texture(normals, fragTexCoord).xyz;
  float emissionMult = texture(normals, fragTexCoord).a;
  float depth = texture(depthTex, fragTexCoord).x;
  vec4 unproj = (invViewProj * vec4(fragTexCoord * vec2(2, 2) - 1, depth, 1));
  unproj /= unproj.w;
  if (emissionMult > 0) {
    outColor.rgb = diffuse * emissionMult;
  } else {
    outColor.rgb = diffuse* (ComputeLighting(unproj.xyz, normal, lights.directLights[0]) + sampleLighting(unproj.xyz, normal));
  }
  outColor.rgb = pow(outColor.rgb, vec3(1.0 / 2.2));
}
