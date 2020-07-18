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

layout(binding = 3) uniform sampler2D diffuse;
layout(binding = 4) uniform sampler2D normals;
layout(binding = 5) uniform sampler2D depthTex;

layout(location = 0) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

float saturate(float a) {
  return clamp(a, 0, 1);
}

vec3 sampleLighting(vec3 worldPos) {
  vec3 locIdx = (worldPos - bmin) / (bmax - bmin + 1e-5f) * gridSize;
  uvec3 idx = uvec3(locIdx);
  uint flatIndices[8];
  flatIndices[0] = idx.x + idx.y * gridSize.x + idx.z * gridSize.x * gridSize.y;
  vec3 lerps;
  if (uint(locIdx.x + 0.5) > idx.x) {
    flatIndices[1] = (idx.x + 1 == gridSize.x) ? flatIndices[0] : flatIndices[0] + 1;
    lerps.x = locIdx.x - idx.x - 0.5;
  }
  else {
    flatIndices[1] = flatIndices[0];
    flatIndices[0] = (idx.x == 0) ? flatIndices[0] : flatIndices[0] - 1;
    lerps.x = locIdx.x - idx.x + 0.5;
  }
  if (uint(locIdx.y + 0.5) > idx.y) {
    flatIndices[2] = (idx.y + 1 == gridSize.y) ? flatIndices[0] : flatIndices[0] + gridSize.x;
    flatIndices[3] = (idx.y + 1 == gridSize.y) ? flatIndices[1] : flatIndices[1] + gridSize.x;
    lerps.y = locIdx.y - idx.y - 0.5;
  }
  else {
    flatIndices[2] = flatIndices[0];
    flatIndices[3] = flatIndices[1];
    flatIndices[0] = (idx.y == 0) ? flatIndices[0] : flatIndices[0] - gridSize.x;
    flatIndices[1] = (idx.y == 0) ? flatIndices[1] : flatIndices[1] - gridSize.x;
    lerps.y = locIdx.y - idx.y + 0.5;
  }
  if (uint(locIdx.z + 0.5) > idx.z) {
    flatIndices[4] = (idx.z + 1 == gridSize.z) ? flatIndices[0] : flatIndices[0] + gridSize.x * gridSize.y;
    flatIndices[5] = (idx.z + 1 == gridSize.z) ? flatIndices[1] : flatIndices[1] + gridSize.x * gridSize.y;
    flatIndices[6] = (idx.z + 1 == gridSize.z) ? flatIndices[2] : flatIndices[2] + gridSize.x * gridSize.y;
    flatIndices[7] = (idx.z + 1 == gridSize.z) ? flatIndices[3] : flatIndices[3] + gridSize.x * gridSize.y;
    lerps.z = locIdx.z - idx.z - 0.5;
  }
  else {
    flatIndices[4] = flatIndices[0];
    flatIndices[5] = flatIndices[1];
    flatIndices[6] = flatIndices[2];
    flatIndices[7] = flatIndices[3];
    flatIndices[0] = (idx.z == 0) ? flatIndices[0] : flatIndices[0] - gridSize.x * gridSize.y;
    flatIndices[1] = (idx.z == 0) ? flatIndices[1] : flatIndices[1] - gridSize.x * gridSize.y;
    flatIndices[2] = (idx.z == 0) ? flatIndices[2] : flatIndices[2] - gridSize.x * gridSize.y;
    flatIndices[3] = (idx.z == 0) ? flatIndices[3] : flatIndices[3] - gridSize.x * gridSize.y;
    lerps.z = locIdx.z - idx.z + 0.5;
  }
  vec3 lighting[8];
  for (uint i = 0; i < 8; ++i) {
    lighting[i].x = dot(lightingBuffer[3 * flatIndices[i] + 0], vec4(0.25, 0.25, 0.25, 0.25));
    lighting[i].y = dot(lightingBuffer[3 * flatIndices[i] + 1], vec4(0.25, 0.25, 0.25, 0.25));
    lighting[i].z = dot(lightingBuffer[3 * flatIndices[i] + 2], vec4(0.25, 0.25, 0.25, 0.25));
  }
  for (uint i = 0; i < 8; i += 2) {
    lighting[i] = mix(lighting[i], lighting[i + 1], vec3(lerps.x));
  }
  for (uint i = 0; i < 8; i += 4) {
    lighting[i] = mix(lighting[i], lighting[i + 2], vec3(lerps.y));
  }
  return mix(lighting[0], lighting[4], vec3(lerps.z));
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
    outColor.rgb = diffuse * (ComputeLighting(unproj.xyz, normal, lights.directLights[0]) + sampleLighting(unproj.xyz));
  }
}
