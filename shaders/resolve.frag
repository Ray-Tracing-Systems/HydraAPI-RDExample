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
  vec3 locIdx = (worldPos - bmin) / (bmax - bmin) * gridSize - 0.5f;
  locIdx = clamp(locIdx, vec3(0, 0, 0), gridSize - 1);
  uvec3 idx = uvec3(locIdx);
  uint flatIndices[8];
  flatIndices[0] = idx.x + idx.y * gridSize.x + idx.z * gridSize.x * gridSize.y;
  for (int i = 0; i < 8; ++i) {
    flatIndices[i] = idx.x + (i & 1) + (idx.y + ((i & 2) >> 1)) * gridSize.x + (idx.z + ((i & 4) >> 2)) * gridSize.x * gridSize.y;
  }

  vec3 lerps = vec3(0, 0, 0);
  lerps = locIdx - idx;
  /*if (uint(locIdx.x + 0.5) > idx.x) {
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
  }*/
  vec3 lighting[8];
  float byNormalWeights[8];
  normal = normalize(normal);
  for (uint i = 0; i < 8; ++i) {
    vec4 row0 = lightingWeightsBuffer[3 * flatIndices[i] + 0];
    vec4 row1 = lightingWeightsBuffer[3 * flatIndices[i] + 1];
    vec4 row2 = lightingWeightsBuffer[3 * flatIndices[i] + 2];
    //vec3 inNormalSpaceCoords = vec3(getAnglesForNormal(normal), 1.0);
    //vec3 baricentric = vec3(dot(row0.xyz, inNormalSpaceCoords), dot(row1.xyz, inNormalSpaceCoords), dot(row2.xyz, inNormalSpaceCoords));
    ////baricentric = inNormalSpaceCoords.x * row0.xyz + inNormalSpaceCoords.y * row1.xyz + inNormalSpaceCoords.z * row2.xyz;
    //baricentric = clamp(baricentric, vec3(0.0), vec3(1.0));
    //float sum = dot(baricentric, vec3(1, 1, 1));
    //baricentric /= sum == 0 ? 1.0f : sum;

    vec3 weights = vec3(0, 0, 0);// baricentric;
    weights.x = max(dot(row0.xyz, normal), 0.0);
    weights.y = max(dot(row1.xyz, normal), 0.0);
    weights.z = max(dot(row2.xyz, normal), 0.0);
    float weightSum = dot(weights, vec3(1, 1, 1));
    bool zeroWeights = weightSum < 1e-5;
    if (weightSum > 1.0f)
      weights /= weightSum;

    byNormalWeights[i] = zeroWeights ? 1.0 : max(row0.w, max(row1.w, row2.w));// max(weights.x * row0.w, max(weights.y * row1.w, weights.z * row2.w));
    //byNormalWeights[i] = zeroWeights ? 1.0 : dot(vec3(row0.w, row1.w, row2.w), weights);// max(weights.x * row0.w, max(weights.y * row1.w, weights.z * row2.w));
    //byNormalWeights[i] = max(weights.x * row0.w, max(weights.y * row1.w, weights.z * row2.w));
    //byNormalWeights[i] = max(weights.x, max(weights.y, weights.z));

    lighting[i].x = dot(lightingBuffer[3 * flatIndices[i] + 0].rgb, weights);
    lighting[i].y = dot(lightingBuffer[3 * flatIndices[i] + 1].rgb, weights);
    lighting[i].z = dot(lightingBuffer[3 * flatIndices[i] + 2].rgb, weights);
  }
  //return lighting[0];
  for (uint i = 0; i < 8; i += 2) {
    lighting[i] = mix(lighting[i], lighting[i + 1], vec3(min(max(1 - byNormalWeights[i], lerps.x), byNormalWeights[i + 1])));
    byNormalWeights[i] = max(byNormalWeights[i], byNormalWeights[i + 1]);
  }
  for (uint i = 0; i < 8; i += 4) {
    lighting[i] = mix(lighting[i], lighting[i + 2], vec3(min(max(1 - byNormalWeights[i], lerps.y), byNormalWeights[i + 2])));
    byNormalWeights[i] = max(byNormalWeights[i], byNormalWeights[i + 2]);
  }
  lighting[0] = mix(lighting[0], lighting[4], vec3(min(max(1 - byNormalWeights[0], lerps.z), byNormalWeights[4])));
  return lighting[0];
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
    outColor.rgb = diffuse * (ComputeLighting(unproj.xyz, normal, lights.directLights[0]) + sampleLighting(unproj.xyz, normal));
  }
  outColor.rgb = pow(outColor.rgb, vec3(1.0 / 2.2));
}
