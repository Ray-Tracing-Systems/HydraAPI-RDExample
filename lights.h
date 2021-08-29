#ifndef LIGHTS_H
#define LIGHTS_H

struct DirectLight {
  float3 position;
  float innerRadius;
  float3 direction;
  float outerRadius;
  float3 color;
  uint32_t id;
};

struct SpotLight {
  float3 position;
  float innerCos;
  float3 direction;
  float outerCos;
  float3 color;
  uint32_t id;
};

const uint32_t MAX_DIRECT_LIGHTS = 1;
const uint32_t MAX_SPOT_LIGHTS = 1;
const uint32_t MAX_LIGHTS = MAX_DIRECT_LIGHTS + MAX_SPOT_LIGHTS;

#endif
