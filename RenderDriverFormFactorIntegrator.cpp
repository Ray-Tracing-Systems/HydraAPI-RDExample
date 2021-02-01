// This is a personal academic project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

#include <array>
#include <chrono>
#include <filesystem>

#include <cstdio>
#include <cassert>

#include <embree3/rtcore.h>
#include <omp.h>

#include <LiteMath.h>

#include "RenderDriverFormFactorIntegrator.h"
#include "dataConfig.h"

using namespace HydraLiteMath;

static HRRenderRef  renderRef;
static HRCameraRef  camRef;
static HRSceneInstRef scnRef;
static std::unordered_map<std::wstring, std::wstring> camParams;

const float PI = 3.14159265359f;

template <typename T>
T pow2(T x) {
  return x * x;
}

IHRRenderDriver* CreateFFIntegrator_RenderDriver()
{
  return new RD_FFIntegrator;
}

HRRenderUpdateInfo RD_FFIntegrator::HaveUpdateNow(int a_maxRaysPerPixel)
{
  HRRenderUpdateInfo res;
  res.finalUpdate = true;
  res.haveUpdateFB = true;
  res.progress = 100.0f;
  return res;
}

static std::vector<RD_FFIntegrator::Triangle> gen_triangles(const HRMeshDriverInput& a_input) {

  std::vector<RD_FFIntegrator::Triangle> triangles;
  const uint32_t* indices = reinterpret_cast<const uint32_t*>(a_input.indices);
  const float4* positions = reinterpret_cast<const float4*>(a_input.pos4f);
  const float4* normals = reinterpret_cast<const float4*>(a_input.norm4f);
  const float4* tangents = reinterpret_cast<const float4*>(a_input.tan4f);
  const float2* texCoords = reinterpret_cast<const float2*>(a_input.texcoord2f);
  const uint32_t* materials = reinterpret_cast<const uint32_t*>(a_input.triMatIndices);

  const bool hasTangent = a_input.tan4f;
  const bool hasNormals = a_input.norm4f;

  for (size_t i = 0, ie = (size_t)a_input.triNum * 3; i < ie; i += 3) {
    RD_FFIntegrator::Triangle tri;
    if (hasTangent) {
      tri.tangent.emplace(std::array<float4, 3>());
    }
    if (hasNormals) {
      tri.normal.emplace(std::array<float4, 3>());
    }
    for (size_t j = 0; j < 3; ++j) {
      const size_t index = (size_t)indices[i + j];
      tri.points[j] = positions[index];
      tri.texCoords[j] = texCoords[index];
      if (hasNormals) {
        tri.normal.value()[j] = normals[index];
      }
      if (hasTangent) {
        tri.tangent.value()[j] = tangents[index];
      }
    }
    tri.materialId = materials[i / 3];
    triangles.push_back(tri);
  }
  return triangles;

}

void RD_FFIntegrator::BeginScene(pugi::xml_node a_sceneNode)
{
  allRemapLists.clear();
  tableOffsetsAndSize.clear();

  pugi::xml_node remapLists = a_sceneNode.child(L"remap_lists");
  if (remapLists != nullptr)
  {
    for (pugi::xml_node listNode : remapLists.children())
    {
      const wchar_t* inputStr = listNode.attribute(L"val").as_string();
      const int listSize = listNode.attribute(L"size").as_int();
      std::wstringstream inStrStream(inputStr);

      tableOffsetsAndSize.push_back(int2(int(allRemapLists.size()), listSize));

      for (int i = 0; i < listSize; i++)
      {
        if (inStrStream.eof())
          break;

        int data;
        inStrStream >> data;
        allRemapLists.push_back(data);
      }
    }
  }
}

bool RD_FFIntegrator::UpdateMesh(int32_t a_meshId, pugi::xml_node a_meshNode, const HRMeshDriverInput& a_input, const HRBatchInfo* a_batchList, int32_t listSize) {
  meshTriangles[a_meshId] = gen_triangles(a_input);
  return true;
}

static inline void mat4x4_transpose(float M[16], const float N[16])
{
  for (int j = 0; j < 4; j++)
  {
    for (int i = 0; i < 4; i++)
      M[i * 4 + j] = N[j * 4 + i];
  }
}

void RD_FFIntegrator::InstanceMeshes(int32_t a_mesh_id, const float* a_matrices, int32_t a_instNum, const int* a_lightInstId, const int* a_remapId, const int* a_realInstId) {
  std::vector<float4x4> models(a_instNum, a_matrices);
  for (int32_t i = 0; i < a_instNum; i++)
  {
    float matrixT2[16];
    mat4x4_transpose(matrixT2, (float*)(a_matrices + i * 16));
    models[i] = float4x4(a_matrices);
  }

  for (int i = 0; i < a_instNum; ++i) {
    const auto& mesh = meshTriangles[a_mesh_id];
    for (const Triangle& q : mesh) {
      Triangle instancedTriangle;
      if (q.normal.has_value()) {
        instancedTriangle.normal = std::array<float4, 3>();
      }
      if (q.tangent.has_value()) {
        instancedTriangle.tangent = std::array<float4, 3>();
      }
      for (int j = 0; j < instancedTriangle.points.size(); ++j) {
        instancedTriangle.points[j] = mul(models[i], q.points[j]);
        if (q.normal.has_value()) {
          instancedTriangle.normal.value()[j] = mul(models[i], q.normal.value()[j]);
        }
        if (q.tangent.has_value()) {
          instancedTriangle.tangent.value()[j] = mul(models[i], q.tangent.value()[j]);
        }
        instancedTriangle.texCoords[j] = q.texCoords[j];
      }
      instancedTriangle.materialId = q.materialId;
      if (a_remapId[i] != -1) {
        const int jBegin = tableOffsetsAndSize[a_remapId[i]].x;
        const int jEnd = tableOffsetsAndSize[a_remapId[i]].x + tableOffsetsAndSize[a_remapId[i]].y;
        for (int j = jBegin; j < jEnd; j += 2) {
          if (allRemapLists[j] == instancedTriangle.materialId) {
            instancedTriangle.materialId = allRemapLists[j + 1];
            break;
          }
        }
      }
      instanceTriangles.push_back(instancedTriangle);
    }
  }
}

bool RD_FFIntegrator::UpdateMaterial(int32_t a_matId, pugi::xml_node a_materialNode) {
  pugi::xml_node clrNode = a_materialNode.child(L"diffuse").child(L"color");
  pugi::xml_node texNode = a_materialNode.child(L"diffuse").child(L"texture");

  pugi::xml_node emisNode = a_materialNode.child(L"emission").child(L"color"); // no diffuse color ? => draw emission color instead!

  if (clrNode != nullptr)
  {
    const wchar_t* clrStr = nullptr;

    if (clrNode.attribute(L"val") != nullptr)
      clrStr = clrNode.attribute(L"val").as_string();
    else
      clrStr = clrNode.text().as_string();

    if (!std::wstring(clrStr).empty())
    {
      float3 color;
      std::wstringstream input(clrStr);
      input >> color.x >> color.y >> color.z;
      matColors[a_matId] = color;
    }
  }

  if (texNode != nullptr) {
    matTexture[a_matId] = texNode.attribute(L"id").as_int();
  }

  if (emisNode != nullptr)
  {
    const wchar_t* clrStr = nullptr;

    if (emisNode.attribute(L"val") != nullptr)
      clrStr = emisNode.attribute(L"val").as_string();
    else
      clrStr = emisNode.text().as_string();

    if (!std::wstring(clrStr).empty())
    {
      float3 color;
      std::wstringstream input(clrStr);
      input >> color.x >> color.y >> color.z;
      matEmission[a_matId] = color;
    }
  }

  return true;
}

struct Sample {
  float3 pos, normal;
};

struct FFSample {
  float3 pos, normal;
  float3 color;
  float square;
  float3 emission;
  uint32_t primId;
};

template <typename T>
T lerpSquare(T p1, T p2, T p3, T p4, float x, float y) {
  T x1_pos = lerp(p1, p2, x);
  T x2_pos = lerp(p3, p4, x);
  return lerp(x1_pos, x2_pos, y);
}

float radicalInverse_VdC(uint32_t bits) {
  bits = (bits << 16u) | (bits >> 16u);
  bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
  bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
  bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
  bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
  return float(bits) * 2.3283064365386963e-10f; // / 0x100000000
}

float2 hammersley2d(uint32_t i, uint32_t N) {
  return float2(float(i) / float(N), radicalInverse_VdC(i));
}

float3 hemisphereSample_uniform(float2 uv) {
  float phi = uv.y * 2.0f * PI;
  float cosTheta = 1.0f - uv.x;
  cosTheta = std::cos(2.0f * PI * uv.x);
  float sinTheta = std::sqrt(1.0f - cosTheta * cosTheta);
  return float3(std::cos(phi) * sinTheta, std::sin(phi) * sinTheta, cosTheta);
}

const uint32_t SAMPLES_PACKET_SIZE_S = 4;
const uint32_t SAMPLES_PACKET_SIZE_M = 8;
const uint32_t SAMPLES_PACKET_SIZE_L = 16;
const uint32_t SAMPLES_PACKET_SIZE_XL = 32;
const uint32_t SAMPLES_PACKET_SIZE_XXL = 128;

const uint32_t SAMPLES_PACKET_SIZE = SAMPLES_PACKET_SIZE_XXL;
const uint32_t SAMPLES_PAIRS = SAMPLES_PACKET_SIZE * SAMPLES_PACKET_SIZE;
const uint32_t RT_PACKET_SIZE = 16;
static_assert(SAMPLES_PAIRS % RT_PACKET_SIZE == 0);
const uint32_t WORDS_FOR_OCCLUSION = (SAMPLES_PAIRS + RT_PACKET_SIZE - 1) / RT_PACKET_SIZE;

const uint32_t WORDS_FOR_SAMPLES = (SAMPLES_PACKET_SIZE + RT_PACKET_SIZE - 1) / RT_PACKET_SIZE;

using SamplesPacket = std::array<Sample, SAMPLES_PACKET_SIZE>;


static std::vector<SamplesPacket> gen_samples(const std::vector<RD_FFIntegrator::Triangle>& triangles) {
  std::vector<float3> randomValues(SAMPLES_PACKET_SIZE);
  for (int i = 0; i < SAMPLES_PACKET_SIZE; ++i) {
    //randomValues[i] = hammersley2d(i / PER_AXIS_COUNT, i % PER_AXIS_COUNT);
    randomValues[i].x = static_cast<float>(rand()) / RAND_MAX;
    randomValues[i].y = static_cast<float>(rand()) / RAND_MAX;
    randomValues[i].z = static_cast<float>(rand()) / RAND_MAX;
    float sum = dot(randomValues[i], float3(1, 1, 1));
    randomValues[i] /= sum;
  }
  std::vector<SamplesPacket> samples;
  for (int i = 0; i < triangles.size(); ++i) {
    SamplesPacket sm;
    for (int j = 0; j < SAMPLES_PACKET_SIZE; ++j) {
      float4 pos = randomValues[j].x * triangles[i].points[0] + randomValues[j].y * triangles[i].points[1] + randomValues[j].z * triangles[i].points[2];
      const auto& normArray = triangles[i].normal.value();
      float4 normal = randomValues[j].x * normArray[0] + randomValues[j].y * normArray[1] + randomValues[j].z * normArray[2];
      sm[j].pos = float3(pos.x, pos.y, pos.z);
      sm[j].normal = normalize(float3(normal.x, normal.y, normal.z));
    }
    samples.push_back(sm);
  }
  return samples;
}

static float triangle_square(const RD_FFIntegrator::Triangle& triangle) {
  const float3 p0(triangle.points[0].x, triangle.points[0].y, triangle.points[0].z);
  const float3 p1(triangle.points[1].x, triangle.points[1].y, triangle.points[1].z);
  const float3 p2(triangle.points[2].x, triangle.points[2].y, triangle.points[2].z);
  return length(cross(p1 - p0, p2 - p0)) * 0.5f;
}

class Timer {
  std::chrono::time_point<std::chrono::steady_clock> start;
  std::string name;
  int index = -1;
public:
  Timer(const std::string& s) {
    start = std::chrono::high_resolution_clock::now();
    name = s;
  }
  Timer(const std::string& s, int idx) {
    start = std::chrono::high_resolution_clock::now();
    name = s;
    index = idx;
  }

  ~Timer() {
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);
    if (index != -1)
      std::cout << duration.count() / 1000.f << " ms for " << name << ' ' << index << std::endl;
    else
      std::cout << duration.count() / 1000.f << " ms for " << name << std::endl;
  }
};

static std::vector<RD_FFIntegrator::Quad> merge_quads(const std::vector<RD_FFIntegrator::Quad>& quads, int &tessFactor) {
  std::vector<RD_FFIntegrator::Quad> result;
  std::vector<int> quadEnds(1, 0);
  for (int i = 1; i < quads.size(); ++i) {
    if (length(quads[i].normal.value()[0] - quads[i - 1].normal.value()[0]) > 1e-3) {
      quadEnds.push_back(i);
    }
  }
  result.resize(quadEnds.size());
  const bool hasNormals = quads[0].normal.has_value();
  const bool hasTangent = quads[0].tangent.has_value();
  for (int i = 0; i < quadEnds.size(); ++i) {
    if (hasNormals) {
      result[i].normal = std::array<float4, 4>();
    }
    if (hasTangent) {
      result[i].tangent = std::array<float4, 4>();
    }
  }
  quadEnds.push_back(static_cast<int>(quads.size()));
  for (int i = 1; i < quadEnds.size(); ++i) {
    tessFactor = static_cast<int>(std::sqrt(static_cast<float>(quadEnds[i] - quadEnds[i - 1])) + 0.5f);
    const std::array<int, 4> cornerQuadIndices = { quadEnds[i - 1], quadEnds[i] - tessFactor, quadEnds[i] - 1, quadEnds[i - 1] + tessFactor - 1 };
    for (int j = 0; j < 4; ++j) {
      result[i - 1].points[j] = quads[cornerQuadIndices[j]].points[j];
      if (hasNormals) {
        result[i - 1].normal.value()[j] = quads[cornerQuadIndices[j]].normal.value()[j];
      }
      if (hasTangent) {
        result[i - 1].tangent.value()[j] = quads[cornerQuadIndices[j]].tangent.value()[j];
      }
      result[i - 1].texCoords[j] = quads[cornerQuadIndices[j]].texCoords[j];
    }
    result[i - 1].materialId = quads[quadEnds[i - 1]].materialId;
  }
  return result;
}

#define CHECK_EMBREE \
{ \
    int errCode; \
    if ((errCode = rtcGetDeviceError(Device)) != RTC_ERROR_NONE) {\
        std::cerr << "Embree error: " << errCode  << " line: " << __LINE__ << std::endl; \
        exit(1); \
    }\
}

class EmbreeTracer {
  RTCDevice Device;
  RTCScene Scene;
  RTCIntersectContext IntersectionContext;

public:
  EmbreeTracer(const std::vector<RD_FFIntegrator::Triangle>& triangles) {
    std::vector<float3> points(triangles.size() * 3);
    std::vector<uint32_t> indices(triangles.size() * 3);
    for (int i = 0; i < triangles.size(); ++i) {
      for (int j = 0; j < 3; ++j) {
        points[i * 3 + j] = make_float3(triangles[i].points[j]);
        indices[i * 3 + j] = i * 3 + j;
      }
    }

    Device = rtcNewDevice("threads=0,frequency_level=simd512"); CHECK_EMBREE
    Scene = rtcNewScene(Device); CHECK_EMBREE
    RTCGeometry geometry = rtcNewGeometry(Device, RTC_GEOMETRY_TYPE_TRIANGLE); CHECK_EMBREE
    RTCBuffer indicesBuffer = rtcNewSharedBuffer(Device, reinterpret_cast<void*>(indices.data()), indices.size() * sizeof(indices[0])); CHECK_EMBREE
    RTCBuffer pointsBuffer = rtcNewSharedBuffer(Device, reinterpret_cast<void*>(points.data()), points.size() * sizeof(points[0])); CHECK_EMBREE

    rtcSetGeometryBuffer(geometry, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, indicesBuffer, 0, sizeof(uint32_t) * 3, indices.size() / 3); CHECK_EMBREE
    rtcSetGeometryBuffer(geometry, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, pointsBuffer, 0, sizeof(points[0]), points.size()); CHECK_EMBREE
    rtcCommitGeometry(geometry); CHECK_EMBREE
    rtcAttachGeometry(Scene, geometry); CHECK_EMBREE
    rtcReleaseGeometry(geometry); CHECK_EMBREE
    rtcCommitScene(Scene); CHECK_EMBREE
    rtcInitIntersectContext(&IntersectionContext); CHECK_EMBREE
    IntersectionContext.flags = RTC_INTERSECT_CONTEXT_FLAG_COHERENT;
    IntersectionContext.instID[0] = 0;

    RTCBounds bounds;
    rtcGetSceneBounds(Scene, &bounds);
  }

  std::array<uint16_t, WORDS_FOR_OCCLUSION> traceRays(const SamplesPacket& samples1, const SamplesPacket& samples2) {
    std::array<uint16_t, WORDS_FOR_OCCLUSION> result = { };
    std::array<RTCRay16, WORDS_FOR_OCCLUSION> raysPacket = {};
    for (int i = 0, target_id = 0; i < SAMPLES_PACKET_SIZE; ++i) {
      const Sample& s1 = samples1[i];
      for (int j = 0; j < SAMPLES_PACKET_SIZE; ++j, ++target_id) {
        const uint32_t packet_id = target_id / RT_PACKET_SIZE;
        const uint32_t sampleInPacket = target_id % RT_PACKET_SIZE;
        const Sample& s2 = samples2[j];
        const float BIAS = 1e-6f;
        float3 dir = s2.pos - s1.pos;
        raysPacket[packet_id].org_x[sampleInPacket] = s1.pos.x;
        raysPacket[packet_id].org_y[sampleInPacket] = s1.pos.y;
        raysPacket[packet_id].org_z[sampleInPacket] = s1.pos.z;
        raysPacket[packet_id].tnear[sampleInPacket] = BIAS;
        raysPacket[packet_id].dir_x[sampleInPacket] = dir.x;
        raysPacket[packet_id].dir_y[sampleInPacket] = dir.y;
        raysPacket[packet_id].dir_z[sampleInPacket] = dir.z;
        raysPacket[packet_id].tfar[sampleInPacket] = 1.f - BIAS;
      }
    }

    const int validMask = ~0u;
    for (int i = 0; i < WORDS_FOR_OCCLUSION; ++i) {
      rtcOccluded16(&validMask, Scene, &IntersectionContext, &raysPacket[i]); //CHECK_EMBREE
      for (uint32_t k = 0; k < RT_PACKET_SIZE; ++k) {
        if (std::isinf(raysPacket[i].tfar[k])) {
          result[i] |= 1u << k;
        }
      }
    }

    return result;
  }

  std::vector<uint16_t> traceRays(const std::vector<FFSample>& samples1, const std::vector<FFSample>& samples2) {
    const uint32_t samplesPairs = static_cast<uint32_t>(samples1.size() * samples2.size());
    const uint32_t samplesWords = (samplesPairs + 15) / 16;
    std::vector<uint16_t> result(samplesWords, 0);
    std::vector<RTCRay16> raysPacket(samplesWords);
    for (int i = 0, target_id = 0; i < samples1.size(); ++i) {
      const FFSample& s1 = samples1[i];
      for (int j = 0; j < samples2.size(); ++j, ++target_id) {
        const uint32_t packet_id = target_id / RT_PACKET_SIZE;
        const uint32_t sampleInPacket = target_id % RT_PACKET_SIZE;
        const FFSample& s2 = samples2[j];
        const float BIAS = 1e-6f;
        float3 dir = s2.pos - s1.pos;
        raysPacket[packet_id].org_x[sampleInPacket] = s1.pos.x;
        raysPacket[packet_id].org_y[sampleInPacket] = s1.pos.y;
        raysPacket[packet_id].org_z[sampleInPacket] = s1.pos.z;
        raysPacket[packet_id].tnear[sampleInPacket] = BIAS;
        raysPacket[packet_id].dir_x[sampleInPacket] = dir.x;
        raysPacket[packet_id].dir_y[sampleInPacket] = dir.y;
        raysPacket[packet_id].dir_z[sampleInPacket] = dir.z;
        raysPacket[packet_id].tfar[sampleInPacket] = 1.f - BIAS;
      }
    }

    const int validMask = ~0u;
    for (uint32_t i = 0; i < samplesWords; ++i) {
      rtcOccluded16(&validMask, Scene, &IntersectionContext, &raysPacket[i]); //CHECK_EMBREE
      for (uint32_t k = 0; k < RT_PACKET_SIZE; ++k) {
        if (std::isinf(raysPacket[i].tfar[k])) {
          result[i] |= 1u << k;
        }
      }
    }

    return result;
  }

  std::vector<FFSample> traceRaysTo(const std::array<float3, SAMPLES_PACKET_SIZE>& dirs, const float3& target, const std::vector<float3>& colors, const std::vector<float3>& emissions, const std::vector<float>& squares, const std::vector<float3>& normals) {
    // Generate positions
    std::array<float3, SAMPLES_PACKET_SIZE> positions;
    for (uint32_t i = 0; i < positions.size(); ++i) {
      positions[i] = target - dirs[i];
    }
    // Fill packets to trace
    const float BIAS = 1e-6f;
    std::array<RTCRayHit16, WORDS_FOR_SAMPLES> rays;
    std::vector<FFSample> samplesDEBUG;
 /*   for (uint32_t i = 0; i < positions.size(); ++i) {
      samplesDEBUG.push_back(FFSample{ positions[i], float3(1, 0, 0), float3(1, 0, 0), 0 });
    }
    return samplesDEBUG;*/
    //samplesDEBUG.push_back(FFSample{ positions[0], float3(0, 0, 0), float3(0, 0, 0), 0 });
    //return samplesDEBUG;
    //samplesDEBUG.push_back(FFSample{ positions[0] + dirs[0] * 2.0f, float3(0, 0, 0), float3(0, 0, 0), 0 });
    //return samplesDEBUG;
    for (uint32_t packetId = 0, rayId = 0; packetId < rays.size(); ++packetId) {
      for (uint32_t localId = 0; localId < RT_PACKET_SIZE && rayId < dirs.size(); ++localId, ++rayId) {
        rays[packetId].ray.dir_x[localId] = dirs[rayId].x * 2;
        rays[packetId].ray.dir_y[localId] = dirs[rayId].y * 2;
        rays[packetId].ray.dir_z[localId] = dirs[rayId].z * 2;
        rays[packetId].ray.tnear[localId] = BIAS;
        rays[packetId].ray.org_x[localId] = positions[rayId].x;
        rays[packetId].ray.org_y[localId] = positions[rayId].y;
        rays[packetId].ray.org_z[localId] = positions[rayId].z;
        rays[packetId].ray.tfar [localId] = 1.f + BIAS;
      }
    }
    // Trace rays
    const int validMask = ~0u;
    for (uint32_t i = 0; i < WORDS_FOR_SAMPLES; ++i) {
      rtcIntersect16(&validMask, Scene, &IntersectionContext, &rays[i]); CHECK_EMBREE
    }
    // Gather sampled results
    std::vector<FFSample> samples;
    for (uint32_t packetId = 0, rayId = 0; packetId < WORDS_FOR_SAMPLES; ++packetId) {
      for (uint32_t localId = 0; localId < RT_PACKET_SIZE && rayId < dirs.size(); ++localId, ++rayId) {
        // Skip no hit
        if (std::isinf(rays[packetId].ray.tfar[localId]) || rays[packetId].ray.tfar[localId] >= 1.f + BIAS) {
          continue;
        }
        // Skip back faces
        const float3 normal = normalize(normals[rays[packetId].hit.primID[localId]]);
        if (dot(normal, -dirs[rayId]) < 0.0) {
          continue;
        }
        // Extract color and square by polygonId
        const float3 color = colors[rays[packetId].hit.primID[localId]];
        const float3 emission = emissions[rays[packetId].hit.primID[localId]];
        const float square = squares[rays[packetId].hit.primID[localId]];
        // Generate sample
        const float3 position = positions[rayId] + 2.0f * dirs[rayId] * rays[packetId].ray.tfar[localId];
        samples.push_back(FFSample{ position, normal, color, square, emission, rays[packetId].hit.primID[localId] });
      }
    }
    return samples;
  }

  ~EmbreeTracer() {
    rtcReleaseScene(Scene);
    rtcReleaseDevice(Device);
  }
};

void RD_FFIntegrator::ComputeFF(uint32_t quadsCount, std::vector<RD_FFIntegrator::Triangle>& triangles, const std::vector<float>& squares)
{
  FF.resize(quadsCount);

  std::wstring ffFilename = DataConfig::get().getBinFilePath(L"FF.bin");
  std::ifstream fin(ffFilename, std::ios::binary);
  if (fin.is_open()) {
    uint32_t countFromFile = 0;
    uint32_t ffVersion = 0;
    fin.read(reinterpret_cast<char*>(&ffVersion), sizeof(ffVersion));
    if (ffVersion == DataConfig::FF_VERSION) {
      fin.read(reinterpret_cast<char*>(&countFromFile), sizeof(countFromFile));
      assert(countFromFile == quadsCount);
      for (uint32_t i = 0; i < quadsCount; ++i) {
        uint32_t rowSize;
        fin.read(reinterpret_cast<char*>(&rowSize), sizeof(rowSize));
        FF[i].resize(rowSize);
        float rowSum = 0;
        for (uint32_t j = 0; j < rowSize; ++j) {
          uint32_t idx;
          float value;
          fin.read(reinterpret_cast<char*>(&idx), sizeof(idx));
          fin.read(reinterpret_cast<char*>(&value), sizeof(value));
          FF[i][j].first = idx;
          FF[i][j].second = value;
          assert(value <= 1 && "Too big FF");
          rowSum += value;
        }
      }
      fin.close();
      return;
    } else {
      fin.close();
    }
  }
  std::ofstream fout(ffFilename, std::ios::binary);

  std::vector<SamplesPacket> samples = gen_samples(instanceTriangles);

  EmbreeTracer tracer(triangles);

  omp_set_dynamic(0);

  std::vector<uint32_t> patchesToProcess(quadsCount);
  std::vector<Sample> patchesToCompute(quadsCount);
  std::vector<uint16_t> occlRes(quadsCount);

  for (int i = 0; i < static_cast<int>(quadsCount) - 1; ++i) {
    //Timer timer("row", i);
    if (100 * i / quadsCount < 100 * (i + 1) / quadsCount) {
      std::cout << 100 * (i + 1) / quadsCount << "% finished" << std::endl;
    }
    const SamplesPacket& samples1 = samples[i];

    patchesToProcess.clear();
    for (uint32_t j = i + 1; j < quadsCount; ++j) {
      const SamplesPacket& samples2 = samples[j];
      const float3 posToPos = samples1[0].pos - samples2[0].pos;
      if (!(dot(samples1[0].normal, posToPos) >= 0 || dot(samples2[0].normal, posToPos) <= 0)) {
        patchesToProcess.push_back(j);
      }
    }

#pragma omp parallel for num_threads(8)
    for (int idx = 0; idx < patchesToProcess.size(); ++idx) {
      int j = patchesToProcess[idx];
      const SamplesPacket& samples2 = samples[j];

      auto occluded = tracer.traceRays(samples1, samples2);
      bool fullOcclusion = true;
      for (auto occl : occluded) {
        fullOcclusion &= (occl == 0xFFFF);
      }
      if (fullOcclusion) {
        continue;
      }
      float value = 0;
      int samplesCount = 0;
      for (int k = 0, occlValue = 0; k < samples1.size(); ++k) {
        const Sample& sample1 = samples1[k];
        for (int h = 0; h < samples2.size(); ++h, occlValue++) {
          const Sample& sample2 = samples2[h];
          if ((occluded[occlValue / RT_PACKET_SIZE] & (1 << (occlValue % RT_PACKET_SIZE))) != 0) {
            samplesCount++;
            continue;
          }
          const float3 r = sample1.pos - sample2.pos;
          const float lengthSq = dot(r, r);
          if (lengthSq < 1e-10) {
            continue;
          }
          const float l = std::sqrt(lengthSq);
          const float invL = 1 / l;
          const float3 toSample = r * invL;
          const float theta1 = max(-dot(sample1.normal, toSample), 0.f);
          const float theta2 = max(dot(sample2.normal, toSample), 0.f);
          value += theta1 * theta2 * invL * invL;
          samplesCount++;
        }
      }
      if (samplesCount > 0) {
        value /= samplesCount * 3.14f;
      }
#pragma omp critical
      if (value > 1e-9) {
        const float val1 = value * squares[j];
        const float val2 = value * squares[i];
        if (val1 > 1e-9) {
          FF[i].emplace_back(j, min(val1, 1));
        }
        if (val2 > 1e-9) {
          FF[j].emplace_back(i, min(val2, 1));
        }
      }
    }
    std::sort(FF[i].begin(), FF[i].end());
    float rowSum = 1e-9f;
    for (const auto& elem : FF[i]) {
      rowSum += elem.second;
    }
    if (rowSum > 1.0f) {
      for (auto& elem : FF[i]) {
        elem.second /= rowSum;
      }
    }
  }

  fout.write(reinterpret_cast<const char*>(&DataConfig::FF_VERSION), sizeof(DataConfig::FF_VERSION));
  fout.write(reinterpret_cast<const char*>(&quadsCount), sizeof(quadsCount));
  for (uint32_t i = 0; i < quadsCount; ++i) {
    uint32_t rowSize = static_cast<uint32_t>(FF[i].size());
    fout.write(reinterpret_cast<char*>(&rowSize), sizeof(rowSize));
    for (auto it = FF[i].begin(); it != FF[i].end(); ++it) {
      uint32_t idx = it->first;
      float value = it->second;
      fout.write(reinterpret_cast<char*>(&idx), sizeof(idx));
      fout.write(reinterpret_cast<char*>(&value), sizeof(value));
    }
  }
  fout.close();
}

void RD_FFIntegrator::ComputeFF_voxelized(
  std::vector<RD_FFIntegrator::Triangle>& triangles,
  const std::vector<float>& squares,
  const std::vector<float3>& voxels_centers,
  float voxel_size,
  std::vector<float3> &colors,
  std::vector<float3> &emission,
  std::vector<float3> &normals)
{
  const uint32_t voxelsCount = static_cast<uint32_t>(voxels_centers.size());
  std::wstring ffFilename = DataConfig::get().getBinFilePath(L"FF_vox.bin");
  std::ifstream fin(ffFilename, std::ios::binary);
  if (fin.is_open() && false) {
    uint32_t countFromFile = 0;
    uint32_t ffVersion = 0;
    fin.read(reinterpret_cast<char*>(&ffVersion), sizeof(ffVersion));
    if (ffVersion == DataConfig::FF_VERSION) {
      fin.read(reinterpret_cast<char*>(&countFromFile), sizeof(countFromFile));
      assert(countFromFile == voxelsCount);
      for (uint32_t i = 0; i < voxelsCount; ++i) {
        uint32_t rowSize;
        fin.read(reinterpret_cast<char*>(&rowSize), sizeof(rowSize));
        FF[i].resize(rowSize);
        float rowSum = 0;
        for (uint32_t j = 0; j < rowSize; ++j) {
          uint32_t idx;
          float value;
          fin.read(reinterpret_cast<char*>(&idx), sizeof(idx));
          fin.read(reinterpret_cast<char*>(&value), sizeof(value));
          FF[i][j].first = idx;
          FF[i][j].second = value;
          assert(value <= 1 && "Too big FF");
          rowSum += value;
        }
        //assert(rowSum <= 1 && "Too big FF sum");
      }
      fin.close();
      return;
    }
    else {
      fin.close();
    }
  }
  std::ofstream fout(ffFilename, std::ios::binary);

  const float halfDiag = voxel_size * 0.5f;
  std::array<float3, SAMPLES_PACKET_SIZE> hammDirs;
  for (uint32_t i = 0; i < hammDirs.size(); ++i) {
    const float u = static_cast<float>(rand()) / RAND_MAX;
    const float v = static_cast<float>(rand()) / RAND_MAX;
    const float theta = u * 2.0f * PI;
    const float phi = 2.0f * v - 1.0f;
    hammDirs[i] = normalize(float3(std::sqrt(1.0f - phi * phi) * std::cos(theta), std::sqrt(1.0f - phi * phi) * std::sin(theta), phi));
    //hammDirs[i] = hemisphereSample_uniform(hammersley2d(i, static_cast<uint32_t>(hammDirs.size())));
    const float dirLength = max(std::abs(hammDirs[i].x), max(std::abs(hammDirs[i].y), std::abs(hammDirs[i].z)));
    hammDirs[i] *= halfDiag / dirLength;
  }

  // Generate samples for each voxel
  std::vector<std::vector<FFSample>> samples(voxelsCount);
  EmbreeTracer tracer(triangles);
  for (uint32_t i = 0; i < voxelsCount; ++i) {
    // Trace rays from sphere surface to sphere center
    samples[i] = tracer.traceRaysTo(hammDirs, voxels_centers[i], colors, emission, squares, normals);
    std::unordered_map<uint32_t, uint32_t> primCounter;
    for (const auto& sample : samples[i]) {
      primCounter[sample.primId]++;
    }
    for (auto& sample : samples[i]) {
      sample.square /= primCounter[sample.primId];
    }
    if (i == 4)
    {
      std::ofstream fout(DataConfig::get().getBinFilePath(L"debugPoints.bin"), std::ios::binary);
      uint32_t samplesCount = static_cast<uint32_t>(samples[i].size());
      fout.write(reinterpret_cast<char*>(&samplesCount), sizeof(samplesCount));
      for (uint32_t j = 0; j < samplesCount; ++j) {
        float3 color(1, 0, 0);
        fout.write(reinterpret_cast<char*>(&samples[i][j].pos), sizeof(samples[i][j].pos));
        //fout.write(reinterpret_cast<char*>(&samples[i][j].color), sizeof(samples[i][j].color));
        fout.write(reinterpret_cast<char*>(&color), sizeof(color));
      }
      fout.close();
      //return;
    }
  }

  // Compute form factors for voxels
  using FFMatrix = std::vector<std::vector<std::pair<uint32_t, float>>>;
  FF.clear();

  std::vector<std::vector<std::pair<uint32_t, float>>> virtualPatchesFF;
  std::vector<std::vector<FFMatrix>> virtualPatchesToPointsFF;
  std::vector<FFMatrix> voxelRow;
  std::vector<uint32_t> patchesToProcess;
  std::vector<FFSample> finalSamples;
  for (uint32_t voxelId = 0; voxelId < voxelsCount; ++voxelId) {
    // Compute form-factors between current voxel and further voxels
    if (100 * voxelId / voxelsCount < 100 * (voxelId + 1) / voxelsCount) {
      std::cout << 100 * (voxelId + 1) / voxelsCount << "% finished" << std::endl;
    }
    std::vector<FFSample>& samples1 = samples[voxelId];
    if (samples1.empty()) {
      for (uint32_t i = 0; i < virtualPatchesToPointsFF.size(); ++i) {
        virtualPatchesToPointsFF[i].erase(virtualPatchesToPointsFF[i].begin());
      }
      continue;
    }

    voxelRow.clear();
    patchesToProcess.clear();
    patchesToProcess.push_back(voxelId);
    for (uint32_t anotherVoxelId = voxelId + 1; anotherVoxelId < voxelsCount; ++anotherVoxelId) {
      const std::vector<FFSample>& samples2 = samples[anotherVoxelId];
      if (samples2.empty()) {
        continue;
      }
      const float3 posToPos = samples1[0].pos - samples2[0].pos;
      //if (!(dot(samples1[0].normal, posToPos) >= 0 || dot(samples2[0].normal, posToPos) <= 0)) {
        patchesToProcess.push_back(anotherVoxelId);
      //}
    }

    voxelRow.resize(voxelsCount - voxelId);
    for (uint32_t i = 0; i < voxelRow.size(); ++i) {
      voxelRow[i].resize(samples1.size());
    }

//#pragma omp parallel for num_threads(8)
    for (int idx = 0; idx < patchesToProcess.size(); ++idx) {
      int j = patchesToProcess[idx];
      voxelRow[j - voxelId].resize(samples1.size());
      const std::vector<FFSample>& samples2 = samples[j];

      auto occluded = tracer.traceRays(samples1, samples2);
      bool fullOcclusion = true;
      for (auto occl : occluded) {
        fullOcclusion &= (occl == 0xFFFF);
      }
      if (fullOcclusion) {
        continue;
      }
      for (int k = 0, occlValue = 0; k < samples1.size(); ++k) {
        const FFSample& sample1 = samples1[k];
        for (int h = 0; h < samples2.size(); ++h, occlValue++) {
          if (idx == 0 && k == h) {
            continue;
          }
          const FFSample& sample2 = samples2[h];
          if ((occluded[occlValue / RT_PACKET_SIZE] & (1 << (occlValue % RT_PACKET_SIZE))) != 0) {
            continue;
          }
          const float3 r = sample1.pos - sample2.pos;
          const float lengthSq = dot(r, r);
          if (lengthSq < 1e-10) {
            continue;
          }
          const float l = std::sqrt(lengthSq);
          const float invL = 1 / l;
          const float3 toSample = r * invL;
          const float theta1 = max(-dot(sample1.normal, toSample), 0.f);
          const float theta2 = max(dot(sample2.normal, toSample), 0.f);
          float value = theta1 * theta2 * invL * invL;
          value *= 1.0f / PI;
          value *= sample2.square;
          if (value > 1e-9f) {
            voxelRow[j - voxelId][k].emplace_back(h, min(value, 1));
          }
        }
      }
      for (int k = 0; k < samples1.size(); ++k) {
        std::sort(voxelRow[j - voxelId][k].begin(), voxelRow[j - voxelId][k].end());
      }
    }

    // Merge samples by form-factors
    const uint32_t MAX_VIRTUAL_PATCHES = 3;
    while (samples1.size() > MAX_VIRTUAL_PATCHES) {
      std::pair<uint32_t, uint32_t> bestMatched(0, 0);
      float bestSimilarity = 1e9f;
      for (uint32_t i = 0; i < samples1.size() - 1; ++i) {
        for (uint32_t j = i + 1; j < samples1.size(); ++j) {
          const float normalsCoef = 2.0f - dot(samples1[j].normal, samples1[i].normal);
          const float colorsCoef = 1.0f + length(samples1[i].color - samples1[j].color);
          float ffCoef = 1.0f;
          for (int idx = 0; idx < patchesToProcess.size(); ++idx) {
            const FFMatrix& pointsSubFF = voxelRow[patchesToProcess[idx] - voxelId];
            uint32_t iter1 = 0, iter2 = 0;
            while (iter1 < pointsSubFF[i].size() && iter2 < pointsSubFF[j].size()) {
              if (pointsSubFF[i][iter1].first < pointsSubFF[j][iter2].first) {
                ffCoef += pow2(pointsSubFF[i][iter1].second);
                iter1++;
              } else if (pointsSubFF[i][iter1].first > pointsSubFF[j][iter2].first) {
                ffCoef += pow2(pointsSubFF[j][iter2].second);
                iter2++;
              } else {
                ffCoef += pow2(pointsSubFF[i][iter1].second - pointsSubFF[j][iter2].second);
                iter1++;
                iter2++;
              }
            }
            while (iter1 < pointsSubFF[i].size()) {
              ffCoef += pow2(pointsSubFF[i][iter1].second);
              iter1++;
            }
            while (iter2 < pointsSubFF[j].size()) {
              ffCoef += pow2(pointsSubFF[j][iter2].second);
              iter2++;
            }
          }
          ffCoef = std::sqrt(ffCoef);
          const float similarity = normalsCoef * colorsCoef * ffCoef;
          if (similarity < bestSimilarity) {
            bestSimilarity = similarity;
            bestMatched = std::make_pair(i, j);
          }
        }
      }

      const float weight1 = samples1[bestMatched.first].square / (samples1[bestMatched.first].square + samples1[bestMatched.second].square);
      const float weight2 = 1.0f - weight1;
      samples1[bestMatched.first].color = samples1[bestMatched.first].color * weight1 + samples1[bestMatched.second].color * weight2;
      samples1[bestMatched.first].emission = samples1[bestMatched.first].emission * weight1 + samples1[bestMatched.second].emission * weight2;
      samples1[bestMatched.first].normal = normalize(samples1[bestMatched.first].normal * weight1 + samples1[bestMatched.second].normal * weight2);
      samples1[bestMatched.first].square += samples1[bestMatched.second].square;
      for (int idx = 0; idx < voxelRow.size(); ++idx) {
        std::vector<std::pair<uint32_t, float>> newFFRow;
        const FFMatrix& pointsSubFF = voxelRow[idx];
        uint32_t iter1 = 0, iter2 = 0;
        while (iter1 < pointsSubFF[bestMatched.first].size() && iter2 < pointsSubFF[bestMatched.second].size()) {
          if (pointsSubFF[bestMatched.first][iter1].first < pointsSubFF[bestMatched.second][iter2].first) {
            newFFRow.emplace_back(pointsSubFF[bestMatched.first][iter1]);
            iter1++;
          } else if (pointsSubFF[bestMatched.first][iter1].first > pointsSubFF[bestMatched.second][iter2].first) {
            newFFRow.emplace_back(pointsSubFF[bestMatched.second][iter2]);
            iter2++;
          } else {
            newFFRow.emplace_back(
              pointsSubFF[bestMatched.first][iter1].first,
              pointsSubFF[bestMatched.first][iter1].second * weight1 + pointsSubFF[bestMatched.second][iter2].second * weight2
            );
            iter1++;
            iter2++;
          }
        }
        while (iter1 < pointsSubFF[bestMatched.first].size()) {
          newFFRow.emplace_back(pointsSubFF[bestMatched.first][iter1]);
          iter1++;
        }
        while (iter2 < pointsSubFF[bestMatched.second].size()) {
          newFFRow.emplace_back(pointsSubFF[bestMatched.second][iter2]);
          iter2++;
        }
        voxelRow[idx][bestMatched.first] = newFFRow;
        voxelRow[idx].erase(voxelRow[idx].begin() + bestMatched.second);
      }
      samples1.erase(samples1.begin() + bestMatched.second);
      for (uint32_t j = 0; j < voxelRow[0].size(); ++j) {
        std::vector<std::pair<uint32_t, float>>& row = voxelRow[0][j];
        auto dest = std::find_if(row.begin(), row.end(), [&bestMatched](const std::pair<uint32_t, float>& elem) {return bestMatched.first == elem.first; });
        auto src = std::find_if(row.begin(), row.end(), [&bestMatched](const std::pair<uint32_t, float>& elem) {return bestMatched.second == elem.first; });
        if (dest != row.end()) {
          if (src != row.end()) {
            dest->second += src->second;
            row.erase(src);
          }
        } else {
          if (src != row.end()) {
            const float value = src->second;
            row.erase(src);
            auto targetToAdd = std::find_if(row.begin(), row.end(), [&bestMatched](const std::pair<uint32_t, float>& elem) {return bestMatched.first < elem.first; });
            row.insert(targetToAdd, std::make_pair(bestMatched.first, value));
          }
        }
        auto indicesToReduceBegin = std::find_if(row.begin(), row.end(), [&bestMatched](const std::pair<uint32_t, float>& elem) {return bestMatched.second < elem.first; });
        std::transform(indicesToReduceBegin, row.end(), indicesToReduceBegin, [](std::pair<uint32_t, float>& elem) { elem.first--; return elem; });
      }
      for (uint32_t i = 0; i < virtualPatchesToPointsFF.size(); ++i) {
        for (uint32_t j = 0; j < virtualPatchesToPointsFF[i][0].size(); ++j) {
          std::vector<std::pair<uint32_t, float>>& row = virtualPatchesToPointsFF[i][0][j];
          auto dest = std::find_if(row.begin(), row.end(), [&bestMatched](const std::pair<uint32_t, float>& elem) {return bestMatched.first == elem.first; });
          auto src = std::find_if(row.begin(), row.end(), [&bestMatched](const std::pair<uint32_t, float>& elem) {return bestMatched.second == elem.first; });
          if (dest != row.end()) {
            if (src != row.end()) {
              dest->second += src->second;
              row.erase(src);
            }
          } else {
            if (src != row.end()) {
              const float value = src->second;
              row.erase(src);
              auto targetToAdd = std::find_if(row.begin(), row.end(), [&bestMatched](const std::pair<uint32_t, float>& elem) {return bestMatched.first < elem.first; });
              row.insert(targetToAdd, std::make_pair(bestMatched.first, value));
            }
          }
          auto indicesToReduceBegin = std::find_if(row.begin(), row.end(), [&bestMatched](const std::pair<uint32_t, float>& elem) {return bestMatched.second < elem.first; });
          std::transform(indicesToReduceBegin, row.end(), indicesToReduceBegin, [](std::pair<uint32_t, float> elem) { elem.first--; return elem; });
        }
      }
    }

    // Add virtual patch to the collection
    // Move virtualPatchesToPointsFF first column to final matrix
    for (uint32_t i = 0, j = 0; i < virtualPatchesToPointsFF.size(); ++i) {
      for (uint32_t k = 0; k < virtualPatchesToPointsFF[i][0].size(); ++k, ++j) {
        const uint32_t samplesOffset = static_cast<uint32_t>(finalSamples.size());
        const uint32_t ffPrevSize = static_cast<uint32_t>(FF[j].size());
        FF[j].resize(FF[j].size() + virtualPatchesToPointsFF[i][0][k].size());
        std::transform(virtualPatchesToPointsFF[i][0][k].begin(), virtualPatchesToPointsFF[i][0][k].end(), FF[j].begin() + ffPrevSize, [samplesOffset](std::pair<uint32_t, float> elem) { elem.first += samplesOffset; return elem; });
      }
    }
    // Add row to final matrix
    FF.resize(FF.size() + samples1.size());
    for (uint32_t i = 0, j = 0; i < virtualPatchesToPointsFF.size(); ++i) {
      for (uint32_t k = 0; k < virtualPatchesToPointsFF[i][0].size(); ++k, ++j) {
        for (uint32_t h = 0; h < virtualPatchesToPointsFF[i][0][k].size(); ++h) {
          const uint32_t samplesOffset = static_cast<uint32_t>(finalSamples.size());
          const uint32_t sampleId = virtualPatchesToPointsFF[i][0][k][h].first + samplesOffset;
          FF[sampleId].emplace_back(j, virtualPatchesToPointsFF[i][0][k][h].second / samples1[virtualPatchesToPointsFF[i][0][k][h].first].square * finalSamples[j].square);
        }
      }
    }
    //Remove first column from virtualPatchesToPointsFF
    for (uint32_t i = 0; i < virtualPatchesToPointsFF.size(); ++i) {
      virtualPatchesToPointsFF[i].erase(virtualPatchesToPointsFF[i].begin());
    }
    // Add row to virtualPatchesToPointsFF
    if (voxelRow.size() > 1) {
      virtualPatchesToPointsFF.emplace_back(voxelRow.begin() + 1, voxelRow.end());
    }
    // Add diagonal submatrix to final matrix
    for (uint32_t i = 0; i < samples1.size(); ++i) {
      const uint32_t samplesOffset = static_cast<uint32_t>(finalSamples.size());
      const uint32_t rowIdx = static_cast<uint32_t>(FF.size() - samples1.size() + i);
      const uint32_t ffPrevSize = static_cast<uint32_t>(FF[rowIdx].size());
      FF[rowIdx].resize(FF[rowIdx].size() + voxelRow[0][i].size());
      std::transform(voxelRow[0][i].begin(), voxelRow[0][i].end(), FF[rowIdx].begin() + ffPrevSize, [samplesOffset](std::pair<uint32_t, float> elem) { elem.first += samplesOffset; return elem; });
    }

    finalSamples.insert(finalSamples.end(), samples1.begin(), samples1.end());
    for (uint32_t i = 0; i < samples1.size(); ++i) {
      virtualPatchVoxelId.push_back(voxelId);
    }
  }

  normals.resize(finalSamples.size());
  emission.resize(finalSamples.size());
  colors.resize(finalSamples.size());
  for (uint32_t i = 0; i < finalSamples.size(); ++i) {
    normals[i] = finalSamples[i].normal;
    colors[i] = finalSamples[i].color;
    emission[i] = finalSamples[i].emission;
  }

  fout.write(reinterpret_cast<const char*>(&DataConfig::FF_VERSION), sizeof(DataConfig::FF_VERSION));
  const uint32_t outSize = static_cast<uint32_t>(finalSamples.size());
  fout.write(reinterpret_cast<const char*>(&outSize), sizeof(outSize));
  for (uint32_t i = 0; i < outSize; ++i) {
    uint32_t rowSize = static_cast<uint32_t>(FF[i].size());
    fout.write(reinterpret_cast<char*>(&rowSize), sizeof(rowSize));
    for (auto it = FF[i].begin(); it != FF[i].end(); ++it) {
      uint32_t idx = it->first;
      float value = it->second;
      fout.write(reinterpret_cast<char*>(&idx), sizeof(idx));
      fout.write(reinterpret_cast<char*>(&value), sizeof(value));
    }
  }
  fout.close();
}

std::vector<float3> RD_FFIntegrator::ComputeLightingClassic(const std::vector<float3>& emission, const std::vector<float3>& colors) {
  const int quadsCount = static_cast<int>(colors.size());
  std::vector<float3> incident;
  std::vector<float3> lighting(emission.size(), float3(0, 0, 0));
  std::vector<float3> excident(emission);
  for (int iter = 0; iter < 5; ++iter) {
    incident.assign(quadsCount, float3());
#pragma omp parallel for num_threads(7)
    for (int i = 0; i < quadsCount; ++i) {
      for (uint32_t j = 0; j < FF[i].size(); ++j)
        incident[i] += FF[i][j].second * excident[FF[i][j].first];
    }
#pragma omp parallel for num_threads(7)
    for (int i = 0; i < quadsCount; ++i) {
      excident[i] = incident[i] * colors[i];
      if (iter != 0 || true) {
        lighting[i] += incident[i];
      }
    }
  }
  return lighting;
}

std::vector<float3> RD_FFIntegrator::ComputeLightingRandom(const std::vector<float3>& emission, const std::vector<float3>& colors) {
  const int quadsCount = static_cast<int>(colors.size());
  std::vector<std::vector<float>> probabilityTable(quadsCount);
  std::vector<std::vector<std::pair<int, int>>> probabilityChoise(quadsCount);
#pragma omp parallel for num_threads(7)
  for (int i = 0; i < FF.size(); ++i) {
    std::vector<std::pair<int, float>> row;
    float sum = 0;
    for (int j = 0; j < FF[i].size(); ++j) {
      row.emplace_back(FF[i][j]);
      sum += FF[i][j].second;
    }
    if (sum > 1) {
      for (uint32_t j = 0; j < row.size(); ++j) {
        row[j].second /= sum;
      }
      sum = 1;
    }
    if (sum < 1) {
      row.emplace_back(-1, 1.f - sum);
    }
    for (uint32_t j = 0; j < row.size(); ++j) {
      row[j].second *= row.size();
    }
    for (uint32_t j = 1, je = static_cast<uint32_t>(row.size()); j < je; ++j) {
      uint32_t lessIdx = 0, moreIdx = 0;
      for (; lessIdx < row.size() && row[lessIdx].second > 1; ++lessIdx);
      for (; moreIdx < row.size() && row[moreIdx].second <= 1; ++moreIdx);
      if (lessIdx == row.size())
        lessIdx = static_cast<uint32_t>(row.size()) - 1;
      if (moreIdx == row.size())
        moreIdx = (lessIdx + 1) % row.size();
      probabilityTable[i].push_back(row[lessIdx].second);
      probabilityChoise[i].emplace_back(row[lessIdx].first, row[moreIdx].first);
      row[moreIdx].second = row[moreIdx].second + row[lessIdx].second - 1;
      std::swap(row[lessIdx], row.back());
      row.resize(row.size() - 1);
    }
    probabilityTable[i].push_back(1);
    probabilityChoise[i].emplace_back(row[0].first, -1);
  }

  std::vector<float3> bounce1(emission);
  std::vector<float3> bounce2(quadsCount, float3(0, 0, 0));
  std::vector<float3> lighting(emission);
  uint32_t iters = 300;
  uint32_t bounces = 60;
  const float invMaxRand = 1.f / RAND_MAX;
  for (uint32_t bounce = 0; bounce < bounces; ++bounce) {
    for (uint32_t it = 0; it < iters; ++it) {
      #pragma omp parallel for num_threads(7)
      for (int i = 0; i < FF.size(); ++i) {
        const uint32_t sampleCell = rand() % probabilityTable[i].size();
        const float sampleLevel = rand() * invMaxRand;
        const int sampleId = sampleLevel <= probabilityTable[i][sampleCell] ? probabilityChoise[i][sampleCell].first : probabilityChoise[i][sampleCell].second;
        if (sampleId < 0) {
          continue;
        }
        bounce2[i] += bounce1[sampleId];
      }
    }

    bounce1 = emission;
#pragma omp parallel for num_threads(7)
    for (int i = 0; i < FF.size(); ++i) {
      lighting[i] = lerp(bounce2[i] * colors[i] / static_cast<float>(iters), lighting[i], 0.96065750035f);
      bounce1[i] += lighting[i];
      bounce2[i] = float3(0, 0, 0);
    }
  }

#pragma omp parallel for num_threads(7)
  for (int i = 0; i < FF.size(); ++i) {
    lighting[i] += emission[i];
  }
  return lighting;
}

void reorder_triangles_by_voxels(
  std::vector<uint32_t>& voxel_ids,
  std::vector<float3>& colors,
  std::vector<float3>& emission,
  std::vector<RD_FFIntegrator::Triangle>& triangles
) {
  std::vector<uint32_t> remapIndices(voxel_ids.size());
  std::vector<uint32_t> reorderedIndices(voxel_ids.size());
  for (uint32_t i = 0; i < voxel_ids.size(); ++i) {
    reorderedIndices[i] = i;
  }
  std::sort(reorderedIndices.begin(), reorderedIndices.end(), [&voxel_ids](uint32_t a, uint32_t b) {return voxel_ids[a] < voxel_ids[b]; });
  for (uint32_t i = 0; i < remapIndices.size(); ++i) {
    remapIndices[reorderedIndices[i]] = i;
  }
  
  uint32_t sortedPrefix = 0;
  while (sortedPrefix != remapIndices.size()) {
    if (remapIndices[sortedPrefix] == sortedPrefix) {
      sortedPrefix++;
      continue;
    }
    uint32_t targetIdx = sortedPrefix + 1;
    while (remapIndices[targetIdx] != sortedPrefix) {
      targetIdx++;
    }
    std::swap(remapIndices[sortedPrefix], remapIndices[targetIdx]);
    std::swap(voxel_ids[sortedPrefix], voxel_ids[targetIdx]);
    std::swap(colors[sortedPrefix], colors[targetIdx]);
    std::swap(emission[sortedPrefix], emission[targetIdx]);
    std::swap(triangles[sortedPrefix], triangles[targetIdx]);
  }
}

/*
Matrix scheme:
| A B |
| C D |
*/


std::vector<std::vector<uint32_t>> merge_triangles(
  std::vector<uint32_t>& voxels,
  std::vector<float3>& colors,
  std::vector<float3>& emission,
  std::vector<float>& squares,
  std::vector<std::vector<std::pair<int, float>>>& ff
) {
  std::vector<std::vector<uint32_t>> clusters;
  std::vector<bool> used(voxels.size(), false);
  for (uint32_t i = 0; i < voxels.size(); ++i) {
    if (used[i]) {
      continue;
    }
    clusters.resize(clusters.size() + 1);
    for (uint32_t j = i; j < voxels.size(); ++j) {
      if (voxels[i] == voxels[j]) {
        clusters.back().push_back(j);
        used[j] = true;
      }
    }
  }
  std::vector<std::vector<float>> compress1(clusters.size());
  std::vector<float3> comEmission(clusters.size()), compColors(clusters.size());
  for (uint32_t i = 0; i < clusters.size(); ++i) {
    compress1[i].assign(clusters.size(), 0);
    std::vector<float> newRow(colors.size(), 0);
    float weight = 0;
    for (uint32_t j = 0; j < clusters[i].size(); ++j) {
      comEmission[i] += emission[clusters[i][j]] * squares[clusters[i][j]];
      compColors[i] += colors[clusters[i][j]] * squares[clusters[i][j]];
      const std::vector<std::pair<int, float>>& row = ff[clusters[i][j]];
      for (const auto& value : row) {
        newRow[value.first] += value.second * squares[clusters[i][j]];
      }
      weight += squares[clusters[i][j]];
    }
    for (uint32_t j = 0; j < clusters.size(); ++j) {
      float w2 = 0;
      for (uint32_t h = 0; h < clusters[j].size(); ++h) {
        compress1[i][j] += newRow[clusters[j][h]] * squares[clusters[j][h]];
        w2 += squares[clusters[j][h]];
      }
      compress1[i][j] /= weight * w2;
    }
    comEmission[i] /= weight;
    compColors[i] /= weight;
  }

  emission = comEmission;
  colors = compColors;

  ff.clear();
  ff.resize(compress1.size());
  for (uint32_t i = 0; i < compress1.size(); ++i) {
    for (uint32_t j = 0; j < compress1[i].size(); ++j) {
      if (compress1[i][j] > 1e-9) {
        ff[i].emplace_back(j, compress1[i][j]);
      }
    }
  }

  return clusters;
}

static int tessFactor;

bool RD_FFIntegrator::UpdateImage(int32_t a_texId, int32_t w, int32_t h, int32_t bpp, const void* a_data, pugi::xml_node a_texNode) {
  textures[a_texId].w = w;
  textures[a_texId].h = h;
  textures[a_texId].bpp = bpp;
  textures[a_texId].data.resize(w * h * bpp);
  memcpy(textures[a_texId].data.data(), a_data, w * h * bpp);
  return true;
}

void merge_ff_by_polygon(std::vector<uint32_t>& voxels,
  std::vector<float3>& colors,
  std::vector<float3>& emission,
  std::vector<float3>& normals,
  std::vector<float>& squares,
  std::vector<std::vector<std::pair<int, float>>>& ff
) {
  //Timer t("Merge ff");

  std::unordered_map<uint32_t, std::vector<uint32_t>> perVoxelSplits;
  for (uint32_t i = 0; i < voxels.size(); ++i) {
    perVoxelSplits[voxels[i]].push_back(i);
  }
  std::vector<std::vector<uint32_t>> patchesToMerge;
  std::vector<uint32_t> remapList(voxels.size(), 0);
  for (auto inVoxelIds : perVoxelSplits) {
    const uint32_t splitId = static_cast<uint32_t>(patchesToMerge.size());
    for (uint32_t i = 0; i < inVoxelIds.second.size(); ++i) {
      const uint32_t idx = inVoxelIds.second[i];
      uint32_t j = splitId;
      for (; j < patchesToMerge.size(); ++j) {
        const uint32_t refIdx = patchesToMerge[j][0];
        if (lengthSquare(normals[idx] - normals[refIdx]) < 1e-5f && lengthSquare(colors[idx] - colors[refIdx]) < 1e-5f) {
          patchesToMerge[j].push_back(idx);
          break;
        }
      }
      if (j == patchesToMerge.size()) {
        patchesToMerge.push_back({ idx });
      }
      remapList[idx] = j;
    }
  }

  std::vector<std::vector<std::pair<int, float>>> compressedFF(patchesToMerge.size());
  std::vector<uint32_t> compressedVoxels(patchesToMerge.size());
  std::vector<float3> compressedColors(patchesToMerge.size());
  std::vector<float3> compressedEmission(patchesToMerge.size());
  std::vector<float3> compressedNormals(patchesToMerge.size());
  std::vector<float> compressedSqaures(patchesToMerge.size(), 0);
#pragma omp parallel for num_threads(7)
  for (int i = 0; i < patchesToMerge.size(); ++i) {
    std::unordered_map<int, float> newFFRow;
    compressedVoxels[i] = voxels[patchesToMerge[i][0]];
    for (uint32_t idx : patchesToMerge[i]) {
      const float square = squares[idx];
      compressedSqaures[i] += square;
      compressedColors[i] += colors[idx];
      compressedEmission[i] += emission[idx];
      compressedNormals[i] += normals[idx];
      for (const auto &ffItem : ff[idx]) {
        newFFRow[remapList[ffItem.first]] += ffItem.second * square;
      }
    }
    compressedColors[i] /= static_cast<float>(patchesToMerge[i].size());
    compressedEmission[i] /= static_cast<float>(patchesToMerge[i].size());
    compressedNormals[i] /= static_cast<float>(patchesToMerge[i].size());
    compressedNormals[i] = normalize(compressedNormals[i]);

    compressedFF[i] = std::vector<std::pair<int, float>>(newFFRow.begin(), newFFRow.end());
    const float invSumSquare = 1 / compressedSqaures[i];
    for (auto& ffItem : compressedFF[i]) {
      ffItem.second *= invSumSquare;
    }
    std::sort(compressedFF[i].begin(), compressedFF[i].end());
  }

  voxels = std::move(compressedVoxels);
  colors = std::move(compressedColors);
  emission = std::move(compressedEmission);
  normals = std::move(compressedNormals);
  squares = std::move(compressedSqaures);
  ff = std::move(compressedFF);
}

float lengthSq(const float3& a) {
  return pow2(a.x) + pow2(a.y) + pow2(a.z);
}

void merge_ff_by_ff_values(std::vector<uint32_t>& voxels,
  std::vector<float3>& colors,
  std::vector<float3>& emissions,
  std::vector<float3>& normals,
  std::vector<float>& squares,
  std::vector<std::vector<std::pair<int, float>>>& ff)
{
  const uint32_t lastVoxId = *std::max_element(voxels.begin(), voxels.end()) + 1;
  std::vector<std::vector<uint32_t>> clustering;
  std::ofstream outVoxelsData("tmp_ff_dist.bin", std::ios::binary | std::ios::out);
  outVoxelsData.write(reinterpret_cast<const char*>(&lastVoxId), sizeof(lastVoxId));
  std::vector<std::vector<uint32_t>> patchesInVoxels(lastVoxId);
  for (uint32_t voxelId = 0; voxelId < lastVoxId; ++voxelId) {
    for (uint32_t patchId = 0; patchId < voxels.size(); ++patchId) {
      if (voxels[patchId] == voxelId) {
        patchesInVoxels[voxelId].push_back(patchId);
      }
    }
    float maxSquare = 0.0;
    for (uint32_t patchId = 0; patchId < patchesInVoxels[voxelId].size(); ++patchId) {
      maxSquare = max(maxSquare, squares[patchesInVoxels[voxelId][patchId]]);
    }
    const float filterThreshold = maxSquare / 100.0f;
    auto newEnd = std::remove_if(patchesInVoxels[voxelId].begin(), patchesInVoxels[voxelId].end(), [&squares, filterThreshold](uint32_t patchId) {
      return squares[patchId] < filterThreshold;
    });
    patchesInVoxels[voxelId].erase(newEnd, patchesInVoxels[voxelId].end());
    if (patchesInVoxels[voxelId].size() < 4) {
      for (uint32_t patchId = 0; patchId < patchesInVoxels[voxelId].size(); ++patchId) {
        std::vector<uint32_t> cluster(1, patchesInVoxels[voxelId][patchId]);
        clustering.push_back(cluster);
      }
      const uint32_t ZERO = 0;
      outVoxelsData.write(reinterpret_cast<const char*>(&ZERO), sizeof(ZERO));
      continue;
    }
    std::vector<std::vector<float>> ffDistancesBetweenPatches(patchesInVoxels[voxelId].size());
    for (uint32_t patchA = 0; patchA < patchesInVoxels[voxelId].size(); ++patchA) {
      ffDistancesBetweenPatches[patchA].assign(patchesInVoxels[voxelId].size(), 0);
    }
    for (uint32_t patchA = 0; patchA < patchesInVoxels[voxelId].size(); ++patchA) {
      for (uint32_t patchB = patchA + 1; patchB < patchesInVoxels[voxelId].size(); ++patchB) {
        uint32_t ffAIdx = 0;
        uint32_t ffBIdx = 0;
        while (ffAIdx < ff[patchA].size() && ffBIdx < ff[patchB].size()) {
          if (ff[patchA][ffAIdx].first < ff[patchB][ffBIdx].first) {
            ffDistancesBetweenPatches[patchA][patchB] += lengthSq(ff[patchA][ffAIdx].second * colors[patchA] / squares[ff[patchA][ffAIdx].first]);
            ffAIdx++;
          } else if (ff[patchA][ffAIdx].first > ff[patchB][ffBIdx].first) {
            ffDistancesBetweenPatches[patchA][patchB] += lengthSq(ff[patchB][ffBIdx].second * colors[patchB] / squares[ff[patchB][ffBIdx].first]);
            ffBIdx++;
          } else {
            ffDistancesBetweenPatches[patchA][patchB] += lengthSq(ff[patchA][ffAIdx].second * colors[patchA] / squares[ff[patchA][ffAIdx].first] - ff[patchB][ffBIdx].second * colors[patchB] / squares[ff[patchB][ffBIdx].first]);
            ffAIdx++;
            ffBIdx++;
          }
        }
        while (ffAIdx < ff[patchA].size()) {
          ffDistancesBetweenPatches[patchA][patchB] += lengthSq(ff[patchA][ffAIdx].second * colors[patchA] / squares[ff[patchA][ffAIdx].first]);
          ffAIdx++;
        }
        while (ffBIdx < ff[patchB].size()) {
          ffDistancesBetweenPatches[patchA][patchB] += lengthSq(ff[patchB][ffBIdx].second * colors[patchB] / squares[ff[patchB][ffBIdx].first]);
          ffBIdx++;
        }
        //ffDistancesBetweenPatches[patchA][patchB] = lengthSq(normals[patchA] * squares[patchA] - normals[patchB] * squares[patchB]);
        //ffDistancesBetweenPatches[patchA][patchB] = -ffDistancesBetweenPatches[patchA][patchB];
        ffDistancesBetweenPatches[patchB][patchA] = ffDistancesBetweenPatches[patchA][patchB];
      }
    }
    {
      const uint32_t patchesInVoxelsCount = static_cast<uint32_t>(patchesInVoxels[voxelId].size());
      outVoxelsData.write(reinterpret_cast<const char*>(&patchesInVoxelsCount), sizeof(patchesInVoxelsCount));
      for (const auto& ffDistLine : ffDistancesBetweenPatches) {
        for (const float dist : ffDistLine) {
          outVoxelsData.write(reinterpret_cast<const char*>(&dist), sizeof(dist));
        }
      }
    }
  }
  outVoxelsData.close();
  int code = system("python merge_voxels.py");
  assert(code == 0);
  std::ifstream inClusteringData("tmp_clust_data.bin", std::ios::binary | std::ios::in);
  for (uint32_t voxelId = 0; voxelId < lastVoxId; ++voxelId) {
    if (patchesInVoxels[voxelId].size() < 4) {
      continue;
    }
    std::array<std::vector<uint32_t>, 3> inVoxelClustering;
    {
      for (auto& cluster : inVoxelClustering) {
        uint32_t ptInCluster;
        inClusteringData.read(reinterpret_cast<char*>(&ptInCluster), sizeof(ptInCluster));
        cluster.resize(ptInCluster);
        for (auto& patchId : cluster) {
          uint32_t localPatchId;
          inClusteringData.read(reinterpret_cast<char*>(&localPatchId), sizeof(localPatchId));
          patchId = patchesInVoxels[voxelId][localPatchId];
        }
      }
    }
    clustering.insert(clustering.end(), inVoxelClustering.begin(), inVoxelClustering.end());
  }
  inClusteringData.close();

  std::vector<uint32_t> mergedVoxels(clustering.size());
  std::vector<float3> mergedColors(clustering.size());
  std::vector<float3> mergedEmission(clustering.size());
  std::vector<float3> mergedNormals(clustering.size());
  std::vector<float> mergedSquares(clustering.size());
  std::vector<std::vector<std::pair<int, float>>> mergedFF(clustering.size());
  std::vector<uint32_t> invClustering(voxels.size());
  for (uint32_t clusterId = 0; clusterId < clustering.size(); ++clusterId) {
    for (auto patchId : clustering[clusterId]) {
      invClustering[patchId] = clusterId;
    }
  }
  for (uint32_t clusterId = 0; clusterId < clustering.size(); ++clusterId) {
    mergedVoxels[clusterId] = voxels[clustering[clusterId][0]];
    float3 color = float3(0, 0, 0);
    float3 emission = float3(0, 0, 0);
    float3 normal = float3(0, 0, 0);
    float square = 0;
    std::vector<float> ffRow(clustering.size(), 0.0f);
    for (uint32_t patchId = 0; patchId < clustering[clusterId].size(); ++patchId) {
      const uint32_t initialId = clustering[clusterId][patchId];
      color += colors[initialId] * squares[initialId];
      emission += emissions[initialId] * squares[initialId];
      normal += normals[initialId] * squares[initialId];
      square += squares[initialId];
      for (auto [idx, value] : ff[initialId]) {
        ffRow[invClustering[idx]] += value * squares[initialId];
      }
    }
    mergedColors[clusterId] = color / square;
    mergedEmission[clusterId] = emission / square;
    mergedNormals[clusterId] = normalize(normal / square);
    mergedSquares[clusterId] = square;
    for (uint32_t newPatchId = 0; newPatchId < ffRow.size(); ++newPatchId) {
      const float ffValue = ffRow[newPatchId] / square;
      if (ffValue > 1e-9f) {
        mergedFF[clusterId].emplace_back(newPatchId, ffValue);
      }
    }
  }
  voxels = mergedVoxels;
  colors = mergedColors;
  emissions = mergedEmission;
  normals = mergedNormals;
  squares = mergedSquares;
  ff = mergedFF;
}

static float2 computeHSfromRGB(const float3& color_rgb) {
  const float minCh = min(color_rgb.x, min(color_rgb.y, color_rgb.z));
  const float maxCh = max(color_rgb.x, max(color_rgb.y, color_rgb.z));
  if (minCh == maxCh) {
    return float2(0, 0);
  }
  const float scale = 1.f / (maxCh - minCh);
  const float S = maxCh == 0 ? 0 : 1 - minCh / maxCh;
  if (maxCh == color_rgb.x) {
    if (color_rgb.y >= color_rgb.z) {
      return float2((color_rgb.y - color_rgb.z) * scale / 6.f, S);
    }
    else {
      return float2((color_rgb.y - color_rgb.z) * scale / 6.f + 1.f, S);
    }
  }
  else if (maxCh == color_rgb.y) {
    return float2((color_rgb.z - color_rgb.x) * scale / 6.f + 1.0f / 3.0f, S);
  }
  return float2((color_rgb.x - color_rgb.y) * scale / 6.f + 2.0f / 3.0f, S);
}

static float2 getAnglesForNormal(const float3& normal) {
  return float2(std::acos(normal.z), atan2(normal.y, normal.x));
}

void RD_FFIntegrator::EndScene() {
  const uint32_t trianglesCount = static_cast<uint32_t>(instanceTriangles.size());

  struct Material
  {
    std::optional<float3> diffuse, emission;
    std::optional<uint32_t> texRef;
  };

  std::vector<Material> mats;
  for (const auto& materialColor : matColors) {
    if (mats.size() <= materialColor.first) {
      mats.resize(materialColor.first + 1);
    }
    mats[materialColor.first].diffuse = materialColor.second;
  }
  for (const auto& materialColor : matEmission) {
    if (mats.size() <= materialColor.first) {
      mats.resize(materialColor.first + 1);
    }
    mats[materialColor.first].emission = materialColor.second;
  }
  for (const auto tex : matTexture) {
    if (mats.size() <= tex.first) {
      mats.resize(tex.first + 1);
    }
    mats[tex.first].texRef = tex.second;
  }

  std::map<int, TexData> locTex = textures;

  std::vector<float3> colors, emission;
  for (uint32_t i = 0; i < trianglesCount; ++i) {
    colors.push_back(matColors[instanceTriangles[i].materialId]);
    emission.push_back(matEmission[instanceTriangles[i].materialId]);
  }

  std::ofstream colorsOut(DataConfig::get().getBinFilePath(L"Colors.bin"), std::ios::binary | std::ios::out);
  colorsOut.write(reinterpret_cast<const char*>(&trianglesCount), sizeof(trianglesCount));
  for (uint32_t i = 0; i < trianglesCount; ++i) {
    colorsOut.write(reinterpret_cast<char*>(&colors[i]), sizeof(colors[i]));
  }
  colorsOut.close();

  std::vector<uint32_t> voxels(trianglesCount);
  std::ifstream voxelsIn(DataConfig::get().getBinFilePath(L"VoxelIds.bin"), std::ios::binary | std::ios::in);
  uint32_t voxTriCount;
  uint3 gridSize;
  float voxelSize;
  voxelsIn.read(reinterpret_cast<char*>(&gridSize), sizeof(gridSize));
  voxelsIn.read(reinterpret_cast<char*>(&voxTriCount), sizeof(voxTriCount));
  voxelsIn.read(reinterpret_cast<char*>(&voxelSize), sizeof(voxelSize));
  for (uint32_t i = 0; i < trianglesCount; ++i) {
    voxelsIn.read(reinterpret_cast<char*>(&voxels[i]), sizeof(voxels[i]));
  }
  voxelsIn.close();

  std::vector<float> squares(trianglesCount);
  for (uint32_t i = 0; i < trianglesCount; ++i) {
    squares[i] = triangle_square(instanceTriangles[i]);
  }

  std::cout << "Max vox: " << *std::max_element(voxels.begin(), voxels.end()) << std::endl;

  for (int i = static_cast<int>(squares.size()) - 1; i >= 0; --i) {
    if (squares[i] < 1e-9) {
      squares.erase(squares.begin() + i);
      instanceTriangles.erase(instanceTriangles.begin() + i);
      colors.erase(colors.begin() + i);
      voxels.erase(voxels.begin() + i);
      emission.erase(emission.begin() + i);
    }
  }

  std::cout << "Max vox: " << *std::max_element(voxels.begin(), voxels.end()) << std::endl;

  std::cout << trianglesCount << " triangles" << std::endl;
  //ComputeFF(static_cast<uint32_t>(instanceTriangles.size()), instanceTriangles, squares);

  std::vector<float3> normals(instanceTriangles.size());
  for (uint32_t i = 0; i < normals.size(); ++i) {
    normals[i] = to_float3(instanceTriangles[i].normal.value()[0]);
  }
  std::vector<float3> voxelsCenters(gridSize.x * gridSize.y * gridSize.z);
  float3 bmin = float3(1e9, 1e9, 1e9);
  float3 bmax = float3(-1e9, -1e9, -1e9);
  for (uint32_t i = 0; i < instanceTriangles.size(); ++i) {
    for (uint32_t j = 0; j < instanceTriangles[i].points.size(); ++j) {
      bmin.x = min(instanceTriangles[i].points[j].x, bmin.x);
      bmin.y = min(instanceTriangles[i].points[j].y, bmin.y);
      bmin.z = min(instanceTriangles[i].points[j].z, bmin.z);
      bmax.x = max(instanceTriangles[i].points[j].x, bmax.x);
      bmax.y = max(instanceTriangles[i].points[j].y, bmax.y);
      bmax.z = max(instanceTriangles[i].points[j].z, bmax.z);
    }
  }

  for (uint32_t z = 0, idx = 0; z < gridSize.z; ++z) {
    for (uint32_t y = 0; y < gridSize.y; ++y) {
      for (uint32_t x = 0; x < gridSize.x; ++x, ++idx) {
        voxelsCenters[idx] = bmin + (float3(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)) + float3(0.5, 0.5, 0.5)) * voxelSize;
      }
    }
  }

  ComputeFF_voxelized(instanceTriangles, squares, voxelsCenters, voxelSize, colors, emission, normals);

  //merge_ff_by_polygon(voxels, colors, emission, normals, squares, FF);
  //merge_ff_by_ff_values(voxels, colors, emission, normals, squares, FF);

  auto lighting = ComputeLightingClassic(emission, colors);
  //auto lighting = ComputeLightingRandom(emission, colors);
  const uint32_t PATCHES_IN_VOXEL = 3;
  std::vector<std::array<float3, PATCHES_IN_VOXEL>> voxelsGridColors(gridSize.x * gridSize.y * gridSize.z);
  std::vector<std::array<float, PATCHES_IN_VOXEL>> inVoxelSquares(gridSize.x* gridSize.y* gridSize.z);
  std::vector<std::array<float3, PATCHES_IN_VOXEL>> inVoxelNormals(gridSize.x* gridSize.y* gridSize.z);
  std::vector<std::array<float4, PATCHES_IN_VOXEL>> inVoxelWeightMatrix(gridSize.x* gridSize.y* gridSize.z);

  for (uint32_t i = 0; i < lighting.size(); ++i) {
    const uint32_t voxelId = virtualPatchVoxelId[i];
    float square = 1.0f;// squares[i];
    float3 light = lighting[i];
    float3 normal = normals[i];
    for (uint32_t j = 0; j < PATCHES_IN_VOXEL; ++j) {
      if (square > inVoxelSquares[voxelId][j]) {
        std::swap(square, inVoxelSquares[voxelId][j]);
        std::swap(light, voxelsGridColors[voxelId][j]);
        std::swap(normal, inVoxelNormals[voxelId][j]);
      }
    }
  }
  for (uint32_t i = 0; i < voxelsGridColors.size(); ++i) {
    for (uint32_t j = 1; j < PATCHES_IN_VOXEL; ++j) {
      if (inVoxelSquares[i][j] < 1e-9f) {
        voxelsGridColors[i][j] = voxelsGridColors[i][0];
      }
    }
    
    std::array<bool, 3> basisVecExist = {true, true, true};
    const float EPS = 1e-5f;
    basisVecExist[0] = dot(inVoxelNormals[i][0], inVoxelNormals[i][0]) > EPS;
    if (basisVecExist[0]) {
      if (std::abs(dot(inVoxelNormals[i][0], inVoxelNormals[i][1])) > 1 - EPS || length(inVoxelNormals[i][1]) < EPS) {
        inVoxelNormals[i][1] = (std::abs(inVoxelNormals[i][0].x) > 0.5) ? float3(0, 1, 0) : float3(1, 0, 0);
        float3 basisVec3 = cross(inVoxelNormals[i][0], inVoxelNormals[i][1]);
        inVoxelNormals[i][1] = normalize(cross(basisVec3, inVoxelNormals[i][0]));
        basisVecExist[1] = false;
      }
      if (std::abs(dot(inVoxelNormals[i][2], normalize(cross(inVoxelNormals[i][1], inVoxelNormals[i][0])))) < EPS || length(inVoxelNormals[i][2]) < EPS) {
        inVoxelNormals[i][2] = normalize(cross(inVoxelNormals[i][0], inVoxelNormals[i][1]));
        basisVecExist[2] = false;
      }
    } else {
      basisVecExist[1] = basisVecExist[2] = false;
    }

    std::array<float2, 3> normalsIn2d;
    for (uint32_t j = 0; j < normalsIn2d.size(); ++j) {
      normalsIn2d[j] = getAnglesForNormal(inVoxelNormals[i][j]);
    }

    float4x4 normalsMat4x4;
    normalsMat4x4.row[0] = float4(normalsIn2d[0].x, normalsIn2d[1].x, normalsIn2d[2].x, 0);
    normalsMat4x4.row[1] = float4(normalsIn2d[0].y, normalsIn2d[1].y, normalsIn2d[2].y, 0);
    normalsMat4x4.row[2] = float4(1, 1, 1, 0);
    normalsMat4x4.row[3] = float4(0, 0, 0, 1);
    float4x4 weightMat4x4 = inverse4x4(normalsMat4x4);
    for (uint32_t j = 0; j < 3; ++j) {
      //inVoxelWeightMatrix[i][j] = weightMat4x4.row[j];
      //inVoxelWeightMatrix[i][j].w = basisVecExist[j] ? 1.0f : 0.0f;
      inVoxelWeightMatrix[i][j] = to_float4(inVoxelNormals[i][j], basisVecExist[j] ? 1.0f : 0.0f);
    }
  }
  std::ofstream VoxelGridLightingOut(DataConfig::get().getBinFilePath(L"VoxelGridLighting.bin"), std::ios::binary | std::ios::out);
  VoxelGridLightingOut.write(reinterpret_cast<const char*>(&gridSize), sizeof(gridSize));
  VoxelGridLightingOut.write(reinterpret_cast<const char*>(&bmin), sizeof(bmin));
  float3 allignedBmax = bmin + voxelSize * float3((float)gridSize.x, (float)gridSize.y, (float)gridSize.z);
  VoxelGridLightingOut.write(reinterpret_cast<const char*>(&allignedBmax), sizeof(allignedBmax));
  for (uint32_t i = 0; i < voxelsGridColors.size(); ++i) {
    VoxelGridLightingOut.write(reinterpret_cast<char*>(&voxelsGridColors[i]), sizeof(voxelsGridColors[i]));
  }
  for (uint32_t i = 0; i < voxelsGridColors.size(); ++i) {
    VoxelGridLightingOut.write(reinterpret_cast<char*>(&inVoxelWeightMatrix[i]), sizeof(inVoxelWeightMatrix[i]));
  }
  VoxelGridLightingOut.close();
}

using DrawFuncType = void (*)();
using InitFuncType = void (*)();

void window_main_ff_integrator(const std::wstring& a_libPath, const std::wstring& scene_name) {
  hrErrorCallerPlace(L"Init");

  HRInitInfo initInfo;
  initInfo.vbSize = 1024 * 1024 * 128;
  initInfo.sortMaterialIndices = false;
  hrSceneLibraryOpen(a_libPath.c_str(), HR_OPEN_EXISTING, initInfo);

  HRSceneLibraryInfo scnInfo = hrSceneLibraryInfo();

  if (scnInfo.camerasNum == 0) // create some default camera
    camRef = hrCameraCreate(L"defaultCam");

  renderRef.id = 0;
  camRef.id = 0;
  scnRef.id = 0;

  hrCameraOpen(camRef, HR_OPEN_READ_ONLY);
  auto camNode = hrCameraParamNode(camRef);
  const std::array<std::wstring, 12> camParamNames = {
    L"position",
    L"fov",
    L"nearClipPlane",
    L"farClipPlane",
    L"enable_dof",
    L"dof_lens_radius",
    L"up",
    L"look_at",
    L"tiltRotX",
    L"tiltRotY",
    L"tiltShiftX",
    L"tiltShiftY",
  };
  for (int i = 0; i < camParamNames.size(); ++i) {
    camParams[camParamNames[i]] = camNode.child(camParamNames[i].c_str()).text().as_string();
  }
  hrCameraClose(camRef);

  renderRef = hrRenderCreate(L"ff_integrator");

  auto pList = hrRenderGetDeviceList(renderRef);

  hrRenderEnableDevice(renderRef, 0, true);

  hrCommit(scnRef, renderRef);
  hrSceneLibraryClose();
}
