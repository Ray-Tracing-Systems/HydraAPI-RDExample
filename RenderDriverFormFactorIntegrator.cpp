// This is a personal academic project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

#include <array>
#include <chrono>

#include <cassert>

#include <embree3/rtcore.h>
#include <omp.h>

#include <LiteMath.h>

#include "RenderDriverFormFactorIntegrator.h"

using namespace HydraLiteMath;

static HRRenderRef  renderRef;
static HRCameraRef  camRef;
static HRSceneInstRef scnRef;
static std::unordered_map<std::wstring, std::wstring> camParams;
static bool recompute_ff = false;

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
  return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

float2 hammersley2d(uint32_t i, uint32_t N) {
  return float2(float(i) / float(N), radicalInverse_VdC(i));
}

static std::vector<std::vector<Sample>> gen_samples(const std::vector<RD_FFIntegrator::Triangle>& triangles) {
  const int PER_AXIS_COUNT = 2;
  const int SAMPLES_COUNT = PER_AXIS_COUNT * PER_AXIS_COUNT;
  std::vector<float3> randomValues(SAMPLES_COUNT);
  for (int i = 0; i < SAMPLES_COUNT; ++i) {
    //randomValues[i] = hammersley2d(i / PER_AXIS_COUNT, i % PER_AXIS_COUNT);
    randomValues[i].x = static_cast<float>(rand()) / RAND_MAX;
    randomValues[i].y = static_cast<float>(rand()) / RAND_MAX;
    randomValues[i].z = static_cast<float>(rand()) / RAND_MAX;
    float sum = dot(randomValues[i], float3(1, 1, 1));
    randomValues[i] /= sum;
  }
  std::vector<std::vector<Sample>> samples;
  for (int i = 0; i < triangles.size(); ++i) {
    std::vector<Sample> sm(SAMPLES_COUNT);
    for (int j = 0; j < SAMPLES_COUNT; ++j) {
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
  return length(cross(p1 - p0, p2 - p0)) * 0.5;
}



static std::vector<RD_FFIntegrator::Quad> merge_quads(const std::vector<RD_FFIntegrator::Quad>& quads, int &tessFactor) {
  std::vector<RD_FFIntegrator::Quad> result;
  std::vector<uint32_t> quadEnds(1, 0);
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
  quadEnds.push_back(quads.size());
  for (int i = 1; i < quadEnds.size(); ++i) {
    tessFactor = std::sqrt(quadEnds[i] - quadEnds[i - 1]) + 0.5f;
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

    Device = rtcNewDevice(""); CHECK_EMBREE
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
    rtcSetSceneFlags(Scene, RTC_SCENE_FLAG_ROBUST); CHECK_EMBREE
    rtcInitIntersectContext(&IntersectionContext); CHECK_EMBREE
    IntersectionContext.flags = RTC_INTERSECT_CONTEXT_FLAG_COHERENT;
    IntersectionContext.instID[0] = 0;

    RTCBounds bounds;
    rtcGetSceneBounds(Scene, &bounds);
  }

  uint16_t traceRays(const std::vector<Sample>& samples1, const std::vector<Sample>& samples2) {
    uint16_t result = 0;
    const int PACKET_SIZE = 4;
    RTCRay16 raysPacket;
    for (int i = 0, target_id = 0; i < samples1.size(); ++i) {
      const Sample& s1 = samples1[i];
      for (int j = 0; j < samples2.size(); ++j, ++target_id) {
        const Sample& s2 = samples2[j];
        const float BIAS = 1e-5f;
        float3 dir = s2.pos - s1.pos;
        raysPacket.org_x[target_id] = s1.pos.x;
        raysPacket.org_y[target_id] = s1.pos.y;
        raysPacket.org_z[target_id] = s1.pos.z;
        raysPacket.tnear[target_id] = BIAS;
        raysPacket.dir_x[target_id] = dir.x;
        raysPacket.dir_y[target_id] = dir.y;
        raysPacket.dir_z[target_id] = dir.z;
        raysPacket.tfar[target_id] = 1.f - BIAS;
        raysPacket.id[target_id] = 0;
        raysPacket.mask[target_id] = 0;
        raysPacket.time[target_id] = 0;
      }
    }

    const int validMask = ~0u;
    rtcOccluded16(&validMask, Scene, &IntersectionContext, &raysPacket); CHECK_EMBREE

    for (uint32_t k = 0; k < PACKET_SIZE * PACKET_SIZE; ++k) {
      if (std::isinf(raysPacket.tfar[k])) {
        result |= 1u << k;
      }
    }

    return result;
  }

  ~EmbreeTracer() {
    rtcReleaseScene(Scene);
    rtcReleaseDevice(Device);
  }
};

void RD_FFIntegrator::ComputeFF(uint32_t quadsCount, std::vector<RD_FFIntegrator::Triangle>& triangles)
{
  FF.resize(quadsCount);

  std::stringstream ss;
  ss << "FF" << quadsCount;
  std::ifstream fin(ss.str(), std::ios::binary);
  if (!recompute_ff && fin.is_open() && false) {
    uint32_t countFromFile = 0;
    fin.read(reinterpret_cast<char*>(&countFromFile), sizeof(countFromFile));
    assert(countFromFile == quadsCount);
    for (int i = 0; i < quadsCount; ++i) {
      uint32_t rowSize;
      fin.read(reinterpret_cast<char*>(&rowSize), sizeof(rowSize));
      for (int j = 0; j < rowSize; ++j) {
        uint32_t idx;
        fin.read(reinterpret_cast<char*>(&idx), sizeof(idx));
        fin.read(reinterpret_cast<char*>(&FF[i][idx]), sizeof(FF[i][idx]));
      }
    }
    fin.close();
    return;
  }
  std::ofstream fout(ss.str(), std::ios::binary);

  std::vector<std::vector<Sample>> samples = gen_samples(instanceTriangles);
  std::vector<float> squares(quadsCount);
  for (int i = 0; i < quadsCount; ++i) {
    squares[i] = triangle_square(instanceTriangles[i]);
  }

  EmbreeTracer tracer(triangles);

  omp_set_dynamic(0);

  for (int i = 0; i < quadsCount - 1; ++i) {
    if (100 * i / quadsCount < 100 * (i + 1) / quadsCount) {
      std::cout << 100 * i / quadsCount << "% finished" << std::endl;
    }
    const std::vector<Sample>& samples1 = samples[i];

#pragma omp parallel for num_threads(7)
    for (int j = i + 1; j < quadsCount; ++j) {
      const std::vector<Sample>& samples2 = samples[j];
      uint16_t occluded = tracer.traceRays(samples1, samples2);
      float value = 0;
      int samplesCount = 0;
      for (int k = 0; k < samples1.size(); ++k) {
        const Sample& sample1 = samples1[k];
        for (int h = 0; h < samples2.size(); ++h, occluded >>= 1) {
          const Sample& sample2 = samples2[h];
          if ((occluded & 1) != 0) {
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
        FF[i][j] = value * squares[j];
        FF[j][i] = value * squares[i];
      }
    }
  }

  fout.write(reinterpret_cast<const char*>(&quadsCount), sizeof(quadsCount));
  for (int i = 0; i < quadsCount; ++i) {
    uint32_t rowSize = FF[i].size();
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
  const int quadsCount = colors.size();
  std::vector<float3> incident;
  std::vector<float3> lighting(emission);
  std::vector<float3> excident(emission);
  for (int iter = 0; iter < 7; ++iter) {
    incident.assign(quadsCount, float3());
#pragma omp parallel for num_threads(7)
    for (int i = 0; i < quadsCount; ++i) {
      for (int j = 0; j < quadsCount; ++j) {
        if (FF[i].find(j) != FF[i].end()) {
          incident[i] += FF[i][j] * excident[j];
        }
      }
    }
#pragma omp parallel for num_threads(7)
    for (int i = 0; i < quadsCount; ++i) {
      excident[i] = incident[i] * colors[i];
      lighting[i] += excident[i];
    }
  }
  return lighting;
}

//std::vector<float3> RD_FFIntegrator::ComputeLightingRandom(const std::vector<float3>& emission, const std::vector<float3>& colors) {
//  const int quadsCount = colors.size();
//  std::vector<std::vector<float>> probabilityTable(quadsCount);
//  std::vector<std::vector<std::pair<int, int>>> probabilityChoise(quadsCount);
//  for (int i = 0; i < FF.size(); ++i) {
//    std::vector<std::pair<float, int>> row;
//    float sum = 0;
//    for (int j = 0; j < FF[i].size(); ++j) {
//      if (FF[i][j] < 1e-9) {
//        continue;
//      }
//      row.emplace_back(j, FF[i][j]);
//      sum += FF[i][j];
//    }
//    if (sum < 1) {
//      row.emplace_back(-1, 1.f - sum);
//    }
//    sum = 1.f;
//    const float columnSum = sum / row.size();
//    while (!row.empty()) {
//      auto [minVal, maxVal] = std::minmax_element(row.begin(), row.end());
//      probabilityChoise[i].emplace_back(minVal->second, maxVal->second);
//      probabilityTable[i].push_back(minVal->first / columnSum);
//      maxVal->first -= columnSum - minVal->first;
//      row.erase(minVal);
//    }
//  }
//}

static int tessFactor;
static bool noInterpolation;

void RD_FFIntegrator::EndScene() {
  const int trianglesCount = instanceTriangles.size();

  std::vector<float3> colors, emission;
  for (int i = 0; i < trianglesCount; ++i) {
    colors.push_back(matColors[instanceTriangles[i].materialId]);
    emission.push_back(matEmission[instanceTriangles[i].materialId]);
  }

  std::stringstream ss;
  ss << "Colors" << trianglesCount;
  std::ofstream colorsOut(ss.str(), std::ios::binary | std::ios::out);
  colorsOut.write(reinterpret_cast<const char*>(&trianglesCount), sizeof(trianglesCount));
  for (int i = 0; i < trianglesCount; ++i) {
    colorsOut.write(reinterpret_cast<char*>(&colors[i]), sizeof(colors[i]));
  }
  colorsOut.close();

  std::cout << trianglesCount << " triangles" << std::endl;
  ComputeFF(trianglesCount, instanceTriangles);

  auto lighting = ComputeLightingClassic(emission, colors);

  auto triangles = instanceTriangles;

  hrSceneLibraryOpen(L"GI_res/scene.xml", HR_WRITE_DISCARD);
  HRSceneInstRef scene = hrSceneCreate(L"Scene");
  hrSceneOpen(scene, HR_WRITE_DISCARD);

  auto cam = hrCameraCreate(L"Camera1");
  hrCameraOpen(cam, HR_WRITE_DISCARD);
  auto proxyCamNode = hrCameraParamNode(cam);
  for (auto it = camParams.begin(); it != camParams.end(); ++it) {
    proxyCamNode.append_child(it->first.c_str()).text().set(it->second.c_str());
  }
  hrCameraClose(cam);

  for (int i = 0; i < lighting.size(); ++i) {
    std::wstringstream ss;
    ss << "Mat" << i;
    auto matRef = hrMaterialCreate(ss.str().c_str());
    hrMaterialOpen(matRef, HR_WRITE_DISCARD);
    auto material = hrMaterialParamNode(matRef);
    auto diffuse = material.append_child();
    diffuse.set_name(L"diffuse");
    diffuse.append_attribute(L"brdf_type").set_value(L"lambert");
    auto color = diffuse.append_child();
    color.set_name(L"color");
    ss = std::wstringstream();
    ss << lighting[i].x << " " << lighting[i].y << ' ' << lighting[i].z;
    color.append_attribute(L"val").set_value(ss.str().c_str());
    hrMaterialClose(matRef);
  }
  std::vector<float4> points;
  std::vector<float4> normals;
  std::vector<float4> tangents;
  std::vector<float2> texCoords;
  std::vector<uint32_t> materials;
  std::vector<uint32_t> indices;

  for (int i = 0; i < lighting.size(); ++i) {
    materials.push_back(i);
    for (int j = 0; j < 3; ++j) {
      indices.push_back(indices.size());
    }
    points.insert(points.end(), triangles[i].points.begin(), triangles[i].points.end());
    normals.insert(normals.end(), triangles[i].normal.value().begin(), triangles[i].normal.value().end());
    tangents.insert(tangents.end(), triangles[i].tangent.value().begin(), triangles[i].tangent.value().end());
    texCoords.insert(texCoords.end(), triangles[i].texCoords.begin(), triangles[i].texCoords.end());
  }
  
  auto mesh = hrMeshCreate(L"Whole scene");
  hrMeshOpen(mesh, HR_TRIANGLE_IND12, HR_WRITE_DISCARD);
  hrMeshPrimitiveAttribPointer1i(mesh, L"mind", reinterpret_cast<int*>(materials.data()));
  hrMeshVertexAttribPointer4f(mesh, L"positions", reinterpret_cast<float*>(points.data()));
  hrMeshVertexAttribPointer4f(mesh, L"normals", reinterpret_cast<float*>(normals.data()));
  hrMeshVertexAttribPointer4f(mesh, L"tangent", reinterpret_cast<float*>(tangents.data()));
  hrMeshVertexAttribPointer2f(mesh, L"texcoord", reinterpret_cast<float*>(texCoords.data()));
  hrMeshAppendTriangles3(mesh, indices.size(), reinterpret_cast<int*>(indices.data()));
  hrMeshClose(mesh);

  std::array<float, 16> matrix = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };
  hrMeshInstance(scene, mesh, matrix.data());

  hrSceneClose(scene);
  hrFlush(scene);// , render, cam);
}

using DrawFuncType = void (*)();
using InitFuncType = void (*)();

void window_main_ff_integrator(const wchar_t* a_libPath, const wchar_t* a_renderName, bool recomputeFF, bool no_interpolation) {
  recompute_ff = recomputeFF;
  noInterpolation = no_interpolation;
  hrErrorCallerPlace(L"Init");

  hrSceneLibraryOpen(a_libPath, HR_OPEN_EXISTING);

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

  renderRef = hrRenderCreate(a_renderName);

  auto pList = hrRenderGetDeviceList(renderRef);

  hrRenderEnableDevice(renderRef, 0, true);

  hrCommit(scnRef, renderRef);
}
