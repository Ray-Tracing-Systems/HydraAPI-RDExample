// This is a personal academic project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

#include <array>

#include <embree3/rtcore.h>
#include <omp.h>

#include <LiteMath.h>

#include "RenderDriverFormFactorIntegrator.h"

using namespace HydraLiteMath;

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

static std::vector<RD_FFIntegrator::Quad> gen_quads(const HRMeshDriverInput& a_input) {

  std::vector<RD_FFIntegrator::Triangle> triangles;
  const uint32_t* indices = reinterpret_cast<const uint32_t*>(a_input.indices);
  const float4* positions = reinterpret_cast<const float4*>(a_input.pos4f);
  const float4* normals = reinterpret_cast<const float4*>(a_input.norm4f);
  const float4* tangents = reinterpret_cast<const float4*>(a_input.tan4f);
  const float2* texCoords = reinterpret_cast<const float2*>(a_input.texcoord2f);
  const uint32_t* materials = reinterpret_cast<const uint32_t*>(a_input.triMatIndices);

  const bool hasTangent = a_input.tan4f;
  const bool hasNormals = a_input.norm4f;

  for (int i = 0; i < a_input.triNum * 3; i += 3) {
    RD_FFIntegrator::Triangle tri;
    if (hasTangent) {
      tri.tangent.emplace(std::array<float4, 3>());
    }
    if (hasNormals) {
      tri.normal.emplace(std::array<float4, 3>());
    }
    for (int j = 0; j < 3; ++j) {
      const uint32_t index = indices[i + j];
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

  std::vector<int> cornerPoints(triangles.size(), 0);
  for (int i = 0; i < cornerPoints.size(); ++i) {
    float4 v1 = triangles[i].points[1] - triangles[i].points[0];
    float4 v2 = triangles[i].points[2] - triangles[i].points[0];
    if (dot3(v1, v2) < 1e-5) {
      cornerPoints[i] = 0;
      continue;
    }
    v1 = triangles[i].points[0] - triangles[i].points[1];
    v2 = triangles[i].points[2] - triangles[i].points[1];
    if (dot3(v1, v2) < 1e-5) {
      cornerPoints[i] = 1;
      continue;
    }
    v1 = triangles[i].points[1] - triangles[i].points[2];
    v2 = triangles[i].points[0] - triangles[i].points[2];
    if (dot3(v1, v2) < 1e-5) {
      cornerPoints[i] = 2;
      continue;
    }
    throw "Triangle without 90 deg angle!";
  }

  std::vector<RD_FFIntegrator::Quad> quads;
  for (int i = 0; i < triangles.size(); ++i) {
    for (int j = i + 1; j < triangles.size(); ++j) {
      if (triangles[i].materialId != triangles[j].materialId) {
        continue;
      }
      const int t1Next = (cornerPoints[i] + 1) % 3;
      const int t1Prev = (cornerPoints[i] + 2) % 3;
      const int t2Next = (cornerPoints[j] + 2) % 3;
      const int t2Prev = (cornerPoints[j] + 1) % 3;
      const bool pos1Match = length3(triangles[i].points[t1Next] - triangles[j].points[t2Prev]) < 1e-3;
      const bool pos2Match = length3(triangles[i].points[t1Prev] - triangles[j].points[t2Next]) < 1e-3;
      const bool normal1Match = length3(triangles[i].normal.value()[t1Next] - triangles[j].normal.value()[t2Prev]) < 1e-3;
      const bool normal2Match = length3(triangles[i].normal.value()[t1Prev] - triangles[j].normal.value()[t2Next]) < 1e-3;
      if (pos1Match && pos2Match && normal1Match && normal2Match) {
        RD_FFIntegrator::Quad q;
        q.materialId = triangles[i].materialId;
        std::array<std::pair<int, int>, 4> indices = { std::make_pair(i, cornerPoints[i]), std::make_pair(i, t1Next),
          std::make_pair(j, cornerPoints[j]), std::make_pair(i, t1Prev) };
        if (hasNormals) {
          q.normal.emplace(std::array<float4, 4>());
        }
        if (hasTangent) {
          q.tangent.emplace(std::array<float4, 4>());
        }
        for (int k = 0; k < 4; ++k) {
          q.points[k] = triangles[indices[k].first].points[indices[k].second];
          q.texCoords[k] = triangles[indices[k].first].texCoords[indices[k].second];
          if (hasNormals) {
            q.normal.value()[k] = triangles[indices[k].first].normal.value()[indices[k].second];
          }
          if (hasTangent) {
            q.tangent.value()[k] = triangles[indices[k].first].tangent.value()[indices[k].second];
          }
        }
        quads.push_back(q);
      }
    }
  }
  return quads;
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
  meshQuads[a_meshId] = gen_quads(a_input);
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
    const auto& mesh = meshQuads[a_mesh_id];
    for (const Quad& q : mesh) {
      Quad instancedQuad;
      if (q.normal.has_value()) {
        instancedQuad.normal = std::array<float4, 4>();
      }
      if (q.tangent.has_value()) {
        instancedQuad.tangent = std::array<float4, 4>();
      }
      for (int j = 0; j < instancedQuad.points.size(); ++j) {
        instancedQuad.points[j] = mul(models[i], q.points[j]);
        if (q.normal.has_value()) {
          instancedQuad.normal.value()[j] = mul(models[i], q.normal.value()[j]);
        }
        if (q.tangent.has_value()) {
          instancedQuad.tangent.value()[j] = mul(models[i], q.tangent.value()[j]);
        }
        instancedQuad.texCoords[j] = q.texCoords[j];
      }
      instancedQuad.materialId = q.materialId;
      if (a_remapId[i] != -1) {
        const int jBegin = tableOffsetsAndSize[a_remapId[i]].x;
        const int jEnd = tableOffsetsAndSize[a_remapId[i]].x + tableOffsetsAndSize[a_remapId[i]].y;
        for (int j = jBegin; j < jEnd; j += 2) {
          if (allRemapLists[j] == instancedQuad.materialId) {
            instancedQuad.materialId = allRemapLists[j + 1];
            break;
          }
        }
      }
      instanceQuads.push_back(instancedQuad);
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

static std::vector<std::vector<Sample>> gen_samples(const std::vector<RD_FFIntegrator::Quad>& quads) {
  const int SAMPLES_COUNT = 16;
  std::vector<float2> randomValues(SAMPLES_COUNT);
  for (int i = 0; i < SAMPLES_COUNT; ++i) {
    randomValues[i].x = static_cast<float>(rand()) / RAND_MAX;
    randomValues[i].y = static_cast<float>(rand()) / RAND_MAX;
  }
  std::vector<std::vector<Sample>> samples;
  for (int i = 0; i < quads.size(); ++i) {
    std::vector<Sample> sm(SAMPLES_COUNT);
    for (int j = 0; j < SAMPLES_COUNT; ++j) {
      float4 pos = lerpSquare(quads[i].points[0], quads[i].points[1], quads[i].points[2], quads[i].points[3], randomValues[j].x, randomValues[j].y);
      const auto& normArray = quads[i].normal.value();
      float4 normal = lerpSquare(normArray[0], normArray[1], normArray[2], normArray[3], randomValues[j].x, randomValues[j].y);
      sm[j].pos = float3(pos.x, pos.y, pos.z);
      sm[j].normal = float3(normal.x, normal.y, normal.z);
    }
    samples.push_back(sm);
  }
  return samples;
}

static float quad_square(const RD_FFIntegrator::Quad& quad) {
  return length3(quad.points[0] - quad.points[1]) * length3(quad.points[0] - quad.points[3]);
}

static std::vector<RD_FFIntegrator::Quad> merge_quads(const std::vector<RD_FFIntegrator::Quad>& quads) {
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
    const int tessFactor = std::sqrt(quadEnds[i] - quadEnds[i - 1]) + 0.5f;
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
  EmbreeTracer(const std::vector<RD_FFIntegrator::Quad>& quads) {
    std::vector<float3> points(quads.size() * 4);
    std::vector<uint32_t> indices(quads.size() * 6);
    for (int i = 0; i < quads.size(); ++i) {
      for (int j = 0; j < 4; ++j) {
        points[i * 4 + j] = make_float3(quads[i].points[j]);
      }
      indices[i * 4 + 0] = i * 4 + 0;
      indices[i * 4 + 1] = i * 4 + 1;
      indices[i * 4 + 2] = i * 4 + 2;
      indices[i * 4 + 3] = i * 4 + 0;
      indices[i * 4 + 4] = i * 4 + 2;
      indices[i * 4 + 5] = i * 4 + 3;
    }

    Device = rtcNewDevice(""); CHECK_EMBREE
    Scene = rtcNewScene(Device); CHECK_EMBREE
    RTCGeometry geometry = rtcNewGeometry(Device, RTC_GEOMETRY_TYPE_TRIANGLE); CHECK_EMBREE
    RTCBuffer indicesBuffer = rtcNewSharedBuffer(Device, reinterpret_cast<void*>(indices.data()), indices.size() * sizeof(indices[0])); CHECK_EMBREE
    RTCBuffer pointsBuffer = rtcNewSharedBuffer(Device, reinterpret_cast<void*>(points.data()), points.size() * sizeof(points[0])); CHECK_EMBREE

    rtcSetGeometryBuffer(geometry, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, indicesBuffer, 0, 0, indices.size() / 3); CHECK_EMBREE
    rtcSetGeometryBuffer(geometry, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, pointsBuffer, 0, sizeof(float), points.size()); CHECK_EMBREE
    rtcCommitGeometry(geometry); CHECK_EMBREE
    rtcAttachGeometry(Scene, geometry); CHECK_EMBREE
    rtcReleaseGeometry(geometry);
    rtcCommitScene(Scene); CHECK_EMBREE
    rtcInitIntersectContext(&IntersectionContext); CHECK_EMBREE
    IntersectionContext.flags = RTC_INTERSECT_CONTEXT_FLAG_COHERENT;
    IntersectionContext.instID[0] = 0;
  }

  std::vector<bool> traceRays(const std::vector<Sample>& samples1, std::vector<Sample>& samples2) {
    std::vector<bool> result;
    const int PACKET_SIZE = 16;
    for (int i = 0; i < samples1.size(); ++i) {
      for (int j = 0; j < samples2.size(); j += PACKET_SIZE) {
        RTCRay16 raysPacket;
        for (int k = 0; k < PACKET_SIZE; ++k) {
          raysPacket.org_x[k] = samples1[i].pos.x;
          raysPacket.org_y[k] = samples1[i].pos.y;
          raysPacket.org_z[k] = samples1[i].pos.z;
          raysPacket.tnear[k] = 1e-5f;
          raysPacket.dir_x[k] = samples2[j + k].pos.x - samples1[i].pos.x;
          raysPacket.dir_y[k] = samples2[j + k].pos.y - samples1[i].pos.y;
          raysPacket.dir_z[k] = samples2[j + k].pos.z - samples1[i].pos.z;
          raysPacket.tfar[k] = length(samples2[j + k].pos - samples1[i].pos) - 1e-5f;
          raysPacket.id[k] = 0;
          raysPacket.mask[k] = 0;
          raysPacket.time[k] = 0;
        }
        const int validMask = ~0u;
        rtcOccluded16(&validMask, Scene, &IntersectionContext, &raysPacket); CHECK_EMBREE

        for (int k = 0; k < PACKET_SIZE; ++k) {
          result.push_back(!std::isinf(raysPacket.tfar[k]));
        }
      }
    }

    return result;
  }

  ~EmbreeTracer() {
    rtcReleaseScene(Scene);
    rtcReleaseDevice(Device);
  }
};

void RD_FFIntegrator::EndScene() {
  const int quadsCount = instanceQuads.size();;
  std::vector<std::vector<Sample>> samples = gen_samples(instanceQuads);
  std::vector<float> squares(quadsCount);
  for (int i = 0; i < quadsCount; ++i) {
    squares[i] = quad_square(instanceQuads[i]);
  }

  std::vector<float3> colors, emission;
  for (int i = 0; i < quadsCount; ++i) {
    colors.push_back(matColors[instanceQuads[i].materialId]);
    emission.push_back(matEmission[instanceQuads[i].materialId]);
  }

  std::vector<RD_FFIntegrator::Quad> bigQuads = merge_quads(instanceQuads);
  EmbreeTracer tracer(bigQuads);

  omp_set_dynamic(0);

  FF.assign(quadsCount, std::vector<float>(quadsCount, 0));
#pragma omp parallel for num_threads(7)
  for (int i = 0; i < quadsCount; ++i) {
    for (int j = i + 1; j < quadsCount; ++j) {
      std::vector<bool> occluded = tracer.traceRays(samples[i], samples[j]);
      float value = 0;
      int samplesCount = 0;
      for (int k = 0; k < samples[i].size(); ++k) {
        for (int h = 0; h < samples[j].size(); ++h) {
          if (!occluded[k * samples[j].size() + h]) {
            samplesCount++;
            continue;
          }
          const float3 r = samples[i][k].pos - samples[j][h].pos;
          const float l = length(r);
          if (l < 1e-1) {
            continue;
          }
          const float3 toSample1 = r / l;
          const float3 toSample2 = -toSample1;
          const float theta1 = max(dot(samples[i][k].normal, toSample2), 0.f);
          const float theta2 = max(dot(samples[j][h].normal, toSample1), 0.f);
          value += theta1 * theta2 / l / l;
          samplesCount++;
        }
      }
      value /= samplesCount * 3.14f;
      FF[i][j] = value * squares[i];
      FF[j][i] = value * squares[j];
    }
  }

  std::vector<float3> incident;
  std::vector<float3> lighting(emission);
  std::vector<float3> excident(emission);
  for (int iter = 0; iter < 4; ++iter) {
    incident.assign(quadsCount, float3());
    for (int i = 0; i < quadsCount; ++i) {
      for (int j = 0; j < quadsCount; ++j) {
        incident[i] += FF[i][j] * excident[j];
      }
    }
    for (int i = 0; i < quadsCount; ++i) {
      excident[i] = incident[i] * colors[i];
      lighting[i] += excident[i];
    }
  }
}

using DrawFuncType = void (*)();
using InitFuncType = void (*)();

static HRRenderRef  renderRef;

static HRSceneInstRef scnRef;

void window_main_ff_integrator(const wchar_t* a_libPath, const wchar_t* a_renderName, InitFuncType a_pInitFunc, DrawFuncType a_pDrawFunc) {
  hrErrorCallerPlace(L"Init");

  hrSceneLibraryOpen(a_libPath, HR_OPEN_EXISTING);

  HRSceneLibraryInfo scnInfo = hrSceneLibraryInfo();

  renderRef.id = 0;
  scnRef.id = 0;

  renderRef = hrRenderCreate(a_renderName);

  auto pList = hrRenderGetDeviceList(renderRef);

  hrRenderEnableDevice(renderRef, 0, true);

  hrCommit(scnRef, renderRef);
}
