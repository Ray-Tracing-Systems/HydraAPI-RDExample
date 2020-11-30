#pragma once

#include <map>
#include <optional>

#include "HydraRenderDriverAPI.h"


struct RD_FFIntegrator : public IHRRenderDriver {
  template<size_t N>
  struct Polygon {
    static const size_t POINTS_COUNT = N;
    std::array<HydraLiteMath::float4, N> points;
    std::array<HydraLiteMath::float2, N> texCoords;
    std::optional<std::array<HydraLiteMath::float4, N>> normal;
    std::optional<std::array<HydraLiteMath::float4, N>> tangent;
    uint32_t materialId = 0;
  };

  using Triangle = Polygon<3>;
  using Quad = Polygon<4>;

  void ClearAll() override {}
  HRDriverAllocInfo AllocAll(HRDriverAllocInfo a_info) override { return a_info; }
  bool UpdateImage(int32_t a_texId, int32_t w, int32_t h, int32_t bpp, const void* a_data, pugi::xml_node a_texNode) override;
  bool UpdateMaterial(int32_t a_matId, pugi::xml_node a_materialNode) override;
  bool UpdateLight(int32_t, pugi::xml_node) override { return false; }
  bool UpdateMesh(int32_t a_meshId, pugi::xml_node a_meshNode, const HRMeshDriverInput& a_input, const HRBatchInfo* a_batchList, int32_t listSize) override;
  bool UpdateImageFromFile(int32_t, const wchar_t*, pugi::xml_node) override { return false; }
  bool UpdateMeshFromFile(int32_t, pugi::xml_node, const wchar_t*) override { return false; }
  bool UpdateCamera(pugi::xml_node) override { return false; }
  bool UpdateSettings(pugi::xml_node) override { return false; }
  void BeginScene(pugi::xml_node a_sceneNode) override;
  void ComputeFF(uint32_t quadsCount, std::vector<RD_FFIntegrator::Triangle>& triangles, const std::vector<float>& squares);
  void ComputeFF_voxelized(uint32_t quadsCount, std::vector<RD_FFIntegrator::Triangle>& triangles, const std::vector<float>& squares);
  std::vector<HydraLiteMath::float3> RD_FFIntegrator::ComputeLightingClassic(const std::vector<HydraLiteMath::float3>& emission, const std::vector<HydraLiteMath::float3>& colors);
  std::vector<HydraLiteMath::float3> RD_FFIntegrator::ComputeLightingRandom(const std::vector<HydraLiteMath::float3>& emission, const std::vector<HydraLiteMath::float3>& colors);
  void EndScene() override;
  void InstanceMeshes(int32_t a_mesh_id, const float* a_matrix, int32_t a_instNum, const int* a_lightInstId, const int* a_remapId, const int* a_realInstId) override;
  void InstanceLights(int32_t, const float*, pugi::xml_node*, int32_t, int32_t) override {}
  void Draw() override {}
  HRRenderUpdateInfo HaveUpdateNow(int a_maxRaysPerPixel) override;
  void GetFrameBufferHDR(int32_t, int32_t, float*, const wchar_t*) override {}
  void GetFrameBufferLDR(int32_t, int32_t, int32_t*) override {}
  void GetGBufferLine(int32_t, HRGBufferPixel*, int32_t, int32_t, const std::unordered_set<int32_t>&) override {}
  const HRRenderDeviceInfoListElem* DeviceList() const override { return nullptr; }
  bool EnableDevice(int32_t, bool) override { return false; }

  std::vector<int>  allRemapLists;
  std::vector<HydraLiteMath::int2> tableOffsetsAndSize;

  std::map<int, std::vector<Triangle>> meshTriangles;
  std::vector<Triangle> instanceTriangles;
  std::vector<std::vector<std::pair<int, float>>> FF;

  std::vector<Quad> bigQuads;
  std::map<int, HydraLiteMath::float3> matColors;
  std::map<int, HydraLiteMath::float3> matEmission;
  std::map<int, uint32_t> matTexture;
  struct TexData {
    uint32_t w, h, bpp;
    std::vector<char> data;
  };
  std::map<int, TexData> textures;
};
