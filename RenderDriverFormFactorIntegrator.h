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
  bool UpdateImage(int32_t a_texId, int32_t w, int32_t h, int32_t bpp, const void* a_data, pugi::xml_node a_texNode) override { return false; }
  bool UpdateMaterial(int32_t a_matId, pugi::xml_node a_materialNode) override;
  bool UpdateLight(int32_t a_lightId, pugi::xml_node a_lightNode) override { return false; }
  bool UpdateMesh(int32_t a_meshId, pugi::xml_node a_meshNode, const HRMeshDriverInput& a_input, const HRBatchInfo* a_batchList, int32_t listSize) override;
  bool UpdateImageFromFile(int32_t a_texId, const wchar_t* a_fileName, pugi::xml_node a_texNode) override { return false; }
  bool UpdateMeshFromFile(int32_t a_meshId, pugi::xml_node a_meshNode, const wchar_t* a_fileName) override { return false; }
  bool UpdateCamera(pugi::xml_node a_camNode) override { return false; }
  bool UpdateSettings(pugi::xml_node a_settingsNode) override { return false; }
  void BeginScene(pugi::xml_node a_sceneNode) override;
  void ComputeFF(const int& quadsCount, std::vector<RD_FFIntegrator::Quad>& bigQuads);
  void EndScene() override;
  void InstanceMeshes(int32_t a_mesh_id, const float* a_matrix, int32_t a_instNum, const int* a_lightInstId, const int* a_remapId, const int* a_realInstId) override;
  void InstanceLights(int32_t a_light_id, const float* a_matrix, pugi::xml_node* a_custAttrArray, int32_t a_instNum, int32_t a_lightGroupId) override {}
  void Draw() override {}
  HRRenderUpdateInfo HaveUpdateNow(int a_maxRaysPerPixel) override;
  void GetFrameBufferHDR(int32_t w, int32_t h, float* a_out, const wchar_t* a_layerName) override {}
  void GetFrameBufferLDR(int32_t w, int32_t h, int32_t* a_out) override {}
  void GetGBufferLine(int32_t a_lineNumber, HRGBufferPixel* a_lineData, int32_t a_startX, int32_t a_endX, const std::unordered_set<int32_t>& a_shadowCatchers) override {}
  const HRRenderDeviceInfoListElem* DeviceList() const override { return nullptr; }
  bool EnableDevice(int32_t id, bool a_enable) override { return false; }

  std::vector<int>  allRemapLists;
  std::vector<HydraLiteMath::int2> tableOffsetsAndSize;

  std::map<int, std::vector<Quad>> meshQuads;
  std::vector<Quad> instanceQuads;
  std::vector<std::vector<float>> FF;

  std::vector<Quad> bigQuads;
  std::map<int, HydraLiteMath::float3> matColors;
  std::map<int, HydraLiteMath::float3> matEmission;
};