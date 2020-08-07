#pragma once

#include <map>
#include <optional>

#include "HydraRenderDriverAPI.h"


using ScenePolygon = std::vector<uint32_t>;

struct Scene
{
  std::vector<HydraLiteMath::float4> positions;
  std::vector<HydraLiteMath::float4> normals;
  std::vector<HydraLiteMath::float2> texCoords;
  std::vector<HydraLiteMath::float4> tangents;

  std::vector<uint32_t> materials;
  std::vector<uint32_t> voxelIds;
  std::vector<ScenePolygon> polygons;

  void addScene(const Scene& scene);
  uint32_t addMidVertex(uint32_t idx1, uint32_t idx2, float t);
  void compress();
};

struct RD_VoxelTessellator : public IHRRenderDriver {
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
  virtual void GetRenderDriverName(std::wstring& name) { name = L"voxelTessellator"; };

  std::vector<int>  allRemapLists;
  std::vector<HydraLiteMath::int2> tableOffsetsAndSize;

  std::map<int, Scene> meshes;

  Scene fullScene;

  std::map<int, HydraLiteMath::float3> matColors;
  std::map<int, HydraLiteMath::float3> matEmission;
  std::map<int, uint32_t> matTexture;
  struct TexData {
    uint32_t w, h, bpp;
    std::vector<char> data;
  };
  std::map<int, TexData> textures;
};
