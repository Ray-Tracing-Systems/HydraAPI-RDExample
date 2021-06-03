// This is a personal academic project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <set>

#include <LiteMath.h>

#include "RenderDriverVoxelTessellator.h"
#include "dataConfig.h"

using namespace HydraLiteMath;

static HRRenderRef  renderRef;
static HRCameraRef  camRef;
static HRSceneInstRef scnRef;
static std::unordered_map<std::wstring, std::wstring> camParams;

IHRRenderDriver* CreateVoxelTessellator_RenderDriver()
{
  return new RD_VoxelTessellator;
}

HRRenderUpdateInfo RD_VoxelTessellator::HaveUpdateNow(int a_maxRaysPerPixel)
{
  HRRenderUpdateInfo res;
  res.finalUpdate = true;
  res.haveUpdateFB = true;
  res.progress = 100.0f;
  return res;
}

static Scene gen_scene(const HRMeshDriverInput& a_input) {

  Scene triangles;
  const uint32_t* indices = reinterpret_cast<const uint32_t*>(a_input.indices);
  const float4* positions = reinterpret_cast<const float4*>(a_input.pos4f);
  const float4* normals = reinterpret_cast<const float4*>(a_input.norm4f);
  const float4* tangents = reinterpret_cast<const float4*>(a_input.tan4f);
  const float2* texCoords = reinterpret_cast<const float2*>(a_input.texcoord2f);
  const uint32_t* materials = reinterpret_cast<const uint32_t*>(a_input.triMatIndices);

  const bool hasTangent = a_input.tan4f;
  const bool hasNormals = a_input.norm4f;

  triangles.polygons.resize(a_input.triNum);

  triangles.positions = std::vector<float4>(positions, positions + a_input.vertNum);
  triangles.texCoords = std::vector<float2>(texCoords, texCoords + a_input.vertNum);
  if (hasNormals) {
    triangles.normals = std::vector<float4>(normals, normals + a_input.vertNum);
  }
  if (hasTangent) {
    triangles.tangents = std::vector<float4>(tangents, tangents + a_input.vertNum);
  }

  triangles.materials = std::vector<uint32_t>(materials, materials + a_input.triNum);
  for (int i = 0; i < a_input.triNum; ++i) {
    triangles.polygons[i].resize(3);
    for (uint32_t j = 0; j < 3; ++j) {
      triangles.polygons[i][j] = indices[3 * i + j];
    }
  }

  return triangles;
}

void RD_VoxelTessellator::BeginScene(pugi::xml_node a_sceneNode)
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

bool RD_VoxelTessellator::UpdateMesh(int32_t a_meshId, pugi::xml_node a_meshNode, const HRMeshDriverInput& a_input, const HRBatchInfo* a_batchList, int32_t listSize) {
  meshes[a_meshId] = gen_scene(a_input);
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

template<typename T>
float lengthSq(const T& a) {
  return dot(a, a);
}

template<typename T>
static bool isNear(const T& a, const T& b) {
  return lengthSq(a - b) < 1e-10f;
}

static bool isNear(const float4& a, const float4& b) {
  return std::abs(a.x - b.x) < 1e-5f && std::abs(a.y - b.y) < 1e-5f && std::abs(a.z - b.z) < 1e-5f;
}

static bool isNear(const float2& a, const float2& b) {
  return std::abs(a.x - b.x) < 1e-5f && std::abs(a.y - b.y) < 1e-5f;
}

template<typename T>
void append(std::vector<T>& a, const std::vector<T>& b) {
  a.insert(a.end(), b.begin(), b.end());
}

void Scene::addScene(const Scene& scene) {
  const uint32_t indexShift = static_cast<uint32_t>(positions.size());
  append(positions, scene.positions);
  append(normals, scene.normals);
  append(tangents, scene.tangents);
  append(texCoords, scene.texCoords);

  append(materials, scene.materials);
  for (uint32_t i = 0; i < scene.polygons.size(); ++i) {
    ScenePolygon pol = scene.polygons[i];
    for (uint32_t j = 0; j < pol.size(); ++j) {
      pol[j] += indexShift;
    }
    polygons.push_back(pol);
  }
}

void RD_VoxelTessellator::InstanceMeshes(int32_t a_mesh_id, const float* a_matrices, int32_t a_instNum, const int* a_lightInstId, const int* a_remapId, const int* a_realInstId) {
  std::vector<float4x4> models(a_instNum, a_matrices);
  for (int32_t i = 0; i < a_instNum; i++)
  {
    float matrixT2[16];
    mat4x4_transpose(matrixT2, (float*)(a_matrices + i * 16));
    models[i] = float4x4(a_matrices);
  }

  for (int i = 0; i < a_instNum; ++i) {
    Scene mesh = meshes[a_mesh_id];
    for (uint32_t j = 0; j < mesh.positions.size(); ++j) {
      mesh.positions[j] = mul(models[i], mesh.positions[j]);
      mesh.normals[j] = mul(models[i], mesh.normals[j]);
      mesh.tangents[j] = mul(models[i], mesh.tangents[j]);
    }
    if (a_remapId[i] != -1) {
      const int jBegin = tableOffsetsAndSize[a_remapId[i]].x;
      const int jEnd = tableOffsetsAndSize[a_remapId[i]].x + tableOffsetsAndSize[a_remapId[i]].y;
      for (uint32_t k = 0; k < mesh.materials.size(); ++k) {
        for (int j = jBegin; j < jEnd; j += 2) {
          if (allRemapLists[j] == mesh.materials[k]) {
            mesh.materials[k] = allRemapLists[j + 1];
            break;
          }
        }
      }
    }
    fullScene.addScene(mesh);
  }
}

bool RD_VoxelTessellator::UpdateMaterial(int32_t a_matId, pugi::xml_node a_materialNode) {
  pugi::xml_node clrNode = a_materialNode.child(L"diffuse").child(L"color");
  pugi::xml_node texNode = a_materialNode.child(L"diffuse").child(L"texture");
  if (texNode == nullptr) {
    texNode = a_materialNode.child(L"diffuse").child(L"color").child(L"texture");
  }

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

template <typename T>
T lerpSquare(T p1, T p2, T p3, T p4, float x, float y) {
  T x1_pos = lerp(p1, p2, x);
  T x2_pos = lerp(p3, p4, x);
  return lerp(x1_pos, x2_pos, y);
}

static float voxelSize;


struct Vertex
{
  HydraLiteMath::float4 point;
  HydraLiteMath::float2 texCoord;
  std::optional<HydraLiteMath::float4> normal;
  std::optional<HydraLiteMath::float4> tangent;
};

Scene PolygonsToTriangles(const Scene& polygons) {
  Scene triangles;
  triangles.positions = polygons.positions;
  triangles.normals = polygons.normals;
  triangles.tangents = polygons.tangents;
  triangles.texCoords = polygons.texCoords;
  for (uint32_t i = 0; i < polygons.polygons.size(); ++i) {
    for (uint32_t j = 1; j < polygons.polygons[i].size() - 1; ++j) {
      ScenePolygon triangle(3);
      triangle[0] = polygons.polygons[i][0];
      triangle[1] = polygons.polygons[i][j];
      triangle[2] = polygons.polygons[i][j + 1];
      triangles.materials.push_back(polygons.materials[i]);
      triangles.polygons.push_back(triangle);
      triangles.voxelIds.push_back(polygons.voxelIds[i]);
    }
  }
  return triangles;
}

uint32_t Scene::addMidVertex(uint32_t idx1, uint32_t idx2, float t) {
  positions.push_back(lerp(positions[idx1], positions[idx2], t));
  normals.push_back(lerp(normals[idx1], normals[idx2], t));
  tangents.push_back(lerp(tangents[idx1], tangents[idx2], t));
  texCoords.push_back(lerp(texCoords[idx1], texCoords[idx2], t));
  return static_cast<uint32_t>(positions.size()) - 1;
}

#undef max
#undef min

template<typename T>
std::pair<T, T> ordPair(const T& a, const T& b) {
  return std::make_pair(std::max(a, b), std::min(a, b));
}

template<int coord>
void split_triangles_by_plane(Scene& scene, float value, uint32_t slice_size) {
  Scene splitted = scene;
  splitted.materials.clear();
  splitted.polygons.clear();
  splitted.voxelIds.clear();
  std::map<std::pair<uint32_t, uint32_t>, uint32_t> vertexCache;
  for (uint32_t i = 0; i < scene.polygons.size(); ++i) {
    bool onTheLeft = true;
    bool onTheRight = true;
    const ScenePolygon& poly = scene.polygons[i];
    for (uint32_t j = 0; j < poly.size(); ++j) {
      onTheLeft &= scene.positions[poly[j]][coord] < value;
      onTheRight &= scene.positions[poly[j]][coord] >= value;
    }
    if (onTheLeft || onTheRight) {
      splitted.materials.push_back(scene.materials[i]);
      splitted.polygons.push_back(scene.polygons[i]);
      if (onTheLeft) {
        splitted.voxelIds.push_back(scene.voxelIds[i]);
      } else {
        splitted.voxelIds.push_back(scene.voxelIds[i] + slice_size);
      }
      continue;
    }

    ScenePolygon left;
    ScenePolygon right;
    for (uint32_t j = 0; j < poly.size(); ++j) {
      const uint32_t nextIdx = (j + 1) % poly.size();
      const bool isLeft = scene.positions[poly[j]][coord] < value;
      const bool nextLeft = scene.positions[poly[nextIdx]][coord] < value;
      if (isLeft) {
        left.push_back(poly[j]);
      } else {
        right.push_back(poly[j]);
      }
      if (isLeft != nextLeft) {
        const auto cacheIdx = ordPair(poly[j], poly[nextIdx]);
        const auto cacheIt = vertexCache.find(cacheIdx);
        uint32_t newIdx;
        if (cacheIt != vertexCache.end()) {
          newIdx = cacheIt->second;
        } else {
          newIdx = splitted.addMidVertex(poly[j], poly[nextIdx], (value - scene.positions[poly[j]][coord]) / (scene.positions[poly[nextIdx]][coord] - scene.positions[poly[j]][coord]));
          vertexCache[cacheIdx] = newIdx;
        }
        left.push_back(newIdx);
        right.push_back(newIdx);
      }
    }
    if (left.size() >= 3) {
      splitted.polygons.push_back(left);
      splitted.materials.push_back(scene.materials[i]);
      splitted.voxelIds.push_back(scene.voxelIds[i]);
    }
    if (right.size() >= 3) {
      splitted.polygons.push_back(right);
      splitted.materials.push_back(scene.materials[i]);
      splitted.voxelIds.push_back(scene.voxelIds[i] + slice_size);
    }
  }
  scene = splitted;
}

template <int coord>
static void split_triangles_along_axis(Scene& scene, uint32_t &last_slice_size, uint32_t &axis_length) {
  float bmax = -1e9;
  float bmin = 1e9;
  for (const float4& pos : scene.positions) {
    bmax = std::max(bmax, pos[coord]);
    bmin = std::min(bmin, pos[coord]);
  }
  
  axis_length = 1;
  for (float edge = bmin + voxelSize; edge < bmax; edge += voxelSize) {
    split_triangles_by_plane<coord>(scene, edge, last_slice_size);
    axis_length++;
  }
  last_slice_size *= axis_length;
}

float3 max(const float3& a, const float4& b) {
  return float3(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z));
}

float3 min(const float3& a, const float4& b) {
  return float3(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z));
}

void Scene::compress() {
  std::vector<uint32_t> removedIndices;
  std::vector<uint32_t> replacers;
  std::vector<std::pair<uint32_t, uint32_t>> removeAndReplaceIdx;
  std::vector<std::vector<uint32_t>> bins;
  const uint32_t sideSize = 100;

  float3 bmax = float3(-1e9f, -1e9f, -1e9f);
  float3 bmin = float3(1e9f, 1e9f, 1e9f);
  for (const float4& pos : positions) {
    bmax = max(bmax, pos);
    bmin = min(bmin, pos);
  }
  bmax += 1e-5f;
  bmin -= 1e-5f;
  float3 gridScale = 1.f / (bmax - bmin) * sideSize;
  bins.resize(sideSize * sideSize * sideSize);
  for (uint32_t i = 0; i < positions.size(); ++i) {
    float3 voxelId = (make_float3(positions[i]) - bmin) * gridScale;
    uint32_t idx = (uint32_t(voxelId.x) * sideSize + uint32_t(voxelId.y)) * sideSize + uint32_t(voxelId.z);
    bins[idx].push_back(i);
  }

  //TODO: We can blur bins to increase accuracy

  for (uint32_t i = 0; i < bins.size(); ++i) {
    for (int iter1 = static_cast<int>(bins[i].size()) - 1; iter1 > 0; --iter1) {
      const uint32_t idx1 = bins[i][iter1];
      for (int iter2 = iter1 - 1; iter2 >= 0; --iter2) {
        const uint32_t idx2 = bins[i][iter2];
        if (isNear(positions[idx1], positions[idx2]) && isNear(normals[idx1], normals[idx2]) && isNear(tangents[idx1], tangents[idx2]) && isNear(texCoords[idx1], texCoords[idx2])) {
          removeAndReplaceIdx.emplace_back(idx1, idx2);
          break;
        }
      }
    }
  }

  std::sort(removeAndReplaceIdx.rbegin(), removeAndReplaceIdx.rend());
  std::vector<float4> compressedPositions;
  std::vector<float4> compressedNormals;
  std::vector<float4> compressedTangents;
  std::vector<float2> compressedTexCoords;
  int idxToRemove = static_cast<int>(removeAndReplaceIdx.size()) - 1;
  uint32_t uncompressedIdx = 0;
  std::vector<uint32_t> newIndex;
  newIndex.reserve(positions.size());
  while (idxToRemove >= 0) {
    while (idxToRemove >= 0 && uncompressedIdx < std::min(removeAndReplaceIdx[idxToRemove].first, (uint32_t)positions.size())) {
      newIndex.push_back(static_cast<uint32_t>(compressedPositions.size()));
      compressedPositions.push_back(positions[uncompressedIdx]);
      compressedNormals.push_back(normals[uncompressedIdx]);
      compressedTangents.push_back(tangents[uncompressedIdx]);
      compressedTexCoords.push_back(texCoords[uncompressedIdx]);
      uncompressedIdx++;
    }
    newIndex.push_back(newIndex[removeAndReplaceIdx[idxToRemove].second]);
    idxToRemove--;
    uncompressedIdx++;
  }
  while (uncompressedIdx < positions.size()) {
    newIndex.push_back(static_cast<uint32_t>(compressedPositions.size()));
    compressedPositions.push_back(positions[uncompressedIdx]);
    compressedNormals.push_back(normals[uncompressedIdx]);
    compressedTangents.push_back(tangents[uncompressedIdx]);
    compressedTexCoords.push_back(texCoords[uncompressedIdx]);
    uncompressedIdx++;
  }

  positions = compressedPositions;
  normals = compressedNormals;
  tangents = compressedTangents;
  texCoords = compressedTexCoords;

  for (auto& p : polygons) {
    for (auto& idx : p) {
      idx = newIndex[idx];
    }
  }
}

uint64_t pack_edge(uint32_t idx1, uint32_t idx2) {
  return ((uint64_t)min(idx1, idx2) << 32) | max(idx1, idx2);
}

static void GatherPolygons(Scene& scene) {
  std::vector<uint32_t> materials;
  std::vector<float3> normals;
  std::vector<uint32_t> polygons;

  for (int i = static_cast<int>(scene.polygons.size()) - 1; i > 0; --i) {
    bool merged = false;
    for (int j = i - 1; !merged && j >= 0; --j) {
      if (scene.materials[i] != scene.materials[j] || !isNear(scene.normals[scene.polygons[i][0]], scene.normals[scene.polygons[j][0]])) {
        continue;
      }
      for (uint32_t k = 0; !merged && k < scene.polygons[i].size(); ++k) {
        const uint32_t nextIdx1 = (k + 1) % scene.polygons[i].size();
        for (uint32_t h = 0; !merged && h < scene.polygons[j].size(); ++h) {
          const uint32_t nextIdx2 = (h + 1) % scene.polygons[j].size();
          const bool needToReverse = (scene.polygons[i][k] == scene.polygons[j][h] && scene.polygons[i][nextIdx1] == scene.polygons[j][nextIdx2]);
          if (needToReverse || scene.polygons[i][nextIdx1] == scene.polygons[j][h] && scene.polygons[i][k] == scene.polygons[j][nextIdx2]) {
            if (nextIdx1) {
              const std::vector<uint32_t> tail(scene.polygons[i].begin() + nextIdx1, scene.polygons[i].end());
              scene.polygons[i].resize(nextIdx1);
              scene.polygons[i].insert(scene.polygons[i].begin(), tail.begin(), tail.end());
            }
            if (needToReverse) {
              scene.polygons[i] = ScenePolygon(scene.polygons[i].rbegin(), scene.polygons[i].rend());
            }
            scene.polygons[j].insert(scene.polygons[j].begin() + nextIdx2, scene.polygons[i].begin() + 1, scene.polygons[i].end() - 1);
            scene.polygons.erase(scene.polygons.begin() + i);
            scene.materials.erase(scene.materials.begin() + i);
            merged = true;
          }
        }
      }
    }
  }
}

bool RD_VoxelTessellator::UpdateImage(int32_t a_texId, int32_t w, int32_t h, int32_t bpp, const void* a_data, pugi::xml_node a_texNode) {
  textures[a_texId].w = w;
  textures[a_texId].h = h;
  textures[a_texId].bpp = bpp;
  textures[a_texId].data.resize(w * h * bpp);
  memcpy(textures[a_texId].data.data(), a_data, w * h * bpp);
  return true;
}

void RD_VoxelTessellator::EndScene() {
  //TODO: optimize generate triangles

  auto t1 = std::chrono::system_clock::now();

  std::cout << fullScene.materials.size() << " polygons on scene" << std::endl;
  fullScene.compress();
  auto t2 = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = t2 - t1;
  std::cout << "Scene compressed " << elapsed_seconds.count() << std::endl;
  t1 = t2;
  //GatherPolygons(fullScene);
  t2 = std::chrono::system_clock::now();
  elapsed_seconds = t2 - t1;
  std::cout << "Polygons gathered " << elapsed_seconds.count() << std::endl;
  t1 = t2;
  fullScene.voxelIds.resize(fullScene.materials.size());
  for (uint32_t i = 0; i < fullScene.materials.size(); ++i) {
    fullScene.voxelIds[i] = 0;
  }
  t2 = std::chrono::system_clock::now();
  elapsed_seconds = t2 - t1;
  std::cout << "Init voxel indices " << elapsed_seconds.count() << std::endl;
  t1 = t2;
  uint32_t gridSize = 1;
  uint3 voxelGridSize;
  split_triangles_along_axis<0>(fullScene, gridSize, voxelGridSize.x);
  t2 = std::chrono::system_clock::now();
  elapsed_seconds = t2 - t1;
  std::cout << "Split 1 finished " << elapsed_seconds.count() << std::endl;
  std::cout << fullScene.materials.size() << " polygons on scene\n";
  t1 = t2;
  split_triangles_along_axis<1>(fullScene, gridSize, voxelGridSize.y);
  t2 = std::chrono::system_clock::now();
  elapsed_seconds = t2 - t1;
  std::cout << "Split 2 finished " << elapsed_seconds.count() << std::endl;
  std::cout << fullScene.materials.size() << " polygons on scene\n";
  t1 = t2;
  split_triangles_along_axis<2>(fullScene, gridSize, voxelGridSize.z);
  t2 = std::chrono::system_clock::now();
  elapsed_seconds = t2 - t1;
  std::cout << "Split 3 finished " << elapsed_seconds.count() << std::endl;
  std::cout << fullScene.materials.size() << " polygons on scene\n";
  t1 = t2;
  fullScene.compress();
  t2 = std::chrono::system_clock::now();
  elapsed_seconds = t2 - t1;
  std::cout << "Final scene compressed " << elapsed_seconds.count() << std::endl;
  std::cout << "Gird size: " << voxelGridSize.x << ' ' << voxelGridSize.y << ' ' << voxelGridSize.z << std::endl;

  struct Material
  {
    std::optional<float3> diffuse, emission;
    std::optional<uint32_t> texRef;
  };

  std::vector<Material> materials;
  for (const auto& materialColor : matColors) {
    if (materials.size() <= materialColor.first) {
      materials.resize(materialColor.first + 1);
    }
    materials[materialColor.first].diffuse = materialColor.second;
  }
  for (const auto& materialColor : matEmission) {
    if (materials.size() <= materialColor.first) {
      materials.resize(materialColor.first + 1);
    }
    materials[materialColor.first].emission = materialColor.second;
  }
  for (const auto tex : matTexture) {
    if (materials.size() <= tex.first) {
      materials.resize(tex.first + 1);
    }
    materials[tex.first].texRef = tex.second;
  }
  Scene triangles = PolygonsToTriangles(fullScene);
  triangles.compress();

  std::map<int, TexData> locTex = textures;

  allRemapLists = std::vector<int>();
  tableOffsetsAndSize = std::vector<int2>();
  meshes = std::map<int, Scene>();
  fullScene = Scene();
  matColors = std::map<int, float3>();
  matEmission = std::map<int, float3>();

  hrSceneLibraryClose();
  HRInitInfo initInfo;
  initInfo.vbSize = 1024 * 1024 * 128;
  initInfo.sortMaterialIndices = false;
  hrSceneLibraryOpen(L"Tessellated", HR_WRITE_DISCARD, initInfo);
  HRSceneInstRef scene = hrSceneCreate(L"Scene");
  hrSceneOpen(scene, HR_WRITE_DISCARD);

  auto cam = hrCameraCreate(L"Camera1");
  hrCameraOpen(cam, HR_WRITE_DISCARD);
  auto proxyCamNode = hrCameraParamNode(cam);
  for (auto it = camParams.begin(); it != camParams.end(); ++it) {
    proxyCamNode.append_child(it->first.c_str()).text().set(it->second.c_str());
  }
  hrCameraClose(cam);

  std::map<int, int> texturesRemap;
  for (auto tex : locTex) {
    texturesRemap[tex.first] = hrTexture2DCreateFromMemory(tex.second.w, tex.second.h, tex.second.bpp, tex.second.data.data()).id;
  }

  for (uint32_t i = 0; i < materials.size(); ++i) {
    std::wstringstream ss;
    ss << "Material" << i;
    auto matRef = hrMaterialCreate(ss.str().c_str());
    hrMaterialOpen(matRef, HR_WRITE_DISCARD);
    auto material = hrMaterialParamNode(matRef);
    if (materials[i].diffuse.has_value()) {
      auto diffuse = material.append_child();
      diffuse.set_name(L"diffuse");
      diffuse.append_attribute(L"brdf_type").set_value(L"lambert");
      auto color = diffuse.append_child();
      color.set_name(L"color");
      ss = std::wstringstream();
      ss << materials[i].diffuse.value().x << " " << materials[i].diffuse.value().y << ' ' << materials[i].diffuse.value().z;
      color.append_attribute(L"val").set_value(ss.str().c_str());
      if (materials[i].texRef.has_value()) {
        auto texRef = diffuse.append_child();
        texRef.set_name(L"texture");
        texRef.append_attribute(L"id").set_value(texturesRemap[materials[i].texRef.value()]);
        texRef.append_attribute(L"type").set_value(L"texref");
      }
    }
    if (materials[i].emission.has_value()) {
      auto emission = material.append_child();
      emission.set_name(L"emission");
      auto color = emission.append_child();
      color.set_name(L"color");
      ss = std::wstringstream();
      ss << materials[i].emission.value().x << " " << materials[i].emission.value().y << ' ' << materials[i].emission.value().z;
      color.append_attribute(L"val").set_value(ss.str().c_str());
    }
    hrMaterialClose(matRef);
  }

  std::ofstream fout(DataConfig::get().getBinFilePath(L"VoxelIds.bin"), std::ios::binary);

  uint32_t trianglesCount = static_cast<uint32_t>(triangles.voxelIds.size());
  fout.write(reinterpret_cast<char*>(&voxelGridSize), sizeof(voxelGridSize));
  fout.write(reinterpret_cast<char*>(&trianglesCount), sizeof(trianglesCount));
  fout.write(reinterpret_cast<char*>(&voxelSize), sizeof(voxelSize));
  for (uint32_t i = 0; i < trianglesCount; ++i) {
    fout.write(reinterpret_cast<char*>(&triangles.voxelIds[i]), sizeof(triangles.voxelIds[i]));
  }
  fout.close();

  auto mesh = hrMeshCreate(L"Whole scene");
  hrMeshOpen(mesh, HR_TRIANGLE_IND12, HR_WRITE_DISCARD);
  std::vector<uint32_t> indices;
  for (const auto& p: triangles.polygons) {
    indices.push_back(p[0]);
    indices.push_back(p[1]);
    indices.push_back(p[2]);
  }
  hrMeshPrimitiveAttribPointer1i(mesh, L"mind", reinterpret_cast<int*>(triangles.materials.data()));
  hrMeshVertexAttribPointer4f(mesh, L"positions", reinterpret_cast<float*>(triangles.positions.data()));
  hrMeshVertexAttribPointer4f(mesh, L"normals", reinterpret_cast<float*>(triangles.normals.data()));
  hrMeshVertexAttribPointer4f(mesh, L"tangent", reinterpret_cast<float*>(triangles.tangents.data()));
  hrMeshVertexAttribPointer2f(mesh, L"texcoord", reinterpret_cast<float*>(triangles.texCoords.data()));
  hrMeshAppendTriangles3(mesh, static_cast<int>(indices.size()), reinterpret_cast<int*>(indices.data()), false);
  hrMeshClose(mesh);

  std::array<float, 16> matrix = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };
  hrMeshInstance(scene, mesh, matrix.data());

  hrSceneClose(scene);
  hrFlush(scene);
}

using DrawFuncType = void (*)();
using InitFuncType = void (*)();

void window_main_voxel_tessellator(const std::wstring& a_libPath, const std::wstring& scene_name, float voxel_size) {
  voxelSize = voxel_size;
  hrErrorCallerPlace(L"Init");

  HRInitInfo initInfo;
  initInfo.vbSize = 1024 * 1024 * 128;
  initInfo.sortMaterialIndices = false;
  const std::wstring scenePath = a_libPath + scene_name + L"/scenelib";
  hrSceneLibraryOpen(scenePath.c_str(), HR_OPEN_EXISTING, initInfo);

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

  renderRef = hrRenderCreate(L"voxelTessellator");

  auto pList = hrRenderGetDeviceList(renderRef);

  hrRenderEnableDevice(renderRef, 0, true);

  hrCommit(scnRef, renderRef);
  hrSceneLibraryClose();
}
