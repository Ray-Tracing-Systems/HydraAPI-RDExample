// This is a personal academic project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

#include <array>
#include <optional>
#include <string>
#include <fstream>

#include <HydraVSGFExport.h>
#include <LiteMath.h>
#include "QuadsTessellator.h"
#include <hydra_api\pugixml.hpp>
#include <direct.h>
#include <codecvt>

using namespace HydraLiteMath;

template<int N>
struct Polygon {
  std::array<float4, N> points;
  std::array<float2, N> texCoords;
  std::optional<std::array<float4, N>> normal;
  std::optional<std::array<float4, N>> tangent;
  uint32_t materialId = 0;
};

using Triangle = Polygon<3>;
using Quad = Polygon<4>;

template <typename T>
T lerpSquare(T p1, T p2, T p3, T p4, float x, float y) {
  T x1_pos = lerp(p1, p2, x);
  T x2_pos = lerp(p3, p4, x);
  return lerp(x1_pos, x2_pos, y);
}

template<typename T>
void genQuad(std::vector<T>& res, std::array<T, 4>& points, int x, int y, float factor) {
  res.push_back(lerpSquare(points[0], points[1], points[3], points[2], x / factor, y / factor));
  res.push_back(lerpSquare(points[0], points[1], points[3], points[2], (x + 1) / factor, y / factor));
  res.push_back(lerpSquare(points[0], points[1], points[3], points[2], x / factor, (y + 1) / factor));
  res.push_back(lerpSquare(points[0], points[1], points[3], points[2], (x + 1) / factor, y / factor));
  res.push_back(lerpSquare(points[0], points[1], points[3], points[2], x / factor, (y + 1) / factor));
  res.push_back(lerpSquare(points[0], points[1], points[3], points[2], (x + 1) / factor, (y + 1) / factor));
}

std::wstring s2ws(const std::string& str)
{
  return std::wstring(str.begin(), str.end());
}

std::string ws2s(const std::wstring& wstr)
{
  return std::string(wstr.begin(), wstr.end());
}

HydraGeomData MeshTessellation(const std::wstring& input, const std::wstring& output, int tessFactor) {
  HydraGeomData inputGeom;
  inputGeom.read(input);

  std::vector<Triangle> triangles;
  const uint32_t* indices = inputGeom.getTriangleVertexIndicesArray();
  const float4* positions = reinterpret_cast<const float4*>(inputGeom.getVertexPositionsFloat4Array());
  const float4* normals = reinterpret_cast<const float4*>(inputGeom.getVertexNormalsFloat4Array());
  const float4* tangents = reinterpret_cast<const float4*>(inputGeom.getVertexTangentsFloat4Array());
  const float2* texCoords = reinterpret_cast<const float2*>(inputGeom.getVertexTexcoordFloat2Array());
  const uint32_t* materials = inputGeom.getTriangleMaterialIndicesArray();

  const bool hasTangent = inputGeom.getHeader().flags & HydraGeomData::HAS_TANGENT;
  const bool hasNormals = !(inputGeom.getHeader().flags & HydraGeomData::HAS_NO_NORMALS);

  for (uint32_t i = 0; i < inputGeom.getIndicesNumber(); i += 3) {
    Triangle tri;
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

  std::vector<Quad> quads;
  for (int i = 0; i < triangles.size(); ++i) {
    for (int j = i + 1; j < triangles.size(); ++j) {
      if (triangles[i].materialId != triangles[j].materialId) {
        continue;
      }
      const int t1Next = (cornerPoints[i] + 1) % 3;
      const int t1Prev = (cornerPoints[i] + 2) % 3;
      const int t2Next = (cornerPoints[j] + 1) % 3;
      const int t2Prev = (cornerPoints[j] + 2) % 3;
      const bool pos1Match = length3(triangles[i].points[t1Next] - triangles[j].points[t2Prev]) < 1e-3;
      const bool pos2Match = length3(triangles[i].points[t1Prev] - triangles[j].points[t2Next]) < 1e-3;
      const bool normal1Match = length3(triangles[i].normal.value()[t1Next] - triangles[j].normal.value()[t2Prev]) < 1e-3;
      const bool normal2Match = length3(triangles[i].normal.value()[t1Prev] - triangles[j].normal.value()[t2Next]) < 1e-3;
      if (pos1Match && pos2Match && normal1Match && normal2Match) {
        Quad q;
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

  std::vector<float4> positionsRes;
  std::vector<float2> texCoordsRes;
  std::vector<float4> normalsRes;
  std::vector<float4> tangentsRes;
  std::vector<uint32_t> materialsRes;
  for (int i = 0; i < quads.size(); ++i) {
    for (int x = 0; x < tessFactor; ++x) {
      for (int y = 0; y < tessFactor; ++y) {
        const float tess = static_cast<float>(tessFactor);
        genQuad(positionsRes, quads[i].points, x, y, tess);
        genQuad(texCoordsRes, quads[i].texCoords, x, y, tess);
        if (hasNormals) {
          genQuad(normalsRes, quads[i].normal.value(), x, y, tess);
        }
        if (hasTangent) {
          genQuad(tangentsRes, quads[i].tangent.value(), x, y, tess);
        }
        materialsRes.push_back(quads[i].materialId);
        materialsRes.push_back(quads[i].materialId);
      }
    }
  }

  std::vector<uint32_t> indicesRes;
  std::vector<float4> comprPositions;
  std::vector<float2> comprTexCoords;
  std::vector<float4> comprNormals;
  std::vector<float4> comprTangents;
  std::vector<int> comprMaterialId;

  for (int i = 0; i < positionsRes.size(); ++i) {
    bool found = false;
    for (int j = 0; j < comprPositions.size(); ++j) {
      const bool posMatch = length3(positionsRes[i] - comprPositions[j]) < 1e-3;
      const bool normMatch = !hasNormals || length3(normalsRes[i] - comprNormals[j]) < 1e-3;
      const bool texCoordMatch = length(texCoordsRes[i] - comprTexCoords[j]) < 1e-3;
      const bool tangMatch = !hasTangent || length3(tangentsRes[i] - comprTangents[j]) < 1e-3;
      const bool matMatch = comprMaterialId[j] == materialsRes[i / 3];
      if (!(matMatch && posMatch && normMatch && texCoordMatch && tangMatch)) {
        continue;
      }
      found = true;
      indicesRes.push_back(j);
      break;
    }
    if (found) {
      continue;
    }
    comprPositions.push_back(positionsRes[i]);
    if (hasNormals) {
      comprNormals.push_back(normalsRes[i]);
    }
    comprTexCoords.push_back(texCoordsRes[i]);
    if (hasTangent) {
      comprTangents.push_back(tangentsRes[i]);
    }
    comprMaterialId.push_back(materialsRes[i / 3]);
    indicesRes.push_back(static_cast<int>(comprPositions.size()) - 1);
  }

  HydraGeomData outputGeom;
  outputGeom.setData(static_cast<uint32_t>(comprPositions.size()), reinterpret_cast<float*>(comprPositions.data()),
    hasNormals ? reinterpret_cast<float*>(comprNormals.data()) : nullptr,
    hasTangent ? reinterpret_cast<float*>(comprTangents.data()) : nullptr,
    reinterpret_cast<float*>(comprTexCoords.data()),
    static_cast<uint32_t>(indicesRes.size()), indicesRes.data(), materialsRes.data());
  outputGeom.write(ws2s(output));
  return outputGeom;
}

int main(int argc, char **argv) {
  const std::string IN_FLAG("-in_dir"), OUT_FLAG("-out_dir"), TESS_FACTOR_FLAG("-tess");
  std::wstring input, output;
  int tessFactor = 2;
  for (int i = 0; i < argc; ++i) {
    if (argv[i] == IN_FLAG && i + 1 < argc) {
      input = s2ws(argv[i + 1]);
    }
    if (argv[i] == OUT_FLAG && i + 1 < argc) {
      output = s2ws(argv[i + 1]);
    }
    if (argv[i] == TESS_FACTOR_FLAG && i + 1 < argc) {
      tessFactor = std::atoi(argv[i + 1]);
    }
  }
  if (input.empty()) {
    throw "Set input file using -in option.";
  }

  pugi::xml_document doc;
  doc.load_file((input + L"/statex_00001.xml").c_str());
  pugi::xml_node geometryNode = doc.child(L"geometry_lib");
  pugi::xml_node resultGeometryNode;

  _mkdir(ws2s(output).c_str());
  _mkdir(ws2s(output + L"/data").c_str());

  for (auto& mesh : geometryNode) {
    const std::wstring loc = std::wstring(L"/") + mesh.attribute(L"loc").as_string();
    const std::wstring outputFilename = output + loc;
    HydraGeomData geomData = MeshTessellation(input + loc, outputFilename, tessFactor);

    HydraGeomData::Header header = geomData.getHeader();
    pugi::xml_node newMesh = mesh;
    newMesh.attribute(L"bytesize").set_value(header.fileSizeInBytes + sizeof(header));
    newMesh.attribute(L"vertNum").set_value(geomData.getVerticesNumber());
    newMesh.attribute(L"triNum").set_value(geomData.getIndicesNumber() / 3);
    unsigned long long offset = sizeof(header);
    unsigned long long bytesize = geomData.getVerticesNumber() * sizeof(float4);
    std::vector<unsigned long long> byteSizes = {
      geomData.getVerticesNumber() * sizeof(float4),
      geomData.getVerticesNumber() * sizeof(float2),
      geomData.getIndicesNumber() * sizeof(uint32_t),
      geomData.getIndicesNumber() / 3 * sizeof(uint32_t)
    };
    std::vector<std::wstring> fieldNames = {
      L"positions",
      L"texcoords",
      L"indices",
      L"matindices"
    };
    if (!(header.flags & geomData.HAS_NO_NORMALS)) {
      byteSizes.insert(byteSizes.begin() + 1, geomData.getVerticesNumber() * sizeof(float4));
      fieldNames.insert(fieldNames.begin() + 1, L"normals");
    }
    if (header.flags & geomData.HAS_TANGENT) {
      const int shift = ((header.flags & geomData.HAS_NO_NORMALS) ? 1 : 2);
      byteSizes.insert(byteSizes.begin() + shift, geomData.getVerticesNumber() * sizeof(float4));
      fieldNames.insert(fieldNames.begin() + shift, L"tangents");
    }
    for (int i = 0; i < fieldNames.size();) {
      pugi::xml_node pos = newMesh.child(fieldNames[i].c_str());
      pos.attribute(L"offset").set_value(offset);
      pos.attribute(L"bytesize").set_value(bytesize);
      offset += bytesize;
      i++;
      if (i < fieldNames.size()) {
        bytesize = byteSizes[i];
      }
    }
  }
  doc.save_file((output + L"/statex_00001.xml").c_str(), L"  ");

  return 0;
}
