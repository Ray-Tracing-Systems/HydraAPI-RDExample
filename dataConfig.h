#pragma once

#include <filesystem>

class DataConfig {
  std::wstring folder;
  DataConfig() = default;
  ~DataConfig() = default;
public:
  static const uint32_t FF_VERSION = 6;


  void init(const std::wstring& scene_name, uint32_t voxel_size) {
    const std::wstring scenesFolder = L"ScenesData/";
    if (!std::filesystem::exists(scenesFolder)) {
      std::filesystem::create_directory(scenesFolder);
    }
    const std::wstring sceneFolder = scenesFolder + scene_name;
    if (!std::filesystem::exists(sceneFolder)) {
      std::filesystem::create_directory(sceneFolder);
    }
    {
      std::wstringstream ss;
      ss << sceneFolder << "/" << voxel_size;
      folder = ss.str();
      if (!std::filesystem::exists(folder)) {
        std::filesystem::create_directory(folder);
      }
    }
  }

  static DataConfig& get() {
    static DataConfig instance;
    return instance;
  }

  std::wstring getBinFilePath(const std::wstring& filename) {
    return folder + L"/" + filename;
  }
};
