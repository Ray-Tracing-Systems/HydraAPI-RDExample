#pragma once

#include <filesystem>

class DataConfig {
  std::wstring folder;
  std::wstring sceneName;
  uint32_t voxelSize = 0;
  bool useExposure = false;
  DataConfig() = default;
  ~DataConfig() = default;
public:
  static const uint32_t FF_VERSION = 8;
  static const uint32_t MAX_VIRTUAL_PATCHES = 4;


  void init(int argc, const char **argv, const std::wstring& scene_name, uint32_t voxel_size) {
    sceneName = scene_name;
    voxelSize = voxel_size;

    if (argc > 1)
    {
      for (int i = 1; i < argc; ++i) {
        if (argv[i] == std::string("-e")) {
          useExposure = true;
        } else if (argv[i] == std::string("-scene")) {
          std::wstringstream ss;
          ss << argv[i + 1];
          sceneName = ss.str();
        }
      }
    }

    const std::wstring scenesFolder = L"ScenesData/";
    if (!std::filesystem::exists(scenesFolder)) {
      std::filesystem::create_directory(scenesFolder);
    }
    const std::wstring sceneFolder = scenesFolder + sceneName;
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

  std::wstring getBinFilePath(const std::wstring& filename) const {
    return folder + L"/" + filename;
  }

  static std::wstring getScreenShotPrefix() {
    std::wstringstream ss;
    ss << get().sceneName << "_" << get().voxelSize;
    return ss.str();
  }

  static std::wstring getSceneName() {
    return get().sceneName;
  }

  bool hasExposure() const { return useExposure; };
};
