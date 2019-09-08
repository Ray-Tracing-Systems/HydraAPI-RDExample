#pragma once

#include <string>
#include <unordered_map>

struct Input
{
  Input();

  // fixed data
  //
  bool noWindow;
  bool exitStatus;
  bool enableOpenGL1;
  bool enableVulkan;

  std::wstring inputLibraryPath;
  std::wstring inputRenderName;

  // mouse and keyboad/oher gui input
  //
  float camMoveSpeed;
  float mouseSensitivity;

  // dynamic data
  //

  bool pathTracingEnabled;
};
