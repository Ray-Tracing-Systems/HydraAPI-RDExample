#include <iostream>
#include <vector>
//#include <zconf.h>

#include "HydraAPI.h"

using pugi::xml_node;

///////////////////////////////////////////////////////////////////////////////////////////////////////////// just leave this it as it is :)
#include "HydraRenderDriverAPI.h"
#include "RenderDrivers.h"

IHRRenderDriver* CreateDriverRTE(const wchar_t* a_cfg) { return nullptr; }
/////////////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined WIN32
#include <windows.h> // for SetConsoleCtrlHandler
#else
#include <unistd.h>
#include <signal.h>
#endif

#include <GLFW/glfw3.h>
#if defined(WIN32)
#pragma comment(lib, "glfw3dll.lib")
#else
#endif

#include "dataConfig.h"

///////////////////////////////////////////////////////////////////////////////////////////////////////////// GLFW
///////////////////////////////////////////////////////////////////////////////////////////////////////////// GLFW

extern int g_width;
extern int g_height;
static GLFWwindow* g_window = nullptr;

using DrawFuncType = void (*)();
using InitFuncType = void (*)();

void window_main_free_look(const wchar_t* a_libPath, const wchar_t* a_renderName,
                           InitFuncType a_pInitFunc = nullptr, DrawFuncType a_pDrawFunc = nullptr);

void window_main_free_look_vulkan(const std::wstring& a_libPath, const std::wstring& scene_name,
                           InitFuncType a_pInitFunc = nullptr, DrawFuncType a_pDrawFunc = nullptr);

void window_main_ff_integrator(const std::wstring& a_libPath, const std::wstring& scene_name);
void window_main_voxel_tessellator(const std::wstring& a_libPath, const std::wstring& scene_name, float voxel_size);

///////////////////////////////////////////////////////////////////////////////////////////////////////////// GLFW
///////////////////////////////////////////////////////////////////////////////////////////////////////////// GLFW

void InfoCallBack(const wchar_t* message, const wchar_t* callerPlace, HR_SEVERITY_LEVEL a_level)
{
  if (a_level >= HR_SEVERITY_WARNING)
  {
    if (a_level == HR_SEVERITY_WARNING)
      std::wcerr << L"WARNING: " << callerPlace << L": " << message; // << std::endl;
    else
      std::wcerr << L"ERROR  : " << callerPlace << L": " << message; // << std::endl;
  }
}

void destroy()
{
  std::cout << "call destroy() --> hrSceneLibraryClose()" << std::endl;
  hrSceneLibraryClose();
}

#ifdef WIN32
BOOL WINAPI HandlerExit(_In_ DWORD fdwControl)
{
  exit(0);
  return TRUE;
}
#else
bool destroyedBySig = false;
void sig_handler(int signo)
{
  if(destroyedBySig)
    return;
  switch(signo)
  {
    case SIGINT : std::cerr << "\nmain_app, SIGINT";      break;
    case SIGABRT: std::cerr << "\nmain_app, SIGABRT";     break;
    case SIGILL : std::cerr << "\nmain_app, SIGINT";      break;
    case SIGTERM: std::cerr << "\nmain_app, SIGILL";      break;
    case SIGSEGV: std::cerr << "\nmain_app, SIGSEGV";     break;
    case SIGFPE : std::cerr << "\nmain_app, SIGFPE";      break;
    default     : std::cerr << "\nmain_app, SIG_UNKNOWN"; break;
    break;
  }
  std::cerr << " --> hrSceneLibraryClose()" << std::endl;
  hrSceneLibraryClose();
  destroyedBySig = true;
}
#endif


int main(int argc, const char** argv)
{
  registerAllVulkanDrivers();
  registerAllGL1Drivers();
  registerAllFFIntegratorDrivers();
  registerAllVoxelTessellatorDrivers();

  hrInfoCallback(&InfoCallBack);
  hrErrorCallerPlace(L"main");  // for debug needs only

  std::string workingDir = "..";
  if(argc > 1)
    workingDir = std::string(argv[1]);

  atexit(&destroy);                           // if application will terminated you have to call hrSceneLibraryClose to free all connections with hydra.exe
#if defined WIN32
  SetConsoleCtrlHandler(&HandlerExit, TRUE);  // if some one kill console :)
  wchar_t NPath[512];
  GetCurrentDirectoryW(512, NPath);
#ifdef NEED_DIR_CHANGE
  SetCurrentDirectoryW(L"../../main");
#endif
  std::wcout << L"[main]: curr_dir = " << NPath << std::endl;
#else

  if(chdir(workingDir.c_str()) != 0)
    std::cout << "[main]: chdir have failed for some reason ... " << std::endl;

  char cwd[1024];
  if (getcwd(cwd, sizeof(cwd)) != nullptr)
    std::cout << "[main]: curr_dir = " << cwd <<std::endl;
  else
    std::cout << "getcwd() error" <<std::endl;

  {
    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = sig_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = SA_RESETHAND;
    sigaction(SIGINT,  &sigIntHandler, NULL);
    sigaction(SIGSTOP, &sigIntHandler, NULL);
    sigaction(SIGABRT, &sigIntHandler, NULL);
    sigaction(SIGILL,  &sigIntHandler, NULL);
    sigaction(SIGTERM, &sigIntHandler, NULL);
    sigaction(SIGSEGV, &sigIntHandler, NULL);
    sigaction(SIGFPE,  &sigIntHandler, NULL);
  }
#endif
  
  std::cout << "sizeof(size_t) = " << sizeof(size_t) <<std::endl;

  float voxelSize = 0.5 / 8;
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "-voxelSize") == 0 && i + 1 < argc) {
      voxelSize = std::stof(argv[i + 1]);
    }
  }

  const uint32_t scaledVoxelSize = static_cast<uint32_t>(voxelSize * 10000);

  //window_main_ff_integrator(L"../Diser/DiffuseReference/01_CornellBoxEmpty/tessellated", L"ff_integrator", recomputeFF, noInterpolation);
  //window_main_ff_integrator(L"../Diser/DiffuseReference/02_CornellBoxWithPrim/tessellated", L"ff_integrator", recomputeFF, noInterpolation);
  //window_main_free_look_vulkan(L"../Diser/DiffuseReference/02_CornellBoxWithPrim/tessellated", L"vulkan");
  //window_main_free_look_vulkan(L"../Diser/DiffuseReference/01_CornellBoxEmpty/tessellated", L"vulkan");
  //window_main_free_look_vulkan(L"../Diser/DiffuseReference/03_CornellBoxWithComplexShape/scenelib", L"vulkan");
  //window_main_free_look_vulkan(L"../Diser/DiffuseReference/04_CornellBoxWithSeveralComplexShape/scenelib", L"vulkan");
  //window_main_free_look_vulkan(L"../Diser/DiffuseReference/05_Sponza/scenelib", L"vulkan");
  //window_main_free_look_vulkan(L"../Diser/DiffuseReference/06_Indoor/scenelib", L"vulkan");
  //window_main_free_look_vulkan(L"../Diser/DiffuseReference/07_Outdoor/scenelib", L"vulkan");
  //window_main_free_look_vulkan(L"../Diser/DiffuseReference/08_BigScene/scenelib", L"vulkan");
  const std::wstring scene = L"01_CornellBoxEmpty"; // 0.5 / 8
  //const std::wstring scene = L"02_CornellBoxWithPrim"; // 0.5 / 8
  //const std::wstring scene = L"SimpleInterior_500tris"; // 0.5 / 4
  //const std::wstring scene = L"SimpleInterior_5K"; // 0.5 / 2
  //const std::wstring scene = L"05_Sponza";
  //const std::wstring scene = L"test_42"; // 0.5
  DataConfig::get().init(scene, scaledVoxelSize);
  window_main_voxel_tessellator(L"../Diser/DiffuseReference/", scene, voxelSize);
  //window_main_ff_integrator(L"Tessellated", scene);
  window_main_ff_integrator(L"../Diser/DiffuseReference/", scene);
  window_main_free_look_vulkan(L"Tessellated", scene);
  //window_main_free_look_vulkan(L"GI_res", scene);
  //window_main_free_look_vulkan(L"Tessellated", L"vulkan");
  //window_main_free_look_vulkan(L"../Diser/DiffuseReference/test_42/scenelib", L"vulkan");

  hrErrorCallerPlace(L"main"); // for debug needs only

  hrSceneLibraryClose();

  return 0;
}
