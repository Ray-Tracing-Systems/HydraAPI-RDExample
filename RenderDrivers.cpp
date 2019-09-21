#include "RenderDrivers.h"

IHRRenderDriver* CreateOpenGL1Debug_RenderDriver();

void registerGL1PlainDriver()
{
  HRDriverInfo plain_gl1_info;
  plain_gl1_info.supportHDRFrameBuffer        = false;
  plain_gl1_info.supportHDRTextures           = false;
  plain_gl1_info.supportMultiMaterialInstance = false;

  plain_gl1_info.supportImageLoadFromInternalFormat = false;
  plain_gl1_info.supportImageLoadFromExternalFormat = false;
  plain_gl1_info.supportMeshLoadFromInternalFormat  = false;
  plain_gl1_info.supportLighting                    = false;

  plain_gl1_info.memTotal = int64_t(8) * int64_t(1024 * 1024 * 1024);

  plain_gl1_info.driverName = L"opengl1";
  plain_gl1_info.createFunction = CreateOpenGL1_RenderDriver;

  RenderDriverFactory::Register(L"opengl1", plain_gl1_info);
}

void registerGL1DebugDriver()
{
  HRDriverInfo plain_gl1_info;
  plain_gl1_info.supportHDRFrameBuffer = false;
  plain_gl1_info.supportHDRTextures = false;
  plain_gl1_info.supportMultiMaterialInstance = false;

  plain_gl1_info.supportImageLoadFromInternalFormat = false;
  plain_gl1_info.supportImageLoadFromExternalFormat = false;
  plain_gl1_info.supportMeshLoadFromInternalFormat = false;
  plain_gl1_info.supportLighting = false;

  plain_gl1_info.memTotal = int64_t(8) * int64_t(1024 * 1024 * 1024);

  plain_gl1_info.driverName = L"opengl1Debug";
  plain_gl1_info.createFunction = CreateOpenGL1Debug_RenderDriver;

  RenderDriverFactory::Register(L"opengl1Debug", plain_gl1_info);
}

void registerAllGL1Drivers()
{
  registerGL1PlainDriver();
  registerGL1DebugDriver();
}

void printAllAvailableDrivers()
{
  auto drivers = RenderDriverFactory::GetListOfRegisteredDrivers();

  std::cout << "Render drivers currently registered in Hydra API: " << std::endl;
  for(const auto &d :drivers)
  {
    if(!d.empty())
    {
      auto tmp = ws2s(d);
      std::cout << tmp.c_str() << std::endl;
    }
  }

}

