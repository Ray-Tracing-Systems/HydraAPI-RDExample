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

void registerVulkanPlainDriver()
{
  HRDriverInfo vulkan_info;
  vulkan_info.supportHDRFrameBuffer = false;
  vulkan_info.supportHDRTextures = false;
  vulkan_info.supportMultiMaterialInstance = false;

  vulkan_info.supportImageLoadFromInternalFormat = false;
  vulkan_info.supportImageLoadFromExternalFormat = false;
  vulkan_info.supportMeshLoadFromInternalFormat = false;
  vulkan_info.supportLighting = false;

  vulkan_info.memTotal = int64_t(8) * int64_t(1024 * 1024 * 1024);

  vulkan_info.driverName = L"vulkan";
  vulkan_info.createFunction = CreateVulkan_RenderDriver;

  RenderDriverFactory::Register(L"vulkan", vulkan_info);
}

void registerAllVulkanDrivers()
{
  registerVulkanPlainDriver();
}

void registerFFIntegratorDrivers()
{
  HRDriverInfo integrator_info;
  integrator_info.supportHDRFrameBuffer = false;
  integrator_info.supportHDRTextures = false;
  integrator_info.supportMultiMaterialInstance = false;

  integrator_info.supportImageLoadFromInternalFormat = false;
  integrator_info.supportImageLoadFromExternalFormat = false;
  integrator_info.supportMeshLoadFromInternalFormat = false;
  integrator_info.supportLighting = false;

  integrator_info.memTotal = int64_t(8) * int64_t(1024 * 1024 * 1024);

  integrator_info.driverName = L"ff_integrator";
  integrator_info.createFunction = CreateFFIntegrator_RenderDriver;

  RenderDriverFactory::Register(L"ff_integrator", integrator_info);
}

void registerAllFFIntegratorDrivers()
{
  registerFFIntegratorDrivers();
}

void registerVoxelTessellatorDrivers()
{
  HRDriverInfo integrator_info;
  integrator_info.supportHDRFrameBuffer = false;
  integrator_info.supportHDRTextures = false;
  integrator_info.supportMultiMaterialInstance = false;

  integrator_info.supportImageLoadFromInternalFormat = false;
  integrator_info.supportImageLoadFromExternalFormat = false;
  integrator_info.supportMeshLoadFromInternalFormat = false;
  integrator_info.supportLighting = false;

  integrator_info.memTotal = int64_t(8) * int64_t(1024 * 1024 * 1024);

  integrator_info.driverName = L"voxelTessellator";
  integrator_info.createFunction = CreateVoxelTessellator_RenderDriver;

  RenderDriverFactory::Register(L"voxelTessellator", integrator_info);
}

void registerAllVoxelTessellatorDrivers()
{
  registerVoxelTessellatorDrivers();
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

