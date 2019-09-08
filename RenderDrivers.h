#ifndef RENDERDRIVERS_H
#define RENDERDRIVERS_H

#include "RenderDriverOpenGL1.h"
#include "RenderDriverVulkan.h"

IHRRenderDriver* CreateOpenGL1_RenderDriver();

void registerAllGL1Drivers();
void printAllAvailableDrivers();

IHRRenderDriver* CreateVulkan_RenderDriver();

void registerAllVulkanDrivers();

#endif //RENDERDRIVERS_H