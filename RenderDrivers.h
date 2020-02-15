#ifndef RENDERDRIVERS_H
#define RENDERDRIVERS_H

#include "RenderDriverOpenGL1.h"
#include "RenderDriverVulkan.h"
#include "RenderDriverFormFactorIntegrator.h"
#include "RenderDriverVoxelTessellator.h"

IHRRenderDriver* CreateOpenGL1_RenderDriver();

void registerAllGL1Drivers();
void printAllAvailableDrivers();

IHRRenderDriver* CreateVulkan_RenderDriver();

void registerAllVulkanDrivers();

void registerAllFFIntegratorDrivers();

IHRRenderDriver* CreateFFIntegrator_RenderDriver();

void registerAllVoxelTessellatorDrivers();

IHRRenderDriver* CreateVoxelTessellator_RenderDriver();

#endif //RENDERDRIVERS_H