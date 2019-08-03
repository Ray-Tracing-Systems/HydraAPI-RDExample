#ifndef RENDERDRIVERS_H
#define RENDERDRIVERS_H

#include "RenderDriverOpenGL1.h"

IHRRenderDriver* CreateOpenGL1_RenderDriver();

void registerAllGL1Drivers();
void printAllAvailableDrivers();

#endif //RENDERDRIVERS_H