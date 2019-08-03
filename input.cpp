#include "input.h"

Input::Input()
{
  enableOpenGL1    = true;
  noWindow         = false;
  exitStatus       = false;

  inputLibraryPath = L"D:/PROG/HydraAPI/main/tests_f/test_201";
  inputRenderName  = L"opengl1";


  camMoveSpeed     = 7.5f;
  mouseSensitivity = 0.1f;

  // dynamic data
  //
  pathTracingEnabled = false;
}


